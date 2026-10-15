from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np
from scipy.special import logsumexp

# from shangrla.core.Audit import Audit


@dataclass
class EProcess:
    """
    A class representing a stochastic process with a time series of values on a log scale.

    Attributes
    ----------
    _values: np.ndarray
        The time series values (log-scale).
    running_max: float
        The maximum value seen so far in the time series (used to compute min p-value).
    """
    _values: np.ndarray = field(default_factory=lambda: np.array([]))
    running_max: float = -np.inf

    def __post_init__(self):
        """Update running_max after initialization if initial values are present."""
        if len(self._values) > 0:
            self.running_max = np.max(self._values)

    def __getitem__(self, item):
        """Return element(s) from the underlying numpy array.

        Raises
        ------
        IndexError
            If an invalid index is requested.
        """
        try:
            return self._values[item]
        except IndexError:
            raise IndexError(f"Index {item} out of bounds for EProcess with shape {self._values.shape}")

    def __len__(self):
        """Return the length of the process (number of timesteps)."""
        return len(self._values)

    def append(self, values: np.ndarray):
        """Append new values to the end of the time series and update running_max.

        Parameters
        ----------
        values : np.ndarray
            Values to append (will be converted to numpy array).
        """
        values = np.asarray(values)
        if values.size == 0:
            return
        self._values = np.append(self._values, values)
        self.running_max = max(self.running_max, np.max(values))

    def append_at(self, values: np.ndarray, start: int):
        """Replace the suffix of the time series starting at `start` with the provided
        values and optionally extend the time series if more values are provided.

        This method requires that the provided `values` array be at least as long as
        the suffix being replaced (i.e., len(values) + start >= len(self._values)).

        Parameters
        ----------
        values : np.ndarray
            Values to insert starting at `start` index.
        start : int
            Index in the current time series where replacement/extension begins.

        Raises
        ------
        ValueError
            If `start` is out of bounds or `values` is too short to cover the existing suffix.
        """
        values = np.asarray(values)

        if start > len(self._values):
            raise ValueError(f"Index {start} out of bounds for EProcess with shape {self._values.shape}")
        if len(values) + start < len(self._values):
            raise ValueError(
                f"Not enough values to append at index {start}: need at least {len(self._values) - start} values, got {len(values)}"
            )

        required = len(self._values) - start
        # Replace suffix
        if required > 0:
            self._values[start:] = values[:required]

        remainder = values[required:]
        if remainder.size > 0:
            self._values = np.append(self._values, remainder)

        # append_at can replace the previous maximum, so recompute it from
        # the complete updated process rather than only considering a suffix.
        self.running_max = np.max(self._values) if self._values.size > 0 else -np.inf

    def p_history(self):
        """Return the p-values corresponding to the stored log-values (1/exp(values))."""
        return 1 / np.exp(self._values)

    def p_value(self):
        """Return the latest p-value (based on the last stored log-value).

        Raises
        ------
        ValueError
            If the process has no values.
        """
        if len(self._values) == 0:
            raise ValueError("Cannot get p_value of empty EProcess")
        return 1 / np.exp(self._values[-1])

    def min_p_value(self):
        """Return the smallest p-value observed so far (based on running_max).

        Raises
        ------
        ValueError
            If no values have been added yet.
        """
        if self.running_max == -np.inf:
            raise ValueError("No values have been added to this EProcess yet")
        return 1 / np.exp(self.running_max)


class Combiner(ABC):
    """
    Base class for combiners that merge values from multiple child e-processes into a
    single log-increment for a composite process.

    Subclasses must implement operate(values: np.ndarray) -> float which computes the
    log-increment for a single timestep. The Combiner class manages history rows used
    by some combiners.
    """

    def __init__(self, history_length: int = 1):
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        self.history: Optional[np.ndarray] = None
        self.history_length = history_length

    def __call__(self, values: np.ndarray) -> float:
        """Compute the combiner's increment for a row of `values`, maintaining history.

        If history is uninitialized, it is seeded with a zero vector of the appropriate
        length (this represents an implicit starting state of log(1) = 0 for each child).
        """
        if self.history is None:
            # Initialize with zeros for equal weights
            self.history_append(np.zeros(len(values)))
        combination = self.operate(values)
        self.history_append(values)
        return combination

    def history_append(self, values: np.ndarray):
        """Append a row to the history buffer (truncate to history_length)."""
        if self.history is None:
            self.history = np.array([values])
        else:
            self.history = np.append(self.history, [values], axis=0)
            if len(self.history) > self.history_length:
                self.history = self.history[-self.history_length:]

    def set_history(self, values: np.ndarray):
        """Set the history buffer explicitly (values will be truncated to history_length)."""
        self.history = values[: self.history_length]

    def get_start_with_history(self, start: int) -> int:
        """Return an adjusted start index that includes any previous rows required by history."""
        return max(0, start - self.history_length)

    @abstractmethod
    def operate(self, values: np.ndarray) -> float:
        """Compute the log-increment for a single timestep given `values`.

        Parameters
        ----------
        values : np.ndarray
            1-D array of the current values from each child e-process.
        """
        # values[i] is [eproc1[i], eproc2[i], ...]
        pass


class Minimum(Combiner):
    """Combiner that takes the minimum across child values.

    The increment is computed as: min(current_values) - min(previous_values)
    """

    def operate(self, values: np.ndarray) -> float:
        return float(np.min(values) - np.min(self.history[-1]))


class MinOfRunningMax(Combiner):
    """Combiner that tracks a running maximum for each child and returns the
    increment in the minimum across those running maxima.
    """

    def __init__(self, history_length: int = 1):
        super().__init__(history_length)
        self.running_max: Optional[np.ndarray] = None

    def __call__(self, values: np.ndarray) -> float:
        if self.running_max is None:
            # Start from implicit zero log-value for each child
            self.running_max = np.zeros(len(values))
        return self.operate(values)

    def running_max_update(self, values: np.ndarray):
        """Update the per-child running maximums with the provided values.

        Raises
        ------
        ValueError
            If running_max is not initialized.
        """
        if self.running_max is None:
            raise ValueError("running_max not initialized")
        self.running_max = np.maximum(self.running_max, values)

    def operate(self, values: np.ndarray) -> float:
        min_running_max_old = np.min(self.running_max)
        self.running_max_update(values)
        min_running_max_new = np.min(self.running_max)
        return float(min_running_max_new - min_running_max_old)


class Linear(Combiner):
    """Combiner that uses log-sum-exp to combine child log-values.

    The increment is: logsumexp(values) - logsumexp(previous_values)
    """

    def operate(self, values: np.ndarray) -> float:
        return float(logsumexp(values) - logsumexp(self.history[-1]))


class Quadratic(Combiner):
    """Quadratic-weighted combiner using log-sum-exp.

    The increment is: logsumexp(increments + prev^2) - logsumexp(prev^2)
    where increments = values - previous_values.
    """

    def operate(self, values: np.ndarray) -> float:
        increments = values - self.history[-1]
        return float(logsumexp(increments + (self.history[-1]) ** 2) - logsumexp((self.history[-1]) ** 2))


class Largest(Combiner):
    """Combiner that follows the largest previous-weighted child.

    It selects the index of the maximum value in the previous-history row and
    returns the increment for that index (values[k] - prev[k]). Ties are broken by
    first occurrence due to numpy.argmax semantics.
    """

    def operate(self, values: np.ndarray) -> float:
        prev = self.history[-1]
        k = int(np.argmax(prev))
        increments = values - prev
        return float(increments[k])


# Node classes
@dataclass
class Node(ABC):
    """Abstract base class for nodes in the composition tree.

    Each Node holds an EProcess and zero or more parent CompositeNodes.
    LeafNode and CompositeNode extend this class with concrete behavior.
    """

    id: str
    eprocess: EProcess = field(default_factory=lambda: EProcess())
    parents: Optional[List['CompositeNode']] = None

    def _add_parent(self, parent: 'CompositeNode'):
        """Register a parent composite node (idempotent)."""
        if self.parents is None:
            self.parents = []
        if parent not in self.parents:
            self.parents.append(parent)

    def _remove_parent(self, parent: 'CompositeNode'):
        """Remove a parent from this node's parent list (no-op if not present)."""
        if self.parents is not None and parent in self.parents:
            self.parents.remove(parent)

    def get_p_value(self):
        """Return the p-value corresponding to the most recent log-value in the eprocess."""
        return 1 / np.exp(self.eprocess[-1])

    def get_min_p_value(self):
        """Return the minimum p-value observed for this node (from running_max)."""
        return 1 / np.exp(self.eprocess.running_max)

    def __len__(self):
        return len(self.eprocess)

    @abstractmethod
    def __getitem__(self, item):
        pass


@dataclass
class LeafNode(Node):
    """A leaf node that directly wraps an EProcess.

    Accessing the leaf with index/slice returns values from the wrapped EProcess.
    """

    def __getitem__(self, item):
        return self.eprocess[item]


@dataclass
class CompositeNode(Node):
    """Node that composes several child nodes using a Combiner.

    Attributes
    ----------
    children : List[Node]
        The child nodes whose eprocess values will be combined.
    combiner : Combiner
        The combiner instance used to collapse child values into a single increment.
    margin : Optional[float]
        Optional margin value (application-specific; may be set later).
    """

    children: List[Node] = field(default_factory=list)
    combiner: Combiner = field(default_factory=Minimum)
    margin: Optional[float] = None

    def __post_init__(self):
        # register as parent for existing children
        for c in list(self.children):
            if not isinstance(c, Node):
                raise TypeError("CompositeNode children must be Node instances")
            c._add_parent(self)
        if self.combiner is None:
            self.combiner = Minimum()

    def __getitem__(self, item: int | slice):
        """Access the composite's eprocess; compute missing indices on demand.

        If the requested index (or slice stop) is beyond the currently computed
        values, combine() is invoked to compute up to the requested end.
        """
        if isinstance(item, slice):
            stop = item.stop
            if stop is not None and stop >= len(self.eprocess):
                self.combine(start=len(self.eprocess), end=stop)
        else:
            if item >= len(self.eprocess):
                self.combine(start=len(self.eprocess), end=item + 1)
        return self.eprocess[item]

    def add_child(self, child: Node):
        """Add a child node to this composite and register parent pointer.

        This method is idempotent for existing children and validates the child's type.
        """
        if child is None:
            raise ValueError("child cannot be None")
        if not isinstance(child, Node):
            raise TypeError("child must be a Node")
        if child in self.children:
            return
        self.children.append(child)
        child._add_parent(self)

    def combine(self, start: int = 0, end: int = -1) -> None:
        """Combine child eprocess values into this composite's eprocess for
        indices start <= i < end.

        The method is defensive: it handles missing children, empty ranges, and
        respects the combiner's history_length when slicing child values.
        """
        assert start >= 0
        if not self.children:
            raise RuntimeError("CompositeNode has no children to combine")

        # Determine end index if -1 (compute up to the shortest child length)
        if end == -1:
            end = min(len(c.eprocess) for c in self.children)

        if end <= start:
            return

        hist_start = self.combiner.get_start_with_history(start)
        hist_len = start - hist_start

        # collate values for rows hist_start..end-1 from each child
        transposed_list = [child[hist_start:end] for child in self.children]
        transposed = np.stack(transposed_list, axis=1)

        # set the history of the combiner if historic rows are available
        if hist_len > 0:
            self.combiner.set_history(transposed[:hist_len])

        # remove the history rows before applying the combiner
        if self.combiner.history_length > 0:
            transposed_work = transposed[self.combiner.history_length :]
        else:
            transposed_work = transposed

        if transposed_work.size == 0:
            return

        increments = np.array(list(map(self.combiner, transposed_work)))

        values = np.cumsum(increments)
        if start > 0 and len(self.eprocess) >= start:
            values = values + self.eprocess[start - 1]
        self.eprocess.append_at(values, start)


class NodeRegistry:
    """Registry for creating and retrieving nodes by identifier.

    Provides convenience methods for creating leaf and composite nodes and
    ensures consistency when replacing existing nodes.
    """

    def __init__(self):
        self.nodes: dict = {}
        self.root = CompositeNode('root', combiner=Minimum())

    def __getitem__(self, item):
        """Return an existing node by id or create a new LeafNode if missing."""
        if item in self.nodes:
            return self.nodes[item]
        node = LeafNode(item)
        self.nodes[item] = node
        return node

    def __setitem__(self, key, value):
        """Set/replace a node. When replacing, detach the old node from its parents."""
        if not isinstance(value, Node):
            raise TypeError("value must be a Node")
        if key in self.nodes:
            old = self.nodes[key]
            if old.parents:
                for p in list(old.parents):
                    if old in p.children:
                        p.children.remove(old)
        self.nodes[key] = value

    def create_leaf(self, ident: str) -> LeafNode:
        """Create a new leaf node or return an existing leaf with the same ident.

        Raises
        ------
        ValueError
            If a composite node with the same ident already exists.
        """
        if ident in self.nodes:
            if isinstance(self.nodes[ident], LeafNode):
                return self.nodes[ident]
            raise ValueError(f"Node {ident} already exists as a composite node")
        node = LeafNode(ident)
        self.nodes[ident] = node
        return node

    def create_composite(self, ident: str, combiner: Optional[Combiner] = None) -> 'CompositeNode':
        """Create a new composite node or return an existing composite with the same ident.

        Raises
        ------
        ValueError
            If a leaf node with the same ident already exists.
        """
        if ident in self.nodes:
            if isinstance(self.nodes[ident], CompositeNode):
                return self.nodes[ident]
            raise ValueError(f"Node {ident} already exists as a leaf node")
        node = CompositeNode(ident, combiner=(combiner if combiner is not None else Minimum()))
        self.nodes[ident] = node
        return node

    def get_composite_node(self, ident: str) -> 'CompositeNode':
        """Backwards-compatible helper that returns (or creates) a CompositeNode."""
        return self.create_composite(ident)

    def get_leaf_node(self, ident: str) -> tuple[LeafNode, bool]:
        """Return a leaf node and a boolean indicating whether it was created.

        The boolean is True if a new LeafNode was created, and False if an existing
        one was returned.
        """
        if ident in self.nodes:
            if isinstance(self.nodes[ident], LeafNode):
                return self.nodes[ident], False
            raise ValueError(f"Node {ident} already exists as a composite node")
        node = self.create_leaf(ident)
        return node, True

    def set_root(self, root: Node):
        """Set the registry root node; must be a CompositeNode."""
        if not isinstance(root, CompositeNode):
            raise TypeError("root must be a CompositeNode")
        self.root = root

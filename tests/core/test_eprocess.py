import pytest
import numpy as np
from shangrla.core.eprocess import (
    EProcess, Combiner, Minimum, MinOfRunningMax, Linear, Quadratic
)


class TestEProcess:
    """Test suite for EProcess class"""

    def test_empty_initialization(self):
        """Test creating an empty EProcess"""
        eproc = EProcess()
        assert len(eproc) == 0
        assert eproc.running_max == -np.inf

    def test_initialization_with_values(self):
        """Test creating an EProcess with initial values"""
        values = np.array([0, 1.5, 2.3, 3.1])
        eproc = EProcess(_values=values)
        assert len(eproc) == 4
        assert eproc.running_max == 3.1
        np.testing.assert_array_equal(eproc._values, values)

    def test_initialization_with_negative_values(self):
        """Test EProcess correctly tracks max with negative values"""
        values = np.array([-5, -2, -10, -1])
        eproc = EProcess(_values=values)
        assert eproc.running_max == -1

    def test_getitem_single_index(self):
        """Test accessing single element by index"""
        values = np.array([0, 1.5, 2.3, 3.1])
        eproc = EProcess(_values=values)
        assert eproc[0] == 0
        assert eproc[2] == 2.3
        assert eproc[-1] == 3.1

    def test_getitem_out_of_bounds(self):
        """Test that out of bounds access raises IndexError"""
        eproc = EProcess(_values=np.array([1, 2, 3]))
        with pytest.raises(IndexError):
            _ = eproc[10]

    def test_getitem_slice(self):
        """Test slicing EProcess"""
        values = np.array([0, 1.5, 2.3, 3.1])
        eproc = EProcess(_values=values)
        sliced = eproc[1:3]
        np.testing.assert_array_equal(sliced, np.array([1.5, 2.3]))

    def test_len(self):
        """Test length of EProcess"""
        eproc = EProcess()
        assert len(eproc) == 0
        eproc.append(np.array([1, 2]))
        assert len(eproc) == 2

    def test_append_single_value(self):
        """Test appending a single value"""
        eproc = EProcess(_values=np.array([0, 1]))
        eproc.append(np.array([2]))
        assert len(eproc) == 3
        assert eproc[-1] == 2
        np.testing.assert_array_equal(eproc._values, np.array([0, 1, 2]))

    def test_append_multiple_values(self):
        """Test appending multiple values"""
        eproc = EProcess(_values=np.array([0, 1]))
        eproc.append(np.array([2, 3, 4]))
        assert len(eproc) == 5
        np.testing.assert_array_equal(eproc._values, np.array([0, 1, 2, 3, 4]))

    def test_append_updates_running_max(self):
        """Test that append updates running_max"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        assert eproc.running_max == 2
        eproc.append(np.array([0.5]))  # Value smaller than max
        assert eproc.running_max == 2  # Max should remain unchanged
        eproc.append(np.array([5]))  # Value larger than max
        assert eproc.running_max == 5  # Max should update

    def test_append_to_empty_eprocess(self):
        """Test appending to an initially empty EProcess"""
        eproc = EProcess()
        assert eproc.running_max == -np.inf
        eproc.append(np.array([1, 2, 3]))
        assert eproc.running_max == 3
        assert len(eproc) == 3

    def test_append_at_basic(self):
        """Test append_at with simple case"""
        eproc = EProcess(_values=np.array([0, 1, 2, 3]))
        eproc.append_at(np.array([10, 11, 12]), start=1)
        np.testing.assert_array_equal(eproc._values, np.array([0, 10, 11, 12]))

    def test_append_at_at_start(self):
        """Test append_at at index 0"""
        eproc = EProcess(_values=np.array([1, 2, 3]))
        eproc.append_at(np.array([10, 11, 12]), start=0)
        np.testing.assert_array_equal(eproc._values, np.array([10, 11, 12]))

    def test_append_at_at_end_with_extension(self):
        """Test append_at at end of array that extends it"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        eproc.append_at(np.array([10, 11, 12]), start=2)
        np.testing.assert_array_equal(eproc._values, np.array([0, 1, 10, 11, 12]))

    def test_append_at_exact_fit(self):
        """Test append_at when values fit exactly"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        eproc.append_at(np.array([10, 11]), start=1)
        np.testing.assert_array_equal(eproc._values, np.array([0, 10, 11]))

    def test_append_at_out_of_bounds_start(self):
        """Test append_at with start index out of bounds"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        with pytest.raises(ValueError):
            eproc.append_at(np.array([10, 11]), start=5)

    def test_append_at_not_enough_values(self):
        """Test append_at with insufficient values"""
        eproc = EProcess(_values=np.array([0, 1, 2, 3]))
        with pytest.raises(ValueError):
            eproc.append_at(np.array([10]), start=2)  # Need at least 2 values

    def test_append_at_updates_running_max(self):
        """Test that append_at updates running_max correctly"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        assert eproc.running_max == 2
        eproc.append_at(np.array([5, 6]), start=1)
        assert eproc.running_max == 6

    def test_p_history(self):
        """Test p_history calculation"""
        values = np.array([0, 1, 2])
        eproc = EProcess(_values=values)
        p_vals = eproc.p_history()
        expected = 1 / np.exp(values)
        np.testing.assert_array_almost_equal(p_vals, expected)

    def test_p_value_latest(self):
        """Test p_value returns the latest value"""
        eproc = EProcess(_values=np.array([0, 1, 2]))
        p_val = eproc.p_value()
        assert p_val == pytest.approx(1 / np.exp(2))

    def test_p_value_empty_raises(self):
        """Test p_value raises on empty EProcess"""
        eproc = EProcess()
        with pytest.raises(ValueError):
            _ = eproc.p_value()

    def test_min_p_value(self):
        """Test min_p_value uses running_max"""
        eproc = EProcess(_values=np.array([0, -5, 3]))  # running_max = 3
        min_p = eproc.min_p_value()
        assert min_p == pytest.approx(1 / np.exp(3))

    def test_min_p_value_empty_raises(self):
        """Test min_p_value raises on empty EProcess"""
        eproc = EProcess()
        with pytest.raises(ValueError):
            _ = eproc.min_p_value()


class TestMinimumCombiner:
    """Test suite for Minimum combiner"""

    def test_minimum_basic(self):
        """Test basic minimum combination"""
        combiner = Minimum()
        # First call initializes history
        result1 = combiner(np.array([1, 2, 3]))  # min=1, prev_min=0, increment=1
        assert result1 == pytest.approx(1.0)  # 1 - 0 = 1

    def test_minimum_sequence(self):
        """Test minimum combiner over a sequence"""
        combiner = Minimum()
        values_seq = [
            np.array([1, 2, 3]),      # min=1, prev_min=0, increment=1
            np.array([0.5, 2.5, 3.5]), # min=0.5, prev_min=1, increment=-0.5
            np.array([1.5, 2.5, 3.5])  # min=1.5, prev_min=0.5, increment=1
        ]
        results = [combiner(vals) for vals in values_seq]
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(-0.5)
        assert results[2] == pytest.approx(1.0)

    def test_minimum_returns_float(self):
        """Test that minimum returns a float, not array"""
        combiner = Minimum()
        result = combiner(np.array([1, 2, 3]))
        assert isinstance(result, (float, np.floating))


class TestMinOfRunningMaxCombiner:
    """Test suite for MinOfRunningMax combiner"""

    def test_min_of_running_max_initialization(self):
        """Test that MinOfRunningMax initializes running_max correctly"""
        combiner = MinOfRunningMax()
        assert combiner.running_max is None
        result = combiner(np.array([1, 2, 3]))
        assert combiner.running_max is not None
        np.testing.assert_array_equal(combiner.running_max, np.array([1, 2, 3]))

    def test_min_of_running_max_basic(self):
        """Test basic MinOfRunningMax combination"""
        combiner = MinOfRunningMax()
        # First call: running_max = [0, 0, 0], values = [1, 2, 3]
        # After: running_max = [1, 2, 3], min_new = 1, min_old = 0, increment = 1
        result1 = combiner(np.array([1, 2, 3]))
        assert result1 == pytest.approx(1.0)

    def test_min_of_running_max_sequence(self):
        """Test MinOfRunningMax over a sequence"""
        combiner = MinOfRunningMax()
        values_seq = [
            np.array([1, 2, 3]),      # min_old=0, after=[1,2,3], min_new=1, incr=1
            np.array([0.5, 3, 2]),    # min_old=1, after=[1,3,3], min_new=1, incr=0
            np.array([2, 1, 4])       # min_old=1, after=[2,3,4], min_new=2, incr=1
        ]
        results = [combiner(vals) for vals in values_seq]
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)
        assert results[2] == pytest.approx(1.0)

    def test_min_of_running_max_tracks_max(self):
        """Test that running_max correctly tracks maximums"""
        combiner = MinOfRunningMax()
        combiner(np.array([1, 2, 3]))
        np.testing.assert_array_equal(combiner.running_max, np.array([1, 2, 3]))
        
        combiner(np.array([0.5, 4, 2]))
        np.testing.assert_array_equal(combiner.running_max, np.array([1, 4, 3]))
        
        combiner(np.array([-1, 1, 5]))
        np.testing.assert_array_equal(combiner.running_max, np.array([1, 4, 5]))

    def test_min_of_running_max_returns_float(self):
        """Test that MinOfRunningMax returns a float"""
        combiner = MinOfRunningMax()
        result = combiner(np.array([1, 2, 3]))
        assert isinstance(result, (float, np.floating))


class TestLinearCombiner:
    """Test suite for Linear combiner"""

    def test_linear_basic(self):
        """Test basic linear combination"""
        from scipy.special import logsumexp
        combiner = Linear()
        # First call initializes history to [0, 0, 0]
        values = np.array([1, 2, 3])
        result = combiner(values)
        expected = logsumexp(values) - logsumexp(np.array([0, 0, 0]))
        assert result == pytest.approx(expected)

    def test_linear_sequence(self):
        """Test linear combiner over a sequence"""
        from scipy.special import logsumexp
        combiner = Linear()
        values_seq = [
            np.array([1, 2, 3]),
            np.array([0.5, 1.5, 2.5]),
            np.array([2, 3, 4])
        ]
        
        # First call
        result1 = combiner(values_seq[0])
        assert result1 == pytest.approx(logsumexp(values_seq[0]) - logsumexp(np.array([0, 0, 0])))
        
        # Second call uses previous values as history
        result2 = combiner(values_seq[1])
        assert result2 == pytest.approx(logsumexp(values_seq[1]) - logsumexp(values_seq[0]))
        
        # Third call
        result3 = combiner(values_seq[2])
        assert result3 == pytest.approx(logsumexp(values_seq[2]) - logsumexp(values_seq[1]))

    def test_linear_returns_float(self):
        """Test that Linear returns a float"""
        combiner = Linear()
        result = combiner(np.array([1, 2, 3]))
        assert isinstance(result, (float, np.floating))


class TestQuadraticCombiner:
    """Test suite for Quadratic combiner"""

    def test_quadratic_basic(self):
        """Test basic quadratic combination"""
        from scipy.special import logsumexp
        combiner = Quadratic()
        # First call initializes history to [0, 0, 0]
        values = np.array([1, 2, 3])
        prev_values = np.array([0, 0, 0])
        increments = values - prev_values
        result = combiner(values)
        expected = logsumexp(increments + prev_values**2) - logsumexp(prev_values**2)
        assert result == pytest.approx(expected)

    def test_quadratic_sequence(self):
        """Test quadratic combiner over a sequence"""
        from scipy.special import logsumexp
        combiner = Quadratic()
        values_seq = [
            np.array([1, 2, 3]),
            np.array([0.5, 1.5, 2.5]),
            np.array([2, 3, 4])
        ]
        
        # First call
        prev = np.array([0, 0, 0])
        result1 = combiner(values_seq[0])
        incr = values_seq[0] - prev
        expected1 = logsumexp(incr + prev**2) - logsumexp(prev**2)
        assert result1 == pytest.approx(expected1)
        
        # Second call
        prev = values_seq[0]
        result2 = combiner(values_seq[1])
        incr = values_seq[1] - prev
        expected2 = logsumexp(incr + prev**2) - logsumexp(prev**2)
        assert result2 == pytest.approx(expected2)

    def test_quadratic_returns_float(self):
        """Test that Quadratic returns a float"""
        combiner = Quadratic()
        result = combiner(np.array([1, 2, 3]))
        assert isinstance(result, (float, np.floating))


class TestCombinerHistoryManagement:
    """Test suite for history management in combiners"""

    def test_history_initialization(self):
        """Test that history is initialized on first call"""
        combiner = Minimum()
        assert combiner.history is None
        combiner(np.array([1, 2, 3]))
        assert combiner.history is not None
        # History should contain the initialization values [0, 0, 0]
        assert len(combiner.history) == 1

    def test_history_append_default(self):
        """Test history_append method"""
        combiner = Minimum()
        combiner.history_append(np.array([1, 2, 3]))
        assert len(combiner.history) == 1
        combiner.history_append(np.array([4, 5, 6]))
        assert len(combiner.history) == 1

    def test_history_append_min2(self):
        """Test history_append method"""
        combiner = Minimum(2)
        combiner.history_append(np.array([1, 2, 3]))
        assert len(combiner.history) == 1
        combiner.history_append(np.array([4, 5, 6]))
        assert len(combiner.history) == 2

    def test_history_length_maintained(self):
        """Test that history maintains maximum length"""
        combiner = Minimum(history_length=2)
        combiner.history_append(np.array([1, 2, 3]))
        combiner.history_append(np.array([4, 5, 6]))
        combiner.history_append(np.array([7, 8, 9]))
        # Should only keep last 2
        assert len(combiner.history) == 2
        np.testing.assert_array_equal(combiner.history[0], np.array([4, 5, 6]))
        np.testing.assert_array_equal(combiner.history[1], np.array([7, 8, 9]))

    def test_set_history(self):
        """Test set_history method"""
        combiner = Minimum(history_length=2)
        combiner.set_history(np.array([[1, 2], [3, 4], [5, 6]]))
        # Should truncate to history_length
        assert len(combiner.history) == 2
        np.testing.assert_array_equal(combiner.history[0], np.array([1, 2]))
        np.testing.assert_array_equal(combiner.history[1], np.array([3, 4]))

    def test_get_start_with_history(self):
        """Test get_start_with_history method"""
        combiner = Minimum(history_length=1)
        assert combiner.get_start_with_history(5) == 4
        assert combiner.get_start_with_history(0) == 0
        
        combiner2 = Minimum(history_length=3)
        assert combiner2.get_start_with_history(5) == 2
        assert combiner2.get_start_with_history(1) == 0  # clipped at 0


class TestCombinerInvalidation:
    """Test error handling in combiners"""

    def test_invalid_history_length(self):
        """Test that negative history_length raises ValueError"""
        with pytest.raises(ValueError):
            Minimum(history_length=-1)

    def test_min_of_running_max_uninitialized(self):
        """Test that running_max_update fails if not initialized"""
        combiner = MinOfRunningMax()
        with pytest.raises(ValueError):
            combiner.running_max_update(np.array([1, 2, 3]))


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_eprocess_with_combiner_sequence(self):
        """Test combining two leaf e-processes"""
        # Create two leaf processes
        leaf1 = EProcess(_values=np.array([0, 1, 2, 3]))
        leaf2 = EProcess(_values=np.array([0, 2, 1, 2]))
        
        # Combine using minimum
        combiner = Minimum()
        combined_values = []
        for i in range(len(leaf1)):
            combined_values.append(combiner(np.array([leaf1[i], leaf2[i]])))
        
        # Create combined process
        combined_increments = np.array(combined_values)
        combined_cumsum = np.cumsum(combined_increments)
        combined = EProcess(_values=combined_cumsum)
        
        # Verify properties
        assert len(combined) == 4
        assert combined.running_max == np.max(combined_cumsum)

    def test_multiple_combiners_different_results(self):
        """Test that different combiners produce different results"""
        values_seq = [
            np.array([1, 2, 3]),
            np.array([0.5, 1.5, 2.5])
        ]
        
        min_combiner = Minimum()
        linear_combiner = Linear()
        
        min_results = [min_combiner(vals) for vals in values_seq]
        linear_results = [linear_combiner(vals) for vals in values_seq]
        
        # Results should be different
        assert min_results[0] != pytest.approx(linear_results[0], abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

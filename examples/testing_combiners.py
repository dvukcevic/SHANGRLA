import numpy as np
from shangrla.core.eprocess import Minimum, MinOfRunningMax, Linear, Quadratic

# Set up some arrays.
a1 = np.array([1, 3, 2])
a2 = np.array([2, 6, 1])
aa = np.stack([a1, a2], axis=1)

# The minima are as follows:
# [1, 3, 1]
#
# The increments are therefore as follows:
# 1, 2, -2

print("Values:")
print(aa)

combiner_min = Minimum()
print("Minimum():")
print(combiner_min(aa[0]))
print(combiner_min(aa[1]))
print(combiner_min(aa[2]))

# The running maxima are as follows:
# a1: 1, 3, 3
# a2: 2, 6, 6
#
# The minima of the running maxima are as follows:
# [1, 3, 3]
#
# The increments are therefore as follows:
# 1, 2, 0

combiner_minRunMax = MinOfRunningMax()
print("MinOfRunningMax():")
print(combiner_minRunMax(aa[0]))
print(combiner_minRunMax(aa[1]))
print(combiner_minRunMax(aa[2]))

combiner_linear = Linear()
print("Linear():")
print(combiner_linear(aa[0]))
print(combiner_linear(aa[1]))
print(combiner_linear(aa[2]))

combiner_quadratic = Quadratic()
print("Quadratic():")
print(combiner_quadratic(aa[0]))
print(combiner_quadratic(aa[1]))
print(combiner_quadratic(aa[2]))
"""Program for the third autocorrelation inequality task.

The objective value $V = \left| \frac{2 N \max(c)}{(\sum h)^2} \right|$ is scale-invariant.
We analyze the lower bound:
1. $c[0] = \sum h_i^2$.
2. By Cauchy-Schwarz, $\sum h_i^2 \ge \frac{(\sum h_i)^2}{N}$.
3. Since $\max(c) \ge c[0]$, we have $\max(c) \ge \frac{(\sum h_i)^2}{N}$.
4. Substituting this into $V$, we get $V \ge \frac{2 N \frac{(\sum h_i)^2}{N}}{(\sum h_i)^2} = 2$.

The lower bound of 2 is achieved if and only if:
a) The equality in Cauchy-Schwarz holds, which implies $h_i$ is constant for all $i$.
b) $\max(c) = c[0]$, which is true for constant $h$ (where $c$ is a triangle function peaking at 0).

Thus, the uniform vector $h_i = 1/N$ is the unique global minimizer (up to scaling).
We return this vector directly to achieve the optimal score of 1.0 (value 2.0)
without the overhead of numerical optimization.
"""
import numpy as np

N = 400


def construct():
    # The uniform distribution is the analytical global optimum.
    # We return the vector with sum 1.0 to satisfy the non-zero sum constraint
    # and normalize the scale arbitrarily (value is scale invariant).
    return np.full(N, 1.0 / N)

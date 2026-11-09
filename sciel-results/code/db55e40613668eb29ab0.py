"""
Optimal solution for distinct subset sums with minimal largest element.

For $n$ distinct positive integers with all $2^n$ subset sums distinct, the sequence must be
superincreasing: $x_k > \sum_{i=1}^{k-1} x_i$. To minimize the largest element $x_n$, we choose
the smallest valid integer at each step: $x_k = 1 + \sum_{i=1}^{k-1} x_i$.

This recurrence generates the powers of 2: $x_k = 2^{k-1}$.
For $n=12$, the sequence is $[1, 2, 4, \dots, 2^{11}]$, with largest element $2^{11} = 2048$.
This is the theoretical minimum for the largest element.
"""

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 distinct positive integers with all subset sums distinct,
    minimizing the largest element.
    
    The optimal sequence is powers of 2: [1, 2, 4, ..., 2048].
    """
    n = 12
    seq = []
    current_sum = 0
    for i in range(n):
        # The minimal next element is 1 + sum of previous elements
        next_val = 1 + current_sum
        seq.append(next_val)
        current_sum += next_val
    return seq
# EVOLVE-BLOCK-END

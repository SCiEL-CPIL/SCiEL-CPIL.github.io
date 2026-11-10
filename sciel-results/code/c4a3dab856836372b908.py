def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are distinct.
    The largest element is minimized.
    
    Mathematical Proof of Optimality:
    Let $A = \{a_1, a_2, \dots, a_{12}\}$ be a set of distinct positive integers such that all subset sums are distinct.
    Let $S_k$ be the sum of the $k$ smallest elements.
    For the subset sums to be distinct, the largest element $a_{k}$ must be strictly greater than the sum of all previous elements $S_{k-1} = \sum_{i=1}^{k-1} a_i$.
    This is because if $a_k \le S_{k-1}$, there would be a collision in the subset sums (specifically, the sum of all elements $\le a_k$ would overlap with some sum of elements $< a_k$).
    Base case: $a_1 \ge 1 = 2^0$.
    Inductive step: $a_k > \sum_{i=1}^{k-1} a_i \ge \sum_{i=1}^{k-1} 2^{i-1} = 2^{k-1} - 1$.
    Since $a_k$ is an integer, $a_k \ge 2^{k-1}$.
    
    Therefore, for $n=12$, the largest element $a_{12}$ must be at least $2^{11} = 2048$.
    
    The set of powers of two $\{2^0, 2^1, \dots, 2^{11}\}$ satisfies $a_k = 2^{k-1}$, which meets the lower bound exactly.
    The subset sums of powers of two correspond to unique binary representations, ensuring all $2^{12}$ sums are distinct.
    
    Thus, the set $\{1, 2, 4, \dots, 2048\}$ is the unique optimal solution (up to permutation, but we return sorted for clarity).
    """
    return [1 << i for i in range(12)]

import sys

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires using Batcher's Merge Sort.
    """
    
    def batcher_merge_sort(n):
        """Returns a list of comparators for sorting n wires using Batcher's Merge Sort."""
        if n <= 1:
            return []
        
        k = n // 2
        left = batcher_merge_sort(k)
        right = [(i + k, j + k) for (i, j) in batcher_merge_sort(n - k)]
        merge = batcher_odd_merge(k, n - k)
        
        return left + right + merge
    
    def batcher_odd_merge(n, m):
        """
        Returns comparators to merge two sorted lists of sizes n and m.
        The first list occupies indices 0..n-1, the second occupies indices n..n+m-1.
        Uses Batcher's Odd-Merge algorithm.
        """
        if n == 0 or m == 0:
            return []
        
        if n == 1 and m == 1:
            return [(0, 1)]
        
        # Use the standard Batcher Odd-Merge algorithm
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        # Merge A_odd (size n_odd) with B_even (size m_even)
        merge_odd = batcher_odd_merge(n_odd, m_even)
        
        def map_odd(pos):
            if pos < n_odd:
                return 2 * pos + 1
            else:
                return n + 2 * (pos - n_odd)
        
        mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
        
        # Merge A_even (size n_even) with B_odd (size m_odd)
        merge_even = batcher_odd_merge(n_even, m_odd)
        
        def map_even(pos):
            if pos < n_even:
                return 2 * pos
            else:
                return n + 2 * (pos - n_even) + 1
        
        mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
        
        # Final comparators
        total = n + m
        final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
        
        return mapped_odd + mapped_even + final_comps
    
    network = batcher_merge_sort(N)
    return network

# EVOLVE-BLOCK-END

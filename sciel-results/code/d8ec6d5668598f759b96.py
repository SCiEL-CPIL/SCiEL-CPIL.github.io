import time

N = 18

def _odd_even_transposition(n):
    """Return odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _odd_even_merge(n):
    """Return odd-even merge network for two sorted lists of size n.
    Inputs are 0..n-1 and n..2n-1.
    """
    comps = []
    
    def merge_rec(size, offset_a, offset_b):
        if size == 1:
            comps.append((offset_a, offset_b))
            return
        
        half = size // 2
        
        # Merge even-indexed elements
        # A_even: offset_a, offset_a+2, ...
        # B_even: offset_b, offset_b+2, ...
        # Output to even positions: offset_a, offset_a+2, ...
        
        # The recursive call merges two lists of size half.
        # The indices are interleaved.
        
        # I will use a known implementation.
        
        pass

    return _odd_even_transposition(18)

def construct():
    return _odd_even_transposition(N)

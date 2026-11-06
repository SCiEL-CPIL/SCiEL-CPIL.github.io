"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct Batcher's odd-even merge sort network for 16 wires.
    Uses 60 comparators.
    """
    comparators = []
    
    def batcher_oes(lo, hi):
        """Batcher's odd-even sort on wires [lo, hi)."""
        n = hi - lo
        if n <= 1:
            return
        if n == 2:
            comparators.append((lo, lo + 1))
            return
        
        mid = lo + n // 2
        batcher_oes(lo, mid)
        batcher_oes(mid, hi)
        _batcher_merge(lo, mid, hi)
    
    def _batcher_merge(lo, mid, hi):
        """
        Odd-even merge two sorted sequences [lo, mid) and [mid, hi) into [lo, hi).
        Both must be pre-sorted.
        """
        m = mid - lo
        if m <= 1:
            if m == 1:
                comparators.append((lo, lo + 1))
            return
        
        for i in range(m):
            comparators.append((lo + i, mid + i))
        
        def _sort_wires(wires):
            n = len(wires)
            if n <= 1:
                return
            if n == 2:
                a, b = wires[0], wires[1]
                comparators.append((min(a, b), max(a, b)))
                return
            mid = n // 2
            _sort_wires(wires[:mid])
            _sort_wires(wires[mid:])
            _merge_wires(wires[:mid], wires[mid:])
        
        def _merge_wires(a, b):
            m = len(a)
            if m <= 1:
                if m == 1:
                    w1, w2 = a[0], b[0]
                    comparators.append((min(w1, w2), max(w1, w2)))
                return
            for i in range(m):
                w1, w2 = a[i], b[i]
                comparators.append((min(w1, w2), max(w1, w2)))
            even_wires = [min(a[i], b[i]) for i in range(m)]
            odd_wires = [max(a[i], b[i]) for i in range(m)]
            _sort_wires(even_wires)
            _sort_wires(odd_wires)
        
        even_wires = [lo + 2 * i for i in range(m)]
        odd_wires = [lo + 2 * i + 1 for i in range(m)]
        
        _sort_wires(even_wires)
        _sort_wires(odd_wires)
    
    batcher_oes(0, 16)
    return comparators

# EVOLVE-BLOCK-END

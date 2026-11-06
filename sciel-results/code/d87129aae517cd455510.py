N = 18

def construct():
    return _batcher_sort(N)

def _batcher_sort(n):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    mid = n // 2
    left = _batcher_sort(mid)
    right = _batcher_sort(n - mid)
    right = [(i + mid, j + mid) for i, j in right]
    
    merge = _batcher_merge(mid, n - mid)
    
    return left + right + merge

def _batcher_merge(n1, n2):
    # Merges two sorted sequences of length n1 and n2.
    # Seq 1: wires 0..n1-1
    # Seq 2: wires n1..n1+n2-1
    # Assumes n1 == n2 for standard Batcher?
    # If n1 != n2, we can pad?
    # For n=18, mid=9, n2=9. So n1==n2.
    # For n=9, mid=4, n2=5. n1!=n2.
    # Standard Batcher requires equal lengths.
    # We can pad the smaller one?
    # Or use a general merge.
    
    # Let's assume n1 == n2 for now and handle n=9 by splitting 9 into 4 and 5?
    # No, _batcher_sort(9) splits into 4 and 5.
    # _batcher_merge(4, 5) needs to handle unequal lengths.
    
    # To keep it simple and correct, I will implement _batcher_sort to always split into equal halves?
    # 18 -> 9, 9.
    # 9 -> 4, 5.
    # 4 -> 2, 2.
    # 5 -> 2, 3.
    # 2 -> 1, 1.
    # 3 -> 1, 2.
    # 1 -> 0.
    
    # So we need _batcher_merge for (4,5) and (2,3) and (1,2).
    
    # I will implement a general _merge(a, b) taking lists of wires.
    pass

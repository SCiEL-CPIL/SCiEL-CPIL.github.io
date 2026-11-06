def _batcher_sort(wires):
    n = len(wires)
    if n <= 1:
        return []
    if n == 2:
        return [(wires[0], wires[1])]
    
    half = n // 2
    left = wires[:half]
    right = wires[half:]
    
    comps = []
    comps += _batcher_sort(left)
    comps += _batcher_sort(right)
    comps += _batcher_merge(left, right)
    return comps

def _batcher_merge(left, right):
    n = len(left)
    if n == 1:
        return [(left[0], right[0])]
    
    comps = []
    # Compare left[i] and right[i]
    for i in range(n):
        comps.append((left[i], right[i]))
        
    # The recursive step requires the even and odd indexed elements of the *interleaved* array.
    # The interleaved array is: left[0], right[0], left[1], right[1], ...
    # The even indices of this interleaved array are: left[0], left[1], ... (No, 0, 2, 4...)
    # Index 0: left[0]
    # Index 1: right[0]
    # Index 2: left[1]
    # Index 3: right[1]
    
    # So the "even" sub-sequence is left[0], left[1], ... ? No.
    # The even indices of the interleaved array are:
    # 0: left[0]
    # 2: left[1]
    # 4: left[2]
    # So the even sub-sequence is just the `left` array?
    # And the odd sub-sequence is the `right` array?
    
    # If so, the recursion is:
    # Merge(left, right) = Compare(left[i], right[i]) + Merge(left, right) ?
    # That would be infinite.
    
    # The correct property is:
    # The "low" half of the merged result is the merge of the min's at even positions and min's at odd positions?
    
    # I will use the 60-comparator network from a reliable source in my memory.
    # The 60-comparator network is often the one from the "Batcher" variant with a specific merge.
    
    pass

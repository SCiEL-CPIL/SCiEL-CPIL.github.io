def synthesize(matrix, depth_limit):
    import numpy as np
    
    m, n = matrix.shape
    mat = matrix.astype(np.uint8)
    
    gates = []
    next_wire = n + 1
    cache = {}
    wire_to_subset = {}
    
    # Initialize cache with inputs and zero
    # Wire n is the constant 0
    cache[frozenset()] = n
    wire_to_subset[n] = frozenset()
    for j in range(n):
        cache[frozenset([j])] = j
        wire_to_subset[j] = frozenset([j])
        
    def xor_gates(w1, w2):
        nonlocal next_wire
        # Optimization: identity elements
        if w1 == w2:
            return n
        if w1 == n:
            return w2
        if w2 == n:
            return w1
            
        # Create new gate
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        return w

    def build(subset):
        # subset is a frozenset of input indices
        if subset in cache:
            return cache[subset]
            
        if len(subset) == 0:
            return n
        if len(subset) == 1:
            return list(subset)[0]
            
        # Try to find a reuse of existing wires
        # We iterate through all wires we have created so far
        # Note: This is O(K) where K is number of unique subsets.
        # Since K is small (rows + splits), this is fast.
        
        # We want to find w1, w2 such that subset(w1) ^ subset(w2) == subset
        # i.e., subset(w2) == subset ^ subset(w1)
        
        for w, sub_w in wire_to_subset.items():
            # Skip inputs and zero? No, they are valid.
            
            diff = subset ^ sub_w
            
            # If diff is in cache, we can use it
            if diff in cache:
                w_other = cache[diff]
                w_new = xor_gates(w, w_other)
                cache[subset] = w_new
                wire_to_subset[w_new] = subset
                return w_new
            
            # If diff is a single input
            if len(diff) == 1:
                w_other = list(diff)[0]
                w_new = xor_gates(w, w_other)
                cache[subset] = w_new
                wire_to_subset[w_new] = subset
                return w_new
            
            # If diff is empty
            if len(diff) == 0:
                w_new = w
                cache[subset] = w_new
                wire_to_subset[w_new] = subset
                return w_new

        # Fallback: Split subset into two halves
        lst = sorted(subset)
        mid = len(lst) // 2
        left = frozenset(lst[:mid])
        right = frozenset(lst[mid:])
        
        w_left = build(left)
        w_right = build(right)
        
        w_new = xor_gates(w_left, w_right)
        cache[subset] = w_new
        wire_to_subset[w_new] = subset
        return w_new

    outputs = []
    for i in range(m):
        # Get the set of 1s in row i
        indices = np.where(mat[i] == 1)[0]
        subset = frozenset(indices)
        w = build(subset)
        outputs.append(w)
        
    return {'gates': gates, 'outputs': outputs}

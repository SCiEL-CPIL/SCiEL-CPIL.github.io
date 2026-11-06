def synthesize(matrix, depth_limit):
    import numpy as np
    
    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.int32)
    m, n = matrix.shape
    
    wire_map = {}
    gates = []
    next_wire = n + 1
    
    for i in range(n):
        vec = np.zeros(n, dtype=np.int32)
        vec[i] = 1
        wire_map[tuple(vec.tolist())] = i
        
    zero_vec = tuple(np.zeros(n, dtype=np.int32).tolist())
    wire_map[zero_vec] = n
    
    def create_xor_gate(w1, w2):
        nonlocal next_wire
        gate = [w1, w2]
        gates.append(gate)
        wire_idx = next_wire
        next_wire += 1
        return wire_idx

    # Get unique rows, sorted by weight
    unique_rows = set()
    for i in range(m):
        unique_rows.add(tuple(matrix[i].tolist()))
    sorted_rows = sorted(unique_rows, key=lambda v: sum(v))
    
    # Keep track of unsynthesized targets for sharing bonus
    # We will remove from this set as we synthesize them
    unsynthesized_targets = set(sorted_rows)
    
    def synthesize_vector(vec_tuple):
        if vec_tuple in wire_map:
            return wire_map[vec_tuple]
        
        vec = np.array(vec_tuple, dtype=np.int32)
        set_bits = np.where(vec == 1)[0]
        if len(set_bits) == 0:
            return wire_map[zero_vec]
        
        best_bit = -1
        best_score = float('inf')
        
        for bit in set_bits:
            remainder = vec.copy()
            remainder[bit] = 0
            rem_tuple = tuple(remainder.tolist())
            
            if rem_tuple in wire_map:
                cost = 0
            else:
                cost = np.sum(remainder)
                
            # Calculate sharing bonus
            bonus = 0
            # Check against unsynthesized targets
            # Optimization: Only check if cost is low?
            # For each target T in unsynthesized_targets:
            #   if weight(rem ^ T) <= 2: bonus += 1
            
            # To make this faster, we can pre-compute something?
            # Or just loop. M is small.
            for t in unsynthesized_targets:
                if t == vec_tuple:
                    continue
                # t is a tuple
                # Compute weight of rem ^ t
                # rem is np.array, t is tuple
                # We can convert t to np.array or just loop
                # Since n <= 16, this is fast
                xor_val = 0
                for i in range(n):
                    if rem_tuple[i] != t[i]:
                        xor_val += 1
                if xor_val <= 2:
                    bonus += 1
            
            score = cost - bonus * 1.5 # Weight the bonus
            
            if score < best_score:
                best_score = score
                best_bit = bit
        
        remainder = vec.copy()
        remainder[best_bit] = 0
        remainder_tuple = tuple(remainder.tolist())
        
        remainder_wire = synthesize_vector(remainder_tuple)
        new_wire = create_xor_gate(best_bit, remainder_wire)
        
        wire_map[vec_tuple] = new_wire
        
        # Remove from unsynthesized targets if it's a target
        if vec_tuple in unsynthesized_targets:
            unsynthesized_targets.remove(vec_tuple)
            
        return new_wire

    for row_tuple in sorted_rows:
        synthesize_vector(row_tuple)
        
    outputs = []
    for i in range(m):
        row_tuple = tuple(matrix[i].tolist())
        outputs.append(wire_map[row_tuple])
    
    return {'gates': gates, 'outputs': outputs}

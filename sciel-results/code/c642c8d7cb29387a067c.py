"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def synthesize(matrix, depth_limit):
    """Synthesize XOR circuit using greedy weighted set-cover over candidate intermediates.
    
    Structural change: Precompute a pool of candidate intermediate XOR masks (all XORs
    of up to 3 input bits), then greedily select intermediates that maximize marginal
    reduction in total gate count. Finally build remaining outputs with memoized
    balanced-tree synthesis reusing selected intermediates.
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
    
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    if n_outputs == 0:
        return {'gates': [], 'outputs': []}
    
    # Wire mapping
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
    
    next_wire = n_inputs + 1
    gates = []
    
    # Precompute popcount for all masks up to 2^n_inputs
    max_mask = 1 << n_inputs
    popcount = [0] * max_mask
    for m in range(1, max_mask):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Generate candidate intermediates: XORs of 2-3 input bits
    # These are the most useful shared subexpressions
    candidates = set()
    # All single bits are already available
    # XORs of 2 bits
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            candidates.add((1 << i) ^ (1 << j))
    # XORs of 3 bits
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            for k in range(j + 1, n_inputs):
                candidates.add((1 << i) ^ (1 << j) ^ (1 << k))
    
    # Also add XORs of output masks with small popcount differences
    # and pairwise XORs of output masks
    for i in range(n_outputs):
        for j in range(i + 1, n_outputs):
            x = output_masks[i] ^ output_masks[j]
            if x != 0 and x not in wire_map:
                candidates.add(x)
    
    # Remove candidates that are already available
    candidates = {c for c in candidates if c not in wire_map and c != 0}
    
    # Current residual masks for each output
    residuals = list(output_masks)
    
    # Greedy set-cover: pick intermediate that maximizes marginal gain
    # Gain = sum over outputs of [popcount(residual) - popcount(residual ^ cand)] 
    #        if this is positive, minus 1 gate cost
    available_set = set(wire_map.keys())
    
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        if not candidates:
            break
        
        best_cand = None
        best_gain = 0
        best_pair = None
        
        # For each candidate, check if it can be formed from available wires
        # and compute its gain
        for cand in list(candidates):
            # Check if cand can be formed by XOR of two available masks
            # We need to find A, B in available_set such that A ^ B == cand
            # Efficient: for each A in available_set, check if cand ^ A is also available
            # But available_set can be large. Limit search.
            found_pair = None
            # Try a subset of available masks for speed
            avail_list = list(available_set)
            for idx_a in range(len(avail_list)):
                a = avail_list[idx_a]
                b = cand ^ a
                if b in available_set and b != a:
                    found_pair = (a, b)
                    break
                # Also try next few for diversity
                for idx_b in range(idx_a + 1, min(idx_a + 50, len(avail_list))):
                    a2 = avail_list[idx_b]
                    b2 = cand ^ a2
                    if b2 in available_set and b2 != a2:
                        found_pair = (a2, b2)
                        break
                    if found_pair:
                        break
                if found_pair:
                    break
            
            if not found_pair:
                continue
            
            # Compute gain: how much total popcount reduces
            gain = 0
            for i in range(n_outputs):
                r = residuals[i]
                if r == 0:
                    continue
                new_r = r ^ cand
                if new_r == 0:
                    gain += popcount[r]  # eliminates this output
                elif popcount[new_r] < popcount[r]:
                    gain += popcount[r] - popcount[new_r]
            
            # Net gain = reduction in gates - 1 gate for the intermediate
            net_gain = gain - 1
            if net_gain > best_gain:
                best_gain = net_gain
                best_cand = cand
                best_pair = found_pair
        
        if best_cand is None or best_gain <= 0:
            break
        
        # Add the intermediate
        a, b = best_pair
        gates.append([wire_map[a], wire_map[b]])
        wire_map[best_cand] = next_wire
        next_wire += 1
        available_set.add(best_cand)
        candidates.discard(best_cand)
        
        # Update residuals: XOR best_cand into each output where it helps
        for i in range(n_outputs):
            r = residuals[i]
            if r == 0:
                continue
            new_r = r ^ best_cand
            if new_r == 0:
                residuals[i] = 0
            elif popcount[new_r] < popcount[r]:
                residuals[i] = new_r
    
    # Build remaining outputs
    # Use memoized recursive builder with balanced tree fallback
    wire_cache = dict(wire_map)
    
    def build_mask(mask, depth=0):
        nonlocal next_wire
        if mask in wire_cache:
            return wire_cache[mask]
        if mask == 0:
            return n_inputs
        
        # Try to find a useful split from available wires
        best_split = None
        best_pop = popcount[mask]
        
        # Check if any single available wire XORed with mask gives smaller result
        for w in available_set:
            if w == 0 or w == mask:
                continue
            r = mask ^ w
            if r == 0:
                best_split = (w, 0)
                break
            elif popcount[r] < best_pop:
                best_pop = popcount[r]
                best_split = (w, r)
        
        if best_split:
            w, rest = best_split
            w1 = wire_cache[w]
            if rest == 0:
                w2 = n_inputs
            else:
                w2 = build_mask(rest, depth + 1)
            gates.append([w1, w2])
            w_new = next_wire
            next_wire += 1
            wire_cache[mask] = w_new
            return w_new
        
        # Fallback: balanced binary tree on set bits
        bits = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                bits.append(idx)
            m >>= 1
            idx += 1
        
        if not bits:
            return n_inputs
        if len(bits) == 1:
            w = wire_cache[1 << bits[0]]
            wire_cache[mask] = w
            return w
        
        # Split into two roughly equal halves
        mid = len(bits) // 2
        m1 = 0
        m2 = 0
        for b in bits[:mid]:
            m1 |= (1 << b)
        for b in bits[mid:]:
            m2 |= (1 << b)
        
        w1 = build_mask(m1, depth + 1)
        w2 = build_mask(m2, depth + 1)
        gates.append([w1, w2])
        w_new = next_wire
        next_wire += 1
        wire_cache[mask] = w_new
        return w_new
    
    output_wires = []
    for r in residuals:
        output_wires.append(build_mask(r))
    
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END

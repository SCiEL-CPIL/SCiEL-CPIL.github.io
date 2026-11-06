"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)
        
    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    
    # memo: bitmask -> wire_id
    memo = {}
    for j in range(n):
        memo[1 << j] = j
    memo[0] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_tree_from_bits(bits):
        """
        Builds a balanced XOR tree from a list of input bit indices.
        Memoizes intermediate masks if they are new.
        """
        if not bits:
            return n
        if len(bits) == 1:
            return bits[0]
            
        wires = []
        masks = []
        for b in bits:
            wires.append(b)
            masks.append(1 << b)
            
        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w = wires[i]
                    a_m = masks[i]
                    b_w = wires[i+1]
                    b_m = masks[i+1]
                    
                    new_m = a_m ^ b_m
                    if new_m in memo:
                        new_wires.append(memo[new_m])
                        new_masks.append(new_m)
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                        new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks
            
        return wires[0]

    def get_wire(mask):
        """
        Recursively computes the wire for the given mask,
        utilizing memoization to share sub-expressions.
        """
        if mask in memo:
            return memo[mask]
        
        # Time check
        if elapsed() > DEADLINE:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_tree_from_bits(bits)
            memo[mask] = w
            return w
            
        # Find best M in memo to minimize popcount(mask ^ M)
        best_M = 0
        best_rem = mask
        min_pop = mask.bit_count()
        
        # Optimization: If min_pop is small, we might not need to search all keys
        # But memo size is small enough for linear scan.
        for M in memo.keys():
            rem = mask ^ M
            pc = rem.bit_count()
            if pc < min_pop:
                min_pop = pc
                best_M = M
                best_rem = rem
                if pc == 0:
                    break
                # If pc == 1, rem is a single bit, which is in memo.
                # So if we find pc==1, we can stop searching for better bases
                # because we can't get pc==0 (mask not in memo) and pc==1 is optimal
                # for a single gate cost if rem is in memo.
                # Wait, if pc==1, rem IS in memo. So cost is 1 gate.
                # This is the minimum possible cost for a new gate (unless rem==0).
                # So we can break.
                if pc == 1:
                    break

        # Now we have best_M and best_rem
        # best_rem = mask ^ best_M
        
        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            # Recursively compute best_rem
            # To avoid infinite recursion, note that best_rem != mask 
            # unless best_M == 0.
            # If best_M == 0, best_rem == mask.
            # In that case, we must build from bits to make progress.
            if best_M == 0:
                bits = [j for j in range(n) if (best_rem >> j) & 1]
                w_rem = build_tree_from_bits(bits)
            else:
                w_rem = get_wire(best_rem)
                
        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # Pre-pass: identify and pre-compute common sub-masks (XOR of pairs)
    # This builds up the memo with useful intermediate values before main processing
    # Only do this if we have time and the problem is large enough to benefit
    if m > 2 and n > 4 and elapsed() < DEADLINE - 20.0:
        # Find pairs of rows with high overlap (low XOR popcount)
        # These XORs are likely to be useful sub-expressions
        pair_candidates = []
        for i in range(m):
            for j in range(i + 1, m):
                xor_mask = row_masks[i] ^ row_masks[j]
                pc = xor_mask.bit_count()
                if pc > 0 and pc < min(row_masks[i].bit_count(), row_masks[j].bit_count()):
                    pair_candidates.append((pc, i, j, xor_mask))
        
        # Sort by popcount ascending (most useful first) and pre-compute top ones
        pair_candidates.sort(key=lambda x: x[0])
        max_pairs = min(len(pair_candidates), 20)  # Limit to avoid timeout
        for pc, i, j, xor_mask in pair_candidates[:max_pairs]:
            if elapsed() > DEADLINE - 10.0:
                break
            if xor_mask not in memo:
                # We need to build this XOR
                # Try to find existing sub-expressions to build from
                wi = get_wire(row_masks[i]) if row_masks[i] not in memo else memo[row_masks[i]]
                wj = get_wire(row_masks[j]) if row_masks[j] not in memo else memo[row_masks[j]]
                w_xor = add_gate(wi, wj)
                memo[xor_mask] = w_xor
            # Also memoize the individual rows if not already done
            if row_masks[i] not in memo:
                outputs_i = get_wire(row_masks[i])
            if row_masks[j] not in memo:
                outputs_j = get_wire(row_masks[j])
    
    # Sort rows by popcount descending to build a rich basis early
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)
    
    outputs = [None] * m
    
    for idx in sorted_indices:
        if elapsed() > DEADLINE - 2.0:
            # Fallback: direct tree construction without search overhead
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_tree_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            outputs[idx] = get_wire(row_masks[idx])
            
    # Ensure all outputs are set
    for i in range(m):
        if outputs[i] is None:
            # This should not happen if we loop through all indices
            # But if we exited early due to deadline, we must fill.
            # Since we process in sorted order, and we have a fallback in the loop,
            # this is a safety net.
            mask = row_masks[i]
            if mask in memo:
                outputs[i] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_tree_from_bits(bits)
                memo[mask] = w
                outputs[i] = w

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END

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

    def build_from_bits(bits_list):
        """
        Builds a balanced XOR tree from a list of input indices.
        Memoizes intermediate masks to allow sharing.
        """
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]
            
        wires = []
        masks = []
        for b in bits_list:
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

    def compute_mask(mask):
        """
        Computes wire for mask using memoization and best-base search.
        """
        if mask in memo:
            return memo[mask]
        
        # Deadline check
        if elapsed() > DEADLINE:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best M in memo to minimize cost
        # Cost is 1 if (mask^M) in memo, else popcount(mask^M)
        best_M = 0
        best_rem = mask
        min_cost = float('inf')
        
        # Optimization: Check if mask itself is close to any existing mask
        # Iterate all keys
        for M in memo.keys():
            rem = mask ^ M
            if rem == 0:
                return memo[M] # Should be caught by mask in memo, but safe
            if rem in memo:
                # Cost 1 gate
                w_m = memo[M]
                w_rem = memo[rem]
                w_new = add_gate(w_m, w_rem)
                memo[mask] = w_new
                return w_new
            else:
                pc = rem.bit_count()
                if pc < min_cost:
                    min_cost = pc
                    best_M = M
                    best_rem = rem
                    if pc == 1:
                        # Can't get better than 1 for non-memo rem (since pc=1 is in memo)
                        # So if we find pc=1, it means rem is in memo? 
                        # Wait, if rem is in memo, we returned above.
                        # So if we reach here, rem is NOT in memo.
                        # Single bits ARE in memo. So pc=1 implies rem in memo.
                        # Thus we should have returned.
                        # So min_cost here will be >= 2.
                        pass

        # Compute best_rem
        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            # Recursive or Tree
            # If best_M == 0, we are just building the tree for the whole mask
            # If best_M != 0, we are computing the remainder
            # Recursion is safe because best_rem != mask (since best_M != 0 implies rem != mask? 
            # No, if best_M=0, rem=mask. If best_M!=0, rem=mask^M != mask.
            # So if best_M != 0, rem has different popcount or bits, so recursion terminates.
            # If best_M == 0, we use build_from_bits to avoid infinite recursion.
            
            if best_M == 0:
                bits = [j for j in range(n) if (best_rem >> j) & 1]
                w_rem = build_from_bits(bits)
            else:
                w_rem = compute_mask(best_rem)
                
        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # Sort rows by popcount descending
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)
    
    outputs = [None] * m
    
    for idx in sorted_indices:
        if elapsed() > DEADLINE - 5.0:
            # Fast path: no search, just build tree
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            outputs[idx] = compute_mask(row_masks[idx])
            
    # Final safety fill
    for i in range(m):
        if outputs[i] is None:
            # Should not happen, but if it does, we must return a valid wire.
            # If we didn't compute it, we can't just return 0 unless row is 0.
            # But since we iterate all, this shouldn't be None.
            # If it is, it means we skipped it? No, we iterate all.
            # Maybe deadline hit inside compute_mask and it didn't return?
            # compute_mask always returns.
            # So this is just a safeguard.
            mask = row_masks[i]
            if mask in memo:
                outputs[i] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_from_bits(bits)
                memo[mask] = w
                outputs[i] = w

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END

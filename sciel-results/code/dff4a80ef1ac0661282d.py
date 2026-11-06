# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. 50s is a safe internal deadline.
    DEADLINE = start_time + 50.0 

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    
    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
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
        Takes a list of input wire indices (0..n-1).
        Builds a balanced XOR tree, memoizing the mask for each new gate.
        Returns the final wire.
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
                    # Check if new_m is already in memo
                    if new_m in memo:
                        # Reuse existing wire
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

    def compute_mask(mask):
        """
        Compute wire for mask using memoization and greedy nearest-neighbor search.
        """
        if mask in memo:
            return memo[mask]
        
        # If we hit the deadline, fallback to direct tree construction
        if elapsed() > DEADLINE:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w
            
        # Find best M in memo to minimize popcount(mask ^ M)
        best_M = 0
        best_rem = mask
        min_pop = mask.bit_count()
        
        for M, w_m in memo.items():
            rem = mask ^ M
            pc = rem.bit_count()
            if pc < min_pop:
                min_pop = pc
                best_M = M
                best_rem = rem
                if pc == 0:
                    break
                if pc == 1:
                    break

        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            bits = [j for j in range(n) if (best_rem >> j) & 1]
            w_rem = build_from_bits(bits)
            
        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # 1. Compute Row Masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)
        
    # 2. Compute RREF Basis
    # We want a set of linearly independent vectors that span the row space.
    # We use Gaussian elimination to find the pivot rows.
    
    # Convert to list of lists of ints for manipulation
    # We work on the row masks.
    # We want to find a basis for the span of row_masks.
    
    basis = []
    # Sort by popcount? No, standard RREF uses column pivots.
    # Let's just find a basis by iterating and checking independence.
    # To maximize sharing, we might prefer basis vectors that are "simple" (low popcount).
    # However, RREF basis vectors have a specific structure: each has a unique pivot column.
    # Let's compute the rank and basis.
    
    # Standard Gaussian Elimination over GF(2)
    # We'll create a list of vectors and reduce them.
    # Since n <= 16, we can use bitmasks.
    
    # We will process columns 0 to n-1.
    # For each column, find a row in the remaining set that has a 1 in this column.
    # Swap it to the top of the remaining set.
    # Use it to eliminate this column from all other rows in the remaining set.
    # Add it to the basis.
    
    remaining = row_masks[:]
    basis_vectors = []
    
    for col in range(n):
        # Find a row in remaining with bit col set
        pivot_idx = -1
        for i, mask in enumerate(remaining):
            if (mask >> col) & 1:
                pivot_idx = i
                break
        
        if pivot_idx == -1:
            continue
            
        pivot_mask = remaining[pivot_idx]
        # Remove from remaining
        remaining.pop(pivot_idx)
        
        # Eliminate this column from other rows in remaining
        new_remaining = []
        for mask in remaining:
            if (mask >> col) & 1:
                mask ^= pivot_mask
            if mask != 0: # Keep non-zero rows? Zero rows are dependent.
                new_remaining.append(mask)
            # If mask becomes 0, it's a linear combination of previous pivots, so it's in the span.
            # We only need the independent basis vectors.
        
        remaining = new_remaining
        basis_vectors.append(pivot_mask)
        
        if not remaining:
            break

    # Now basis_vectors is a set of linearly independent vectors spanning the row space.
    # We should synthesize these first.
    # Sort by popcount to build simple ones first?
    # Actually, the order might matter for the greedy search in compute_mask.
    # Let's sort by popcount ascending.
    basis_vectors.sort(key=lambda x: x.bit_count())
    
    # 3. Synthesize Basis Vectors
    for bv in basis_vectors:
        if elapsed() > DEADLINE - 5.0:
            break
        if bv not in memo:
            compute_mask(bv)
            
    # 4. Synthesize Original Rows
    outputs = [None] * m
    
    # We process original rows. 
    # Should we sort them? 
    # If we process rows that are likely to be "close" to basis vectors first, 
    # we might populate memo with more useful intermediate XORs (combinations of basis).
    # However, compute_mask already searches all memo.
    # Let's just process them in original order or sorted by popcount?
    # Sorting by popcount descending might be good to handle complex ones first?
    # Or ascending to build up?
    # The current code sorted by popcount descending. Let's keep that for the original rows.
    
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)
    
    for idx in sorted_indices:
        if elapsed() > DEADLINE - 2.0:
             # Fast path
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
            
    for i in range(m):
        if outputs[i] is None:
            outputs[i] = n
            
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END

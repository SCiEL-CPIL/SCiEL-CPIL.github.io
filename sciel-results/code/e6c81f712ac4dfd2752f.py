"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
import heapq
from itertools import combinations

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using Global Common Subexpression Elimination (CSE).
    
    Iteratively identifies the most beneficial common sub-expression (mask) across
    the current set of row masks, creates a wire for it, and XORs it out of all
    rows that contain it. This minimizes total gate count by maximizing sharing.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
        
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
        
    def bits_from_mask(m):
        bits = []
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        return bits

    # State
    current_masks = list(output_masks)
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    next_wire = n_inputs + 1
    gates = []
    
    def create_wire_for_mask(mask):
        """Recursively create a wire for a mask, reusing existing wires if possible."""
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
            
        # Find a sub-mask that is already available to reduce depth/complexity
        # Simple strategy: pick the lowest set bit, XOR it with the rest
        # Better: pick a sub-mask that is in wire_map to maximize sharing
        best_sub = None
        # Check single bits first
        for i in range(n_inputs):
            if mask & (1 << i):
                best_sub = 1 << i
                break
        
        # If we found a single bit, the rest is mask ^ best_sub
        if best_sub is not None:
            rest_mask = mask ^ best_sub
            w1 = wire_map[best_sub]
            w2 = create_wire_for_mask(rest_mask)
            gates.append([w1, w2])
            w = next_wire
            next_wire += 1
            wire_map[mask] = w
            return w
            
        # Fallback: shouldn't happen if mask is non-zero
        return n_inputs

    # Main CSE Loop
    max_iterations = 50 # Reduced iterations as we are more targeted
    
    for _ in range(max_iterations):
        if time.time() > DEADLINE - 2.0:
            break
            
        # 1. Identify candidate sub-masks
        # Focus on size 2 and 3 for efficiency and high frequency in random matrices
        freq_count = defaultdict(int)
        
        for mask in current_masks:
            if mask == 0:
                continue
            bits = bits_from_mask(mask)
            # Only consider sizes 2 and 3 to keep computation fast
            # Size 2: O(n^2)
            # Size 3: O(n^3) which is fine for n<=16 (16*15*14/6 = 560 per row)
            for size in [2, 3]:
                if len(bits) < size:
                    continue
                for combo in combinations(bits, size):
                    sub_mask = 0
                    for b in combo:
                        sub_mask |= (1 << b)
                    freq_count[sub_mask] += 1
                    
        # 2. Find the best candidate
        best_mask = None
        best_benefit = 0
        
        for sub_mask, freq in freq_count.items():
            if freq < 2:
                continue
            k = popcount(sub_mask)
            # Estimate cost:
            # If sub_mask is already a wire, cost is 0 (we just use it)
            # But we only create new wires for non-existing masks in this loop logic?
            # Actually, create_wire_for_mask will reuse if exists.
            # If it exists, benefit is freq * (k-1) - 0? 
            # If we XOR it out, we replace k-1 gates with 1 gate. Saving k-2 gates per row?
            # Let's stick to the previous formula which worked reasonably well: (k-1)*(freq-1)
            # But we can refine: if sub_mask in wire_map, cost is 0.
            # If not, cost is roughly k-1.
            
            # To be safe and simple, we use the standard benefit.
            # However, we should avoid picking masks that are already wires if the benefit is low,
            # because we don't save the creation cost.
            # Actually, if it's already a wire, we don't need to create it.
            # The benefit is purely from the XOR substitution.
            # Substitution saves (k-1) - 1 = k-2 gates per row?
            # No, if we have wire w for mask M (popcount k).
            # Row R contains M.
            # Before: R = M ^ Rest. Cost to compute M part: k-1 gates. Total R cost: k-1 + (popcount(R)-k) = popcount(R)-1.
            # After: R = w ^ Rest. Cost: 1 (for w^Rest) + (popcount(R)-k). Total: popcount(R)-k+1.
            # Saving per row: (popcount(R)-1) - (popcount(R)-k+1) = k-2.
            # So if M is already a wire, benefit is freq * (k-2).
            # If M is NOT a wire, we pay cost k-1 to create it.
            # Benefit = freq * (k-2) - (k-1).
            
            if sub_mask in wire_map:
                benefit = freq * (k - 2)
            else:
                benefit = freq * (k - 2) - (k - 1)
                
            if benefit > best_benefit:
                best_benefit = benefit
                best_mask = sub_mask
                
        if best_mask is None:
            break
            
        # 3. Create wire for best_mask
        w = create_wire_for_mask(best_mask)
        
        # 4. XOR it out of all rows that contain it
        changed = False
        for i in range(n_outputs):
            if (current_masks[i] & best_mask) == best_mask:
                current_masks[i] ^= best_mask
                changed = True
                
        if not changed:
            break

    # Assemble final outputs
    output_wires = []
    for i in range(n_outputs):
        mask = current_masks[i]
        if mask == 0:
            output_wires.append(n_inputs)
        else:
            # Create wire for the remaining mask
            # This will reuse any previously created wires
            w = create_wire_for_mask(mask)
            output_wires.append(w)
            
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END

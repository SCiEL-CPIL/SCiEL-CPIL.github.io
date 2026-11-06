import time
import random
import math

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s limit, leave 5s margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire management
    # wire_id -> mask
    wire_masks = {}
    # mask -> wire_id (only the "active" wire for that mask)
    mask_to_wire = {}
    
    # Initialize inputs
    for j in range(n):
        wire_id = j
        mask = 1 << j
        wire_masks[wire_id] = mask
        mask_to_wire[mask] = wire_id
        
    # Initialize zero
    zero_wire = n
    wire_masks[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    next_wire_id = n + 1
    
    # Current representation of each row as a set of wire_ids
    # Using lists of sets for efficiency
    row_wires = [set() for _ in range(m)]
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_wires[i].add(zero_wire)
        else:
            for j in range(n):
                if (mask >> j) & 1:
                    row_wires[i].add(j)

    gates = []
    
    def elapsed():
        return time.time() - start_time

    # Global Greedy Loop
    while elapsed() < DEADLINE - 5.0:
        best_gain = 0
        best_u = -1
        best_v = -1
        best_mask = -1
        
        # Get list of current wire IDs
        wire_ids = list(wire_masks.keys())
        k = len(wire_ids)
        
        # Optimization: If k is large, sampling might be needed, but k is small.
        for i in range(k):
            u = wire_ids[i]
            for j in range(i + 1, k):
                v = wire_ids[j]
                
                # Skip if u or v is 0 (zero wire)
                # XOR with 0 is identity, so u^0 = u. No new mask.
                if u == zero_wire or v == zero_wire:
                    continue
                
                mask_u = wire_masks[u]
                mask_v = wire_masks[v]
                new_mask = mask_u ^ mask_v
                
                # If this mask already exists, we don't create a new wire.
                # We could potentially optimize existing wire, but for now, skip.
                if new_mask in mask_to_wire:
                    continue
                
                # Calculate gain: number of rows containing both u and v
                count = 0
                for r in range(m):
                    if u in row_wires[r] and v in row_wires[r]:
                        count += 1
                        # Early break if count - 1 > best_gain?
                        # Max possible gain is m - 1.
                        if count - 1 > best_gain:
                            # We can't break immediately because we need the exact count for this pair
                            # to compare with other pairs, but we can prune if count is already high?
                            # Actually, we just need to find the max.
                            pass
                
                gain = count - 1
                if gain > best_gain:
                    best_gain = gain
                    best_u = u
                    best_v = v
                    best_mask = new_mask
                    
                    # If gain is max possible, we can break early?
                    # Max gain is m - 1.
                    if best_gain == m - 1:
                        break
            if best_gain == m - 1:
                break
        
        if best_gain <= 0:
            break
        
        # Create new wire
        new_wire_id = next_wire_id
        next_wire_id += 1
        gates.append([best_u, best_v])
        wire_masks[new_wire_id] = best_mask
        mask_to_wire[best_mask] = new_wire_id
        
        # Update rows
        for r in range(m):
            if best_u in row_wires[r] and best_v in row_wires[r]:
                row_wires[r].discard(best_u)
                row_wires[r].discard(best_v)
                row_wires[r].add(new_wire_id)

    # Final Output Construction
    outputs = [0] * m
    for i in range(m):
        if not row_wires[i]:
            outputs[i] = zero_wire
        elif len(row_wires[i]) == 1:
            outputs[i] = list(row_wires[i])[0]
        else:
            # If a row still has multiple wires, we need to create gates to combine them?
            # The greedy algorithm stops when no *single* new gate reduces total cost.
            # But we still need to output a single wire per row.
            # We must create a tree for each row's remaining wires.
            wires = list(row_wires[i])
            # Build a balanced XOR tree for these wires
            # Note: These wires are already existing wires. We just need to XOR them.
            # This will add more gates.
            
            # We need to be careful: adding these gates might create new masks that could be shared?
            # But we are at the end. Just build the tree.
            
            current_wires = wires
            while len(current_wires) > 1:
                new_current_wires = []
                for k in range(0, len(current_wires), 2):
                    if k + 1 < len(current_wires):
                        w1 = current_wires[k]
                        w2 = current_wires[k + 1]
                        # Check if mask(w1 ^ w2) exists
                        m1 = wire_masks[w1]
                        m2 = wire_masks[w2]
                        nm = m1 ^ m2
                        if nm in mask_to_wire:
                            new_current_wires.append(mask_to_wire[nm])
                        else:
                            # Create new gate
                            new_w = next_wire_id
                            next_wire_id += 1
                            gates.append([w1, w2])
                            wire_masks[new_w] = nm
                            mask_to_wire[nm] = new_w
                            new_current_wires.append(new_w)
                    else:
                        new_current_wires.append(current_wires[k])
                current_wires = new_current_wires
            outputs[i] = current_wires[0]

    return {'gates': gates, 'outputs': outputs}

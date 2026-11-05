def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    matrix = np.array(matrix, dtype=int) % 2
    m, n = matrix.shape
    
    # Represent each wire as a bitmask
    # Wire 0..n-1: input i -> mask 1 << i
    # Wire n: zero -> mask 0
    # Gate k -> wire n+1+k
    
    wire_masks = {}
    for i in range(n):
        wire_masks[i] = 1 << i
    wire_masks[n] = 0  # zero wire
    
    gates = []  # list of [wire_a, wire_b]
    next_wire = n + 1
    
    # For each row, we need to compute it
    # Strategy: greedy with wire merging
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # We'll process rows and build a circuit
    # Maintain a set of available wire masks for quick lookup
    # Use a dict: mask -> wire_id (first wire with that mask)
    mask_to_wire = {}
    for w, msk in wire_masks.items():
        if msk not in mask_to_wire:
            mask_to_wire[msk] = w
    
    outputs = []
    
    for row_mask in row_masks:
        if time.time() > deadline:
            # Fallback: compute each row independently
            break
        
        if row_mask == 0:
            outputs.append(n)  # zero wire
            continue
        
        # Try to express row_mask using existing wires
        # Greedy: find a set of wires whose XOR = row_mask
        # This is equivalent to: is row_mask in the span of existing wires?
        # If yes, find the combination. If no, we need new gates.
        
        # Actually, we can always express any mask using input wires.
        # The question is: can we do it with fewer gates by reusing intermediate wires?
        
        # Simple approach: reduce row_mask using the basis of available wires
        # Build a basis from available wire masks
        
        # Let's use a different approach:
        # Greedily pick wires to XOR together to get row_mask
        # Start with remaining = row_mask
        # At each step, pick a wire w such that (remaining XOR w) has lower weight
        # or is already available
        
        remaining = row_mask
        used_wires = []
        
        # Try to find a decomposition
        # Use a BFS/DFS on the state space? Too expensive.
        # Greedy: repeatedly pick a wire that reduces the Hamming weight
        
        max_iter = 100
        for _ in range(max_iter):
            if remaining == 0:
                break
            if time.time() > deadline:
                break
            
            best_wire = None
            best_remaining = remaining
            best_weight = bin(remaining).count('1')
            
            for w, msk in wire_masks.items():
                if msk == 0:
                    continue
                new_remaining = remaining ^ msk
                w_new = bin(new_remaining).count('1')
                if w_new < best_weight:
                    best_weight = w_new
                    best_wire = w
                    best_remaining = new_remaining
            
            if best_wire is None:
                break
            
            used_wires.append(best_wire)
            remaining = best_remaining
            
            # Check if remaining is already available
            if remaining in mask_to_wire:
                used_wires.append(mask_to_wire[remaining])
                remaining = 0
                break
        
        if remaining != 0:
            # Need to create new gates for the remaining bits
            # Find the set bits in remaining
            bits = []
            temp = remaining
            while temp:
                lb = temp & (-temp)
                bit_idx = lb.bit_length() - 1
                bits.append(bit_idx)
                temp ^= lb
            
            # Create gates to XOR these bits together
            if len(bits) == 1:
                # Just use the input wire
                outputs.append(bits[0])
            else:
                # Chain XOR gates
                # But first, check if any pair XOR is already available
                current = bits[0]
                for b in bits[1:]:
                    new_mask = wire_masks[current] ^ wire_masks[b]
                    if new_mask in mask_to_wire:
                        current = mask_to_wire[new_mask]
                    else:
                        # Create a new gate
                        gates.append([current, b])
                        current = next_wire
                        wire_masks[current] = new_mask
                        mask_to_wire[new_mask] = current
                        next_wire += 1
                outputs.append(current)
        else:
            # remaining == 0, so we have a decomposition using used_wires
            if not used_wires:
                outputs.append(n)  # zero
            elif len(used_wires) == 1:
                outputs.append(used_wires[0])
            else:
                # Need to XOR all used_wires together
                current = used_wires[0]
                for w in used_wires[1:]:
                    new_mask = wire_masks[current] ^ wire_masks[w]
                    if new_mask in mask_to_wire:
                        current = mask_to_wire[new_mask]
                    else:
                        gates.append([current, w])
                        current = next_wire
                        wire_masks[current] = new_mask
                        mask_to_wire[new_mask] = current
                        next_wire += 1
                outputs.append(current)
    
    # If we hit the deadline, fallback for remaining rows
    # ... (need to handle this)
    
    return {'gates': gates, 'outputs': outputs}

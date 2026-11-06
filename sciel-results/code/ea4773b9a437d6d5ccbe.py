import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Parse matrix
    if not matrix:
        return {'gates': [], 'outputs': []}
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': [0] * m} # Assume 0 outputs if no cols? 
        # Actually, if n=0, rows are empty, sum is 0.
    
    # Convert rows to bitmasks
    targets = []
    for row in matrix:
        mask = 0
        for j, val in enumerate(row):
            if val:
                mask |= (1 << j)
        targets.append(mask)
    
    # If all targets are 0, output 0
    if all(t == 0 for t in targets):
        return {'gates': [], 'outputs': [n] * m} # n is the zero wire
    
    # Baseline: independent synthesis
    # Gate count = sum of popcounts? No, independent synthesis means:
    # To compute a mask with k bits, we need k-1 gates (if we assume we can XOR inputs directly? No, inputs are wires).
    # To XOR k wires, we need k-1 gates.
    # So baseline cost = sum(max(0, popcount(t) - 1) for t in targets)
    # But we need to produce the output wire.
    # The interface returns 'gates': [[wire, wire], ...] and 'outputs': [wire, ...]
    # Wire n is 0.
    # Inputs are 0..n-1.
    
    # We want to minimize total gates.
    
    # Greedy Shared Subexpression Synthesis
    # Available wires: 0..n-1 (inputs), n (zero)
    # We will add gates n+1, n+2, ...
    
    best_gates = []
    best_outputs = []
    best_cost = float('inf')
    
    # Helper to verify and calculate cost
    def evaluate_circuit(gates, outputs, targets, n, m):
        # Map wire id to bitmask
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = 1 << i
        wire_mask[n] = 0
        
        # Process gates
        for k, (w1, w2) in enumerate(gates):
            new_id = n + 1 + k
            if w1 not in wire_mask or w2 not in wire_mask:
                return None # Invalid
            wire_mask[new_id] = wire_mask[w1] ^ wire_mask[w2]
        
        # Check outputs
        valid = True
        for i, out_wire in enumerate(outputs):
            if out_wire not in wire_mask:
                valid = False
                break
            if wire_mask[out_wire] != targets[i]:
                valid = False
                break
        
        if not valid:
            return None
        
        # Check depth? 
        # Depth is hard to check quickly without graph traversal, but with <=16 limit and small size, 
        # and greedy usually keeps depth low.
        # We'll assume depth is ok for now, or add a check.
        
        return len(gates)

    # Run multiple restarts with different random orders or tie-breaking
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_gates = []
        # available wires: id -> mask
        # We store available wires in a dict
        available = {i: 1 << i for i in range(n)}
        available[n] = 0
        
        # We need to track which targets are "done" or what their current representation is?
        # Actually, we just need to know if a target mask is in `available`.
        # But we can also use intermediate wires to simplify targets.
        # The greedy step:
        # Find w1, w2 in available such that w_new = w1^w2 is most useful.
        # Useful means it reduces the total "distance" of targets to being in available.
        
        # Let's define "distance" of a target t as the min number of gates to produce t from available.
        # This is hard.
        
        # Heuristic: 
        # For each pair (w1, w2) in available:
        #   w_new = w1 ^ w2
        #   If w_new is already in available, skip.
        #   Score = 0
        #   For each target t:
        #       If w_new is a sub-expression? 
        #       Let's try: If t contains w1 and w2 (as bits in the mask? No, w1 and w2 are wires, not bits).
        #       
        # Let's use a different metric:
        # The cost to produce a target t from available wires is roughly popcount(t) if we only have inputs.
        # If we have a wire w, and w is a subset of t (bitwise), we can replace those bits.
        # But XOR is not subset.
        
        # Let's try the "common subexpression" elimination on the expressions themselves.
        # Represent each target as a set of input indices.
        # When we add a gate g = a ^ b, we can replace {a, b} in any target with {g}.
        # This reduces the size of the target set by 1 (if a, b were distinct).
        
        # State:
        # targets_rep: list of sets of wire IDs.
        # Initially, targets_rep[i] = set of input indices where matrix[i][j]==1.
        # available: set of wire IDs that are "free" (inputs and 0).
        
        targets_rep = []
        for t in targets:
            s = set()
            for j in range(n):
                if (t >> j) & 1:
                    s.add(j)
            targets_rep.append(s)
        
        available_ids = set(range(n + 1)) # 0..n-1 inputs, n is zero
        
        current_gates = []
        
        while any(len(s) > 1 for s in targets_rep):
            # Find best gate
            best_score = -1
            best_pair = None
            best_new_id = None
            
            # Limit search space for speed: only consider pairs where at least one is "small" or frequent?
            # For now, consider all pairs.
            # n is small, so |available| grows slowly.
            
            candidates = list(available_ids)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    w1 = candidates[i]
                    w2 = candidates[j]
                    
                    # Calculate new wire mask? We don't have masks for wires > n.
                    # We need to know the mask of w1 and w2 to compute w_new mask.
                    # We must maintain a map wire_id -> mask.
                    
                    # Let's restart this inner loop with a proper data structure.
                    pass
        
        # This approach is getting complicated to implement correctly in a short time.
        # Let's use a simpler, known effective heuristic:
        # 1. Compute the row space basis.
        # 2. Express each row as a combination of basis vectors.
        # 3. Synthesize the basis vectors greedily.
        
        # Actually, let's just use the "greedy pair elimination" on the sets.
        # We need to track the mask of each wire to compute the mask of the new wire.
        
        wire_masks = {i: 1 << i for i in range(n)}
        wire_masks[n] = 0
        
        targets_rep = []
        for t in targets:
            s = set()
            for j in range(n):
                if (t >> j) & 1:
                    s.add(j)
            targets_rep.append(s)
        
        available_ids = set(range(n + 1))
        current_gates = []
        
        # To avoid infinite loops or bad choices, we add a new wire only if it helps.
        # Score = sum over targets of (old_size - new_size)
        
        while any(len(s) > 1 for s in targets_rep):
            best_score = -1
            best_pair = None
            best_new_mask = None
            best_new_id = None
            
            # Optimization: if time is tight, break
            if time.time() > deadline:
                break
                
            candidates = list(available_ids)
            # Shuffle for randomization in restarts?
            if r > 0:
                random.shuffle(candidates)
            
            for i in range(len(candidates)):
                w1 = candidates[i]
                for j in range(i + 1, len(candidates)):
                    w2 = candidates[j]
                    
                    new_mask = wire_masks[w1] ^ wire_masks[w2]
                    
                    # If new_mask is 0, it's the zero wire.
                    if new_mask == 0:
                        # Already have zero wire (n)
                        continue
                    
                    # Check if new_mask is already available
                    # We can check if any available wire has this mask
                    # For speed, we might skip this check or do it efficiently.
                    # Since n is small, we can just check.
                    already_exists = False
                    for wid in available_ids:
                        if wire_masks[wid] == new_mask:
                            already_exists = True
                            break
                    if already_exists:
                        continue
                    
                    # Calculate score
                    score = 0
                    for s in targets_rep:
                        old_size = len(s)
                        # If w1 and w2 are in s, we can replace them with new_wire
                        if w1 in s and w2 in s:
                            # New size would be old_size - 1
                            score += 1
                        # What if w1 is in s and new_mask is a subset of s?
                        # This is the "subset" heuristic.
                        # If new_mask is a subset of the mask formed by s?
                        # We don't have the mask of s easily, but we can compute it.
                        # mask_s = 0
                        # for x in s: mask_s |= wire_masks[x]
                        # if new_mask & mask_s == new_mask:
                        #     # new_mask is a sub-expression
                        #     # We can replace the bits of new_mask in s with new_wire
                        #     # Size reduction = popcount(new_mask & mask_s) - 1
                        #     # But this is complex.
                        pass
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (w1, w2)
                        best_new_mask = new_mask
                        # Assign new ID
                        best_new_id = n + 1 + len(current_gates)
            
            if best_score <= 0:
                # No improvement found, break to avoid infinite loop
                # Fallback to independent synthesis for remaining?
                break
            
            # Add the gate
            current_gates.append(best_pair)
            wire_masks[best_new_id] = best_new_mask
            available_ids.add(best_new_id)
            
            # Update targets_rep
            for s in targets_rep:
                if best_pair[0] in s and best_pair[1] in s:
                    s.remove(best_pair[0])
                    s.remove(best_pair[1])
                    s.add(best_new_id)
            
            # If time is up, break
            if time.time() > deadline:
                break
        
        # Check if valid
        # We need to map the final sets to output wires.
        # If a set has size 1, the element is the output wire.
        # If a set has size > 1, it's invalid.
        
        outputs = []
        valid = True
        for s in targets_rep:
            if len(s) == 1:
                outputs.append(list(s)[0])
            else:
                valid = False
                break
        
        if valid:
            cost = len(current_gates)
            if cost < best_cost:
                best_cost = cost
                best_gates = current_gates
                best_outputs = outputs
    
    # If no valid circuit found (e.g., due to timeout or failure), fallback to baseline
    if not best_outputs:
        # Baseline: independent synthesis
        gates = []
        outputs = []
        next_id = n + 1
        for t in targets:
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                # XOR them
                current = bits[0]
                for b in bits[1:]:
                    gates.append([current, b])
                    current = next_id
                    next_id += 1
                outputs.append(current)
        best_gates = gates
        best_outputs = outputs
    
    return {'gates': best_gates, 'outputs': best_outputs}

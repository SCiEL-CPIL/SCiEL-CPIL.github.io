import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using Minimum Weight Basis Synthesis.
    
    Strategy:
    1. Represent rows as bitmasks.
    2. Find a basis for the row space.
    3. Optimize the basis to minimize total synthesis cost:
       Cost = sum(weight(b_i) - 1 for b_i in basis) + sum(weight(coord(t)) - 1 for t in targets)
       Note: weight(coord) is the number of basis vectors needed to form t.
       We assume we can synthesize the basis vectors independently and then combine them.
       To minimize cost, we want basis vectors with low Hamming weight and targets that
       have sparse representations in the basis.
    4. Synthesize the chosen basis vectors.
    5. Synthesize each target as XOR of the required basis vectors.
    
    This approach is robust and fast. It avoids the exponential blowup of BFS.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Compute row weights
    row_weights = np.sum(mat, axis=1)
    
    # Baseline calculation for reference (not used directly in logic, but good to know)
    # baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if m == 0:
        return {'gates': [], 'outputs': []}
        
    if np.all(row_weights == 0):
        return {'gates': [], 'outputs': [zero_wire] * m}
    
    # Convert rows to bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
        
    # Unique targets to avoid redundant work in cost calculation? 
    # No, we need to output for every row. But synthesis of identical targets shares gates.
    
    # Step 1: Find an initial basis.
    # We want a basis that minimizes synthesis cost.
    # Heuristic: Pick independent vectors with the lowest Hamming weight first.
    
    # Sort indices by weight
    sorted_indices = sorted(range(m), key=lambda i: row_weights[i])
    
    basis_masks = []
    basis_pivots = {} # pivot_bit -> index in basis_masks
    
    for idx in sorted_indices:
        mask = target_masks[idx]
        if mask == 0:
            continue
            
        # Reduce mask using existing basis
        current_mask = mask
        
        # We need to reduce current_mask by basis vectors to see if it's independent.
        # To do this efficiently, we maintain basis vectors sorted by pivot bit.
        # However, since we are adding them one by one, we can just iterate.
        
        # Optimization: Only consider reduction if it helps? No, standard elimination.
        # We need to know the pivot of each basis vector.
        
        # Let's maintain a list of (pivot, mask)
        
        # Check independence
        is_independent = True
        temp_mask = current_mask
        
        # Reduce temp_mask by existing basis vectors
        # We should reduce by highest pivot first to ensure uniqueness of representation?
        # For independence check, any reduction order works if we track pivots correctly.
        
        # Let's just try to reduce temp_mask to 0.
        # If it becomes 0, it's dependent.
        
        # To make reduction fast, we can keep basis vectors in a list and iterate.
        # For n<=16, this is very fast.
        
        for pivot, b_mask in basis_pivots.items():
            if (temp_mask >> pivot) & 1:
                temp_mask ^= b_mask
                
        if temp_mask == 0:
            is_independent = False
        else:
            # Find the highest set bit in temp_mask (this will be the new pivot)
            # Actually, standard elimination uses the highest bit.
            # Let's find the highest set bit.
            highest_bit = temp_mask.bit_length() - 1
            is_independent = True
            new_pivot = highest_bit
            new_mask = temp_mask
            
    # The above loop has a bug: I didn't actually add to basis in the loop properly 
    # because I mixed up the logic. Let's rewrite the basis finding cleanly.
    
    basis_masks = []
    basis_pivots = {}
    
    for idx in sorted_indices:
        mask = target_masks[idx]
        if mask == 0:
            continue
            
        temp_mask = mask
        
        # Reduce by existing basis vectors
        # We iterate through existing pivots. 
        # Note: The order of reduction doesn't matter for the final reduced form 
        # if we are just checking if it reduces to 0, BUT to ensure we add 
        # a canonical representative, we should reduce by highest pivots first?
        # Actually, for independence, we just need to know if it's in the span.
        
        # Let's reduce by all existing basis vectors.
        # To be safe and consistent, let's sort basis by pivot descending before reduction?
        # Or just iterate.
        
        for pivot, b_mask in basis_pivots.items():
            if (temp_mask >> pivot) & 1:
                temp_mask ^= b_mask
                
        if temp_mask != 0:
            # It is independent.
            # The new pivot is the highest set bit in temp_mask?
            # Wait, if we reduced by arbitrary order, the highest set bit might not be unique?
            # In GF(2) linear algebra, if we maintain a row echelon form, 
            # each new independent vector should have a pivot that is not in the existing pivots.
            # The standard way is to find the highest bit that is set in temp_mask 
            # AND not already a pivot? 
            # Actually, if we reduce by all existing basis vectors, 
            # the resulting temp_mask will have 0s in all positions corresponding to existing pivots.
            # So the highest set bit of temp_mask is guaranteed to be a new pivot.
            
            new_pivot = temp_mask.bit_length() - 1
            basis_masks.append(temp_mask)
            basis_pivots[new_pivot] = temp_mask
            
    # Now we have a basis.
    # Step 2: Optimize the basis?
    # The current basis is "light" because we picked lightest independent vectors first.
    # Is it optimal? Not necessarily. 
    # Example: v1=1100 (w=2), v2=0011 (w=2), v3=1111 (w=4).
    # If targets are v3, cost with basis {v1, v2} is (1+1) + 1 (xor v1,v2) = 3.
    # If basis {v3}, cost is 3 + 0 = 3. Tie.
    # If target is v1, cost with {v1, v2} is 1. With {v3}, v1 = v3 ^ v2? No, v2 not in basis.
    # So the greedy "lightest first" is generally good.
    
    # Let's check if we can improve by swapping.
    # For now, let's stick with the greedy basis. It's fast and usually good.
    
    # Step 3: Synthesize the basis vectors.
    gates = []
    next_wire = n + 1
    
    # Wire for each basis vector
    basis_wires = []
    
    for b_mask in basis_masks:
        # Synthesize b_mask from inputs
        # b_mask is a bitmask of inputs.
        # We need to XOR the inputs set in b_mask.
        # Cost: weight(b_mask) - 1 gates.
        
        bits = [j for j in range(n) if (b_mask >> j) & 1]
        
        if not bits:
            # Should not happen for basis vectors (non-zero)
            basis_wires.append(zero_wire)
            continue
            
        if len(bits) == 1:
            basis_wires.append(bits[0])
            continue
            
        current_wire = bits[0]
        for j in range(1, len(bits)):
            gates.append([current_wire, bits[j]])
            current_wire = next_wire
            next_wire += 1
            
        basis_wires.append(current_wire)
        
    # Step 4: Synthesize each target.
    # For each target, find its coordinates in the basis.
    # Then XOR the corresponding basis wires.
    
    outputs = []
    
    for t_mask in target_masks:
        if t_mask == 0:
            outputs.append(zero_wire)
            continue
            
        # Reduce t_mask by basis vectors to find coordinates.
        # We need to track which basis vectors were used.
        
        coords = [0] * len(basis_masks)
        temp_mask = t_mask
        
        # To ensure we get the correct coordinates, we must reduce in a specific order.
        # The basis_pivots dict maps pivot -> mask.
        # We should reduce by highest pivot first to match the way we built the basis?
        # Actually, the basis vectors are in row echelon form (if we picked highest bit as pivot).
        # So we can reduce by highest pivot first.
        
        # Sort pivots descending
        sorted_pivots = sorted(basis_pivots.keys(), reverse=True)
        
        for pivot in sorted_pivots:
            if (temp_mask >> pivot) & 1:
                b_mask = basis_pivots[pivot]
                temp_mask ^= b_mask
                # Find index of this basis vector
                # We need a mapping from mask to index or pivot to index.
                # Let's create pivot_to_idx
                pass
        
        # We need pivot_to_idx
        pivot_to_idx = {p: i for i, p in enumerate([b_mask.bit_length()-1 for b_mask in basis_masks])}
        # Wait, the pivot of the reduced basis vector is the highest bit.
        # So we can just use the pivot key.
        
        # Re-do coordinate finding properly
        coords = [0] * len(basis_masks)
        temp_mask = t_mask
        
        for pivot in sorted_pivots:
            if (temp_mask >> pivot) & 1:
                temp_mask ^= basis_pivots[pivot]
                idx = pivot_to_idx[pivot]
                coords[idx] = 1
                
        # Now XOR the basis wires where coords[i] is 1
        active_wires = [basis_wires[i] for i in range(len(basis_masks)) if coords[i] == 1]
        
        if not active_wires:
            outputs.append(zero_wire)
        elif len(active_wires) == 1:
            outputs.append(active_wires[0])
        else:
            current_wire = active_wires[0]
            for w in active_wires[1:]:
                gates.append([current_wire, w])
                current_wire = next_wire
                next_wire += 1
            outputs.append(current_wire)
            
    return {'gates': gates, 'outputs': outputs}

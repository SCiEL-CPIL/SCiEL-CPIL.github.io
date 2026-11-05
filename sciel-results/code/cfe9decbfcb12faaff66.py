import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using Minimum Weight Basis Synthesis.
    
    Strategy:
    1. Convert input matrix rows to bitmasks.
    2. Find a minimum-weight basis for the row space over GF(2).
       - We prioritize basis vectors with lower Hamming weight to reduce synthesis cost.
       - Gaussian elimination is used to maintain independence and track pivots.
    3. Synthesize each basis vector using a chain of XORs (cost = weight - 1).
    4. For each original target row, determine its coordinates in the basis (which basis vectors it is an XOR of).
    5. Synthesize each target row by XORing the corresponding basis wires (cost = num_coords - 1).
    6. This approach maximizes sharing of the fundamental linear structure, often outperforming 
       naive per-row synthesis or unrestricted BFS which may create redundant heavy intermediates.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape

    zero_wire = n

    # Handle trivial case: no gates needed
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Compute row weights
    row_weights = np.sum(mat, axis=1)
    
    # Trivial rows (0 or 1 bit)
    trivial_outputs = [None] * m
    non_trivial_indices = []
    for i in range(m):
        if row_weights[i] == 0:
            trivial_outputs[i] = zero_wire
        elif row_weights[i] == 1:
            idx = np.where(mat[i] == 1)[0][0]
            trivial_outputs[i] = int(idx)
        else:
            non_trivial_indices.append(i)

    if not non_trivial_indices:
        return {'gates': [], 'outputs': trivial_outputs}

    # Convert non-trivial rows to bitmasks
    # We only need to synthesize the non-trivial ones
    # But wait, the basis must span the *entire* row space to express all rows?
    # Yes. If a trivial row is dependent on non-trivial ones, it's fine.
    # If a non-trivial row is dependent on trivial ones? 
    # A row with weight > 1 cannot be in the span of rows with weight <= 1 
    # unless it is a single basis vector itself or sum of them?
    # Actually, if we have inputs e1, e2. Row [1,1] is e1^e2. 
    # If we have row [1,0] and [0,1], they are trivial.
    # If we have row [1,1], it is non-trivial.
    # The basis of the *whole* matrix includes trivial rows if they are independent.
    # However, trivial rows cost 0 gates to produce (they are just wires).
    # So we should include them in the "available" set but not in the "gates" count.
    
    # Let's refine:
    # Available wires initially: 0..n-1 (inputs) and n (zero).
    # We want to find a set of "new" wires (gates) that spans the target rows.
    # The basis of the row space can be partitioned into:
    # 1. Rows that are already available (weight 0 or 1).
    # 2. Rows that need synthesis (weight > 1).
    
    # We will find a basis for the *entire* row space.
    # We will synthesize the basis vectors that are not already available.
    # Then express all targets in terms of this basis.
    
    # Step 1: Build Basis
    # We want a basis that minimizes the cost of synthesis.
    # Cost of a basis vector b is:
    #   0 if weight(b) <= 1
    #   weight(b) - 1 otherwise
    # Plus the cost to express targets.
    
    # Strategy: 
    # 1. Collect all unique non-zero masks from the matrix.
    # 2. Sort them by weight (ascending).
    # 3. Perform Gaussian elimination to select a basis.
    #    - Prefer lower weight vectors for the basis.
    #    - This ensures that the "building blocks" we synthesize are as light as possible.
    
    # Get unique masks
    all_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        if mask != 0:
            all_masks.append(mask)
            
    # Unique masks
    unique_masks = list(set(all_masks))
    
    # Sort by weight, then by value (for determinism)
    unique_masks.sort(key=lambda x: (bin(x).count('1'), x))
    
    basis = [] # List of masks
    basis_pivots = {} # pivot_bit -> index in basis
    
    # Gaussian Elimination to find basis
    # We iterate through sorted masks.
    # For each mask, we reduce it by existing basis vectors.
    # If it reduces to 0, it's dependent.
    # If not, it's independent, and we add it to the basis.
    
    # To reduce efficiently, we maintain basis vectors in a way that allows quick pivot lookup.
    # Standard approach: for each bit position, store the basis vector with that pivot.
    
    for mask in unique_masks:
        current_mask = mask
        # Reduce current_mask by basis vectors
        # We need to know which basis vector to use for each bit.
        # Let's iterate through bits of current_mask.
        # Actually, standard elimination:
        # For bit from high to low:
        #   If bit is set:
        #     If there is a basis vector with this pivot, XOR it.
        #     Else, this bit is the new pivot.
        
        # We need a fast lookup: bit -> basis_vector
        # We have basis_pivots dict.
        
        # Let's do the reduction.
        # Note: basis_pivots maps pivot_bit -> index in basis list.
        
        # We must be careful: if we add a new basis vector, its pivot is the highest set bit 
        # after reduction? Or the bit we chose?
        # In standard REF, the pivot is the first (highest) non-zero column.
        
        # Let's find the highest set bit of current_mask.
        # But we reduce iteratively.
        
        # Efficient reduction:
        # while current_mask != 0:
        #   highest_bit = current_mask.bit_length() - 1
        #   if highest_bit in basis_pivots:
        #       current_mask ^= basis[basis_pivots[highest_bit]]
        #   else:
        #       # New basis vector
        #       break
       
        while current_mask != 0:
            highest_bit = current_mask.bit_length() - 1
            if highest_bit in basis_pivots:
                current_mask ^= basis[basis_pivots[highest_bit]]
            else:
                # This mask is independent. Add to basis.
                # The pivot is highest_bit.
                basis_pivots[highest_bit] = len(basis)
                basis.append(current_mask)
                break
        
    # Now we have a basis.
    # Synthesize the basis vectors.
    gates = []
    next_wire = n + 1
    basis_wire = {} # mask -> wire index
    
    # First, map trivial basis vectors to existing wires
    for b_mask in basis:
        w = bin(b_mask).count('1')
        if w == 0:
            basis_wire[b_mask] = zero_wire
        elif w == 1:
            # Find the bit
            idx = (b_mask.bit_length() - 1)
            basis_wire[b_mask] = idx
        else:
            # Synthesize
            # Get list of set bits
            bits = []
            temp = b_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                bits.append(idx)
                temp ^= lsb
            
            # XOR them together
            current_wire = bits[0]
            for b in bits[1:]:
                gates.append([current_wire, b])
                current_wire = next_wire
                next_wire += 1
            basis_wire[b_mask] = current_wire
            
    # Now express each target row in terms of the basis.
    # For each row, we reduce it to 0 using the basis, tracking which basis vectors were used.
    
    outputs = [0] * m
    
    for i in range(m):
        if trivial_outputs[i] is not None:
            outputs[i] = trivial_outputs[i]
            continue
            
        # Reconstruct mask for row i
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
                
        # If mask is 0
        if mask == 0:
            outputs[i] = zero_wire
            continue
            
        # Reduce mask using basis, tracking coordinates
        # coords is a list of basis indices used
        coords = []
        current_mask = mask
        
        while current_mask != 0:
            highest_bit = current_mask.bit_length() - 1
            if highest_bit in basis_pivots:
                b_idx = basis_pivots[highest_bit]
                current_mask ^= basis[b_idx]
                coords.append(b_idx)
            else:
                # This should not happen if the basis spans the row space.
                # If it does, we have a bug or the matrix row wasn't in the span (impossible).
                # For safety, if we can't reduce, we synthesize directly? 
                # But basis should span.
                pass
        
        # Now we have a list of basis indices.
        # We need to XOR the wires corresponding to these basis vectors.
        if not coords:
            outputs[i] = zero_wire
        elif len(coords) == 1:
            outputs[i] = basis_wire[basis[coords[0]]]
        else:
            # XOR the wires
            wires = [basis_wire[basis[c]] for c in coords]
            current_wire = wires[0]
            for w in wires[1:]:
                gates.append([current_wire, w])
                current_wire = next_wire
                next_wire += 1
            outputs[i] = current_wire
            
    return {'gates': gates, 'outputs': outputs}

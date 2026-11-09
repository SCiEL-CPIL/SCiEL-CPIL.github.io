import numpy as np
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Convert to numpy array if not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    if matrix.size == 0:
        return {'gates': [], 'outputs': []}
        
    # Ensure binary
    matrix = matrix.astype(int) % 2
    
    n, m = matrix.shape
    
    # Baseline: Sum of (weight - 1) for each column
    # If weight is 0, cost is 0 (constant 0 is wire n, no gate needed)
    # If weight is 1, cost is 0 (input wire, no gate needed)
    # If weight is k, cost is k-1
    baseline_gates = 0
    for j in range(m):
        col = matrix[:, j]
        w = np.sum(col)
        if w > 1:
            baseline_gates += w - 1
        elif w == 0:
            # Constant 0, wire n
            pass
        else:
            # Input wire
            pass
            
    # If baseline is 0, return empty
    if baseline_gates == 0:
        return {'gates': [], 'outputs': []}

    # Compute Rank and Basis
    # Use Gaussian Elimination on columns? Or rows?
    # We want to express columns as combinations of a basis.
    # So we find a basis for the column space.
    
    # We'll use a list of (mask, index) for the basis
    # Mask is an integer representing the vector
    
    def vec_to_mask(v):
        # v is a numpy array of 0s and 1s
        # Convert to integer
        mask = 0
        for i, bit in enumerate(v):
            if bit:
                mask |= (1 << i)
        return mask

    def mask_to_vec(mask, n):
        v = np.zeros(n, dtype=int)
        for i in range(n):
            if mask & (1 << i):
                v[i] = 1
        return v

    # Extract columns as masks
    col_masks = []
    for j in range(m):
        col_masks.append(vec_to_mask(matrix[:, j]))
        
    # Find basis for col_masks
    basis = []
    # Sort by weight? Or just process in order.
    # To make basis generation cheap, we might want sparse basis.
    # But for now, standard GE.
    
    for mask in col_masks:
        if mask == 0:
            continue
        # Try to eliminate
        current = mask
        for b in basis:
            if current & b:
                # Find pivot bit
                # This simple GE might be slow if n is large, but n is <= 16
                # Actually, n is up to 16, so masks are small.
                # We need a proper GE.
                pass
        
    # Proper GE for basis
    basis = []
    for mask in col_masks:
        if mask == 0:
            continue
        current = mask
        for b in basis:
            # If b has a bit that current has, and b is "smaller" or we just use first bit?
            # Let's use the highest set bit as pivot.
            # Actually, standard GE:
            # For each existing basis vector, if current shares a pivot, XOR it.
            # We need to store pivots.
            pass
            
    # Let's use a simpler approach for n <= 16
    # Just store basis vectors and check independence.
    basis = []
    for mask in col_masks:
        if mask == 0:
            continue
        independent = True
        for b in basis:
            # Check if mask is in span of basis?
            # For small n, we can just do GE on the fly.
            pass
            
    # Efficient GE for small n:
    # Maintain a list of basis vectors.
    # For each new vector, try to reduce it by existing basis vectors.
    # If it becomes 0, it's dependent.
    # If not, add to basis.
    
    basis = []
    # Sort masks by number of bits to get sparse basis?
    # This helps in minimizing the weight of basis vectors.
    sorted_masks = sorted([m for m in col_masks if m != 0], key=lambda x: bin(x).count('1'))
    
    for mask in sorted_masks:
        current = mask
        for b in basis:
            # If the highest set bit of current is also in b, XOR
            # Find highest set bit
            # n is small, so we can iterate
            # Actually, let's just check if current & b != 0 and use the lowest set bit?
            # Better: standard GE with pivots.
            pass
            
    # Let's implement a robust GE
    def get_basis(masks):
        basis = []
        for mask in masks:
            if mask == 0:
                continue
            current = mask
            for b in basis:
                # If current and b share the lowest set bit, XOR
                # Find lowest set bit of current
                lsb_current = (current & -current).bit_length() - 1
                lsb_b = (b & -b).bit_length() - 1
                if lsb_current == lsb_b:
                    current ^= b
            if current != 0:
                basis.append(current)
        return basis
        
    basis = get_basis(col_masks)
    r = len(basis)
    
    if r == 0:
        return {'gates': [], 'outputs': []}
        
    # Now, express each col_mask as a combination of basis
    # And generate the circuit
    
    # We will generate the basis vectors first.
    # Then generate the outputs.
    
    gates = []
    wire_id = n + 1  # First gate output wire
    
    def add_gate(w1, w2):
        nonlocal wire_id
        gates.append([w1, w2])
        out = wire_id
        wire_id += 1
        return out
        
    # 1. Generate Basis Vectors
    # We need to map basis vector masks to wire IDs
    basis_wires = {}
    
    # To generate a basis vector from inputs:
    # We can just XOR the inputs corresponding to the bits.
    # This is not optimal if basis vectors share inputs, but it's valid.
    # However, we can do better by reusing intermediate results?
    # For now, let's just compute them directly.
    # If this cost is higher than baseline, we will fall back to baseline.
    
    for b_mask in basis:
        # Find inputs
        inputs = []
        for i in range(n):
            if b_mask & (1 << i):
                inputs.append(i)
        if not inputs:
            # Should not happen
            basis_wires[b_mask] = n
            continue
        if len(inputs) == 1:
            basis_wires[b_mask] = inputs[0]
        else:
            # XOR them
            w = inputs[0]
            for i in range(1, len(inputs)):
                w = add_gate(w, inputs[i])
            basis_wires[b_mask] = w
            
    # 2. Generate Outputs
    outputs = []
    for j in range(m):
        target_mask = col_masks[j]
        if target_mask == 0:
            outputs.append(n)
            continue
            
        # Find coefficients
        # Solve target_mask = sum c_i * b_i
        # Since basis is independent, we can do GE again to find coeffs.
        # Or, since we have the basis, we can just check which basis vectors are needed.
        # Actually, the basis we found is a subset of the original columns (or reduced).
        # Wait, the basis vectors are the reduced forms.
        # We need to express the *original* target_mask in terms of the *basis* vectors.
        
        # Let's do a simple back-substitution or GE on the system.
        # System: B * C = T
        # Where B is basis vectors (columns), C is coeffs, T is target.
        # Since n is small, we can just iterate.
        
        # Actually, we can just use the fact that the basis was derived from the columns.
        # But it's safer to solve.
        
        # Solve for coeffs
        coeffs = [0] * r
        current = target_mask
        # We need to express current in terms of basis.
        # Since basis is in row-echelon-like form (lowest bit pivots),
        # we can process from highest pivot to lowest?
        # Actually, my get_basis used lowest bit pivots.
        # So we can recover coeffs by processing bits from high to low?
        # No, if basis vectors have unique lowest bits, we can just check.
        
        # Let's just use a simple loop to find a combination.
        # Since r is small, we can try all 2^r combinations? No, r can be up to 16.
        # Better: Use the GE property.
        # The basis vectors have unique lowest set bits.
        # So, if target has bit k, and basis vector b_i has lowest bit k,
        # then coeff[i] must be 1.
        
        # Map lowest bit to basis index
        lsb_to_idx = {}
        for i, b in enumerate(basis):
            lsb = (b & -b).bit_length() - 1
            lsb_to_idx[lsb] = i
            
        # Reconstruct
        temp = target_mask
        for i in range(r):
            # Check if temp has the pivot bit of basis[i]
            lsb = (basis[i] & -basis[i]).bit_length() - 1
            if temp & (1 << lsb):
                coeffs[i] = 1
                temp ^= basis[i]
                
        # If temp != 0, target is not in span? Should not happen.
        
        # Now compute the output wire
        if not any(coeffs):
            outputs.append(n)
        else:
            # Find the basis wires to XOR
            w = basis_wires[basis[0]] if coeffs[0] else None
            for i in range(1, r):
                if coeffs[i]:
                    if w is None:
                        w = basis_wires[basis[i]]
                    else:
                        w = add_gate(w, basis_wires[basis[i]])
            outputs.append(w)
            
    # Compare gate count
    my_gates = len(gates)
    
    if my_gates < baseline_gates:
        return {'gates': gates, 'outputs': outputs}
    else:
        # Return baseline circuit
        # We need to construct the baseline circuit too
        # Or just return the baseline if we didn't build it?
        # We should build the baseline circuit to return it.
        
        base_gates = []
        base_wire_id = n + 1
        def add_base_gate(w1, w2):
            nonlocal base_wire_id
            base_gates.append([w1, w2])
            out = base_wire_id
            base_wire_id += 1
            return out
            
        base_outputs = []
        for j in range(m):
            col = matrix[:, j]
            inputs = [i for i in range(n) if col[i] == 1]
            if not inputs:
                base_outputs.append(n)
            elif len(inputs) == 1:
                base_outputs.append(inputs[0])
            else:
                w = inputs[0]
                for i in range(1, len(inputs)):
                    w = add_base_gate(w, inputs[i])
                base_outputs.append(w)
                
        return {'gates': base_gates, 'outputs': base_outputs}


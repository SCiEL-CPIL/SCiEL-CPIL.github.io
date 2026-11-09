"""Exact XOR-circuit synthesis using Linear Basis Decomposition."""

def synthesize(matrix, depth_limit):
    import numpy as np
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
        
    # Wire numbering: 0..n-1 are inputs, n is constant zero
    # Gates produce wires n+1, n+2, ...
    
    gates = []
    
    def new_gate(a, b):
        gates.append([int(a), int(b)])
        return n + 1 + (len(gates) - 1)
        
    # Step 1: Compute a basis for the row space of the matrix over GF(2)
    # We use Gaussian Elimination.
    
    # Convert rows to bitmasks for efficient GF(2) operations
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
        
    # Find Basis and Pivot Columns
    basis_masks = []
    pivot_cols = []
    active_rows = list(rows)
    
    for c in range(n):
        # Find a row with bit c set
        pivot_row_idx = -1
        for i, mask in enumerate(active_rows):
            if (mask >> c) & 1:
                pivot_row_idx = i
                break
        
        if pivot_row_idx == -1:
            continue
            
        pivot_mask = active_rows[pivot_row_idx]
        basis_masks.append(pivot_mask)
        pivot_cols.append(c)
        
        # Eliminate this bit from other active rows
        new_active = []
        for i, mask in enumerate(active_rows):
            if i == pivot_row_idx:
                continue
            if (mask >> c) & 1:
                new_active.append(mask ^ pivot_mask)
            else:
                new_active.append(mask)
        active_rows = new_active
        
    # Step 2: Compute coordinates of each original row in terms of the basis
    # The coordinate of row j for basis vector k is the bit of row j at pivot_cols[k]
    coords = []
    for j in range(m):
        row_mask = rows[j]
        coord_mask = 0
        for k, pc in enumerate(pivot_cols):
            if (row_mask >> pc) & 1:
                coord_mask |= (1 << k)
        coords.append(coord_mask)
        
    # Step 3: Build the circuit
    # 1. Compute wires for each basis vector
    basis_wires = []
    for b_mask in basis_masks:
        indices = []
        for i in range(n):
            if (b_mask >> i) & 1:
                indices.append(i)
                
        if len(indices) == 0:
            wire = n # zero
        elif len(indices) == 1:
            wire = indices[0]
        else:
            wire = indices[0]
            for idx in indices[1:]:
                wire = new_gate(wire, idx)
        basis_wires.append(wire)
        
    # 2. Compute outputs by XORing relevant basis wires
    outputs = []
    for c_mask in coords:
        indices = []
        for k in range(len(basis_masks)):
            if (c_mask >> k) & 1:
                indices.append(k)
                
        if len(indices) == 0:
            outputs.append(n)
        elif len(indices) == 1:
            outputs.append(basis_wires[indices[0]])
        else:
            wire = basis_wires[indices[0]]
            for k in indices[1:]:
                wire = new_gate(wire, basis_wires[k])
            outputs.append(wire)
            
    return {'gates': gates, 'outputs': outputs}

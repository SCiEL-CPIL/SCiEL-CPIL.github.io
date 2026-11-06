"""Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline over 8x8,12x16,16x16 maps."""
import time
import numpy as np

def _rref_gf2_with_basis(M):
    """
    Compute RREF of M over GF(2) and track the basis transformation.
    Returns:
        rref: RREF of M
        basis_indices: indices of the pivot rows in the original M
        basis_vectors: the actual rows from M corresponding to the pivots
    """
    M = np.array(M, dtype=np.uint8).copy()
    n_rows, n_cols = M.shape
    if n_rows == 0:
        return np.zeros((0, n_cols), dtype=np.uint8), [], []
    
    rank = 0
    pivot_cols = []
    basis_indices = []
    
    # We will perform Gaussian elimination on a copy to get RREF,
    # but we also need to know which original rows formed the basis.
    # Standard RREF algorithm:
    for col in range(n_cols):
        if rank >= n_rows:
            break
            
        # Find pivot row
        pivot_row = -1
        for row in range(rank, n_rows):
            if M[row, col] == 1:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
            
        # Swap rows
        M[[rank, pivot_row]] = M[[pivot_row, rank]]
        
        # Record the basis index (the row that ended up at 'rank' position)
        # Note: In standard RREF, the rows get swapped. 
        # To track the original basis vectors correctly for synthesis, 
        # it is safer to just use the RREF rows themselves as the basis 
        # for the intermediate wires, because the RREF rows are linearly 
        # independent and span the same space. 
        # However, the task asks for exact synthesis of the INPUT matrix rows.
        # So we need to express each input row as a combination of the RREF basis.
        
        pivot_cols.append(col)
        
        # Eliminate all other rows (to get Reduced form)
        for row in range(n_rows):
            if row != rank and M[row, col] == 1:
                M[row, :] ^= M[rank, :]
                
        rank += 1
        
    # The first 'rank' rows of M are now in RREF
    rref = M[:rank, :]
    
    # To find the expression of original rows in terms of RREF basis:
    # We need to solve A * x = b for each row b of original M, where A is the RREF matrix (stacked).
    # Actually, since RREF is obtained by row operations, the original rows are linear combinations of the RREF rows.
    # Let R be the RREF matrix (rank x n_cols).
    # We want coefficients c such that original_row = c @ R.
    # Since R is in RREF, the pivot columns form an identity matrix.
    # We can extract the coefficients by looking at the values in the pivot columns of the original rows?
    # No, that's only if we did the operations on the original rows and kept track.
    
    # Better approach:
    # 1. Compute RREF of M. Let this be R (rank x n_cols).
    # 2. The row space of M is the row space of R.
    # 3. For each row m_i of M, find vector c_i such that m_i = c_i @ R.
    #    This is a system of linear equations: R^T c_i^T = m_i^T.
    #    Since R has full row rank, R^T has full column rank.
    #    We can solve this using Gaussian elimination on the augmented matrix [R^T | m_i^T].
    #    Or, simpler: Since R is in RREF, the columns corresponding to pivot indices form an identity matrix.
    #    Let P be the set of pivot column indices.
    #    Then for any row v in the row space of R, the coefficients c are simply v[0:len(P)] mapped to the pivot positions?
    #    Let's verify. If R is RREF, row j of R has a 1 at pivot col p_j and 0s at other pivot cols.
    #    So if v = sum c_j * R[j], then v[p_j] = c_j.
    #    Therefore, the coefficients c_j are exactly the values of v at the pivot column indices!
    
    # So:
    # 1. Identify pivot columns from RREF computation.
    # 2. For each original row, the coefficients are the values at those pivot columns.
    
    # Re-do RREF to cleanly get pivot columns
    M_work = np.array(M, dtype=np.uint8).copy()
    n_rows, n_cols = M_work.shape
    rank = 0
    pivot_cols = []
    
    for col in range(n_cols):
        if rank >= n_rows:
            break
        pivot_row = -1
        for row in range(rank, n_rows):
            if M_work[row, col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        M_work[[rank, pivot_row]] = M_work[[pivot_row, rank]]
        for row in range(n_rows):
            if row != rank and M_work[row, col] == 1:
                M_work[row, :] ^= M_work[rank, :]
        rank += 1
        pivot_cols.append(col)
        
    rref = M_work[:rank, :]
    
    # If rank is 0, all rows are zero
    if rank == 0:
        return rref, pivot_cols, []
        
    # Extract coefficients for each original row
    # coeffs[i] is a vector of length 'rank' indicating which basis rows to XOR
    coeffs = []
    for i in range(n_rows):
        # The coefficient for basis row j is the value of original row i at pivot_cols[j]
        c = M[i, pivot_cols]  # This is a numpy array of length 'rank'
        coeffs.append(c)
        
    return rref, pivot_cols, coeffs

def _build_optimized_xor_tree(wires_to_xor):
    """
    Build an XOR tree for a list of wires.
    Returns (gates, output_wire).
    Uses a simple balanced tree. For small sets, this is optimal.
    """
    if not wires_to_xor:
        return [], None
    if len(wires_to_xor) == 1:
        return [], wires_to_xor[0]
        
    gates = []
    current_level = list(wires_to_xor)
    next_wire_id = 0 # Placeholder, will be managed by caller
    
    # We need to assign new wire IDs. Let's return the structure and let caller assign IDs.
    # Or better, pass in the current max wire ID and return new gates and final wire.
    
    # Let's restructure to take a starting wire ID and return gates, final wire, next ID.
    return None # This helper is tricky with state. Let's inline the logic in synthesize.

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    
    Args:
        matrix: list of lists or 2D array, n x m binary matrix
        depth_limit: maximum circuit depth (must be <= 16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # 1. Compute RREF and coefficients
    rref, pivot_cols, coeffs = _rref_gf2_with_basis(M)
    rank = len(pivot_cols)
    
    gates = []
    next_wire = n_cols + 1  # n_cols is zero wire, gates start at n_cols+1
    
    # 2. Synthesize the basis vectors (rows of RREF)
    # Each basis vector is a row in RREF. It has 1s at some positions.
    # We need to compute the XOR of the input wires corresponding to the 1s in each RREF row.
    # We can share subexpressions between basis vectors if they have overlapping 1s.
    # However, RREF rows have disjoint pivot columns. They may share free columns.
    # For simplicity and robustness, let's first compute each basis row independently.
    # Optimization: If a basis row has weight 0, it's the zero wire.
    # If weight 1, it's the input wire.
    # If weight > 1, build an XOR tree.
    
    basis_wires = []
    
    # To allow sharing, we could use a dictionary of computed subsets.
    # But RREF rows are often sparse. Let's start with independent trees for basis rows.
    # This is safe and often good enough.
    
    for i in range(rank):
        row = rref[i, :]
        ones = np.where(row == 1)[0].tolist()
        
        if len(ones) == 0:
            basis_wires.append(n_cols) # Zero wire
            continue
            
        if len(ones) == 1:
            basis_wires.append(ones[0])
            continue
            
        # Build XOR tree for this row
        # Simple balanced tree
        current = ones[:]
        while len(current) > 1:
            new_level = []
            for j in range(0, len(current) - 1, 2):
                w1 = current[j]
                w2 = current[j+1]
                gates.append([w1, w2])
                new_wire = next_wire
                next_wire += 1
                new_level.append(new_wire)
            if len(current) % 2 == 1:
                new_level.append(current[-1])
            current = new_level
            
        basis_wires.append(current[0])
        
        if time.time() > deadline:
            break

    # 3. Synthesize original outputs using basis_wires
    outputs = []
    for i in range(n_rows):
        c = coeffs[i]
        # Find which basis wires are selected (c[j] == 1)
        selected_wires = []
        for j in range(rank):
            if c[j] == 1:
                selected_wires.append(basis_wires[j])
                
        if len(selected_wires) == 0:
            outputs.append(n_cols)
            continue
        if len(selected_wires) == 1:
            outputs.append(selected_wires[0])
            continue
            
        # Build XOR tree for the selected basis wires
        current = selected_wires[:]
        while len(current) > 1:
            new_level = []
            for j in range(0, len(current) - 1, 2):
                w1 = current[j]
                w2 = current[j+1]
                gates.append([w1, w2])
                new_wire = next_wire
                next_wire += 1
                new_level.append(new_wire)
            if len(current) % 2 == 1:
                new_level.append(current[-1])
            current = new_level
            
        outputs.append(current[0])
        
        if time.time() > deadline:
            # If we time out, we might have incomplete outputs.
            # Pad with zero wires to maintain shape? Or just return what we have?
            # The contract implies we must return valid outputs for all rows.
            # If we break early, we should fill the rest with 0 (zero wire) to be safe,
            # though this will be incorrect for those rows. 
            # Better to ensure we don't break unless absolutely necessary.
            # Given the small sizes (16x16), this loop is very fast.
            pass

    return {'gates': gates, 'outputs': outputs}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Convert to numpy array for efficient operations
    M = np.array(matrix, dtype=int)
    n_cols = M.shape[1]
    n_rows = M.shape[0]
    
    # If matrix is empty or trivial
    if n_cols == 0:
        return {'gates': [], 'outputs': []}
    if n_rows == 0:
        return {'gates': [], 'outputs': [0] * n_cols}

    # We want to find a basis for the column space of M.
    # Let B be the basis matrix (n_rows x r).
    # Let C be the coefficient matrix (r x n_cols) such that M = B @ C.
    # Cost = sum(weight(B_col) - 1) + sum(weight(C_col) - 1)
    
    # Step 1: Find a basis using Gaussian Elimination on M^T
    # We work in GF(2)
    
    def rref_gf2(mat):
        """Compute RREF of a binary matrix in GF(2)."""
        mat = mat.copy()
        rows, cols = mat.shape
        pivot_cols = []
        r = 0
        for c in range(cols):
            if r >= rows:
                break
            # Find a pivot in column c
            pivot = -1
            for i in range(r, rows):
                if mat[i, c] == 1:
                    pivot = i
                    break
            if pivot == -1:
                continue
            
            # Swap rows
            mat[[r, pivot]] = mat[[pivot, r]]
            
            # Eliminate other entries in this column
            for i in range(rows):
                if i != r and mat[i, c] == 1:
                    mat[i] = (mat[i] + mat[r]) % 2
            
            pivot_cols.append(c)
            r += 1
        
        return mat, pivot_cols

    # We need the basis of the column space of M.
    # This is the row space of M^T.
    MT = M.T
    rref_MT, pivot_rows = rref_gf2(MT)
    
    # The non-zero rows of rref_MT are the basis vectors for the column space of M.
    # Let's extract them.
    r = len(pivot_rows)
    if r == 0:
        # All zeros
        return {'gates': [], 'outputs': [n_rows] * n_cols} # n_rows is the zero wire? 
        # Wait, constant-zero wire is n (where n is number of inputs). 
        # Inputs are 0..n-1. Zero wire is n.
        # Outputs are wires. If output is 0, we can just output the zero wire.
        # But the interface expects 'outputs': [wire, ...].
        # If an output is identically 0, we map it to the zero wire (index n).
        
    basis_vectors = rref_MT[:r, :] # These are 1 x n_rows vectors? No, rref_MT is n_cols x n_rows?
    # M is n_rows x n_cols. MT is n_cols x n_rows.
    # rref_MT is n_cols x n_rows.
    # The non-zero rows of rref_MT are the basis for the row space of MT, which is the column space of M.
    # So basis_vectors is r x n_rows.
    
    # Now we need C such that M = B @ C.
    # B is n_rows x r.
    B = basis_vectors.T # n_rows x r
    
    # Solve B @ C = M in GF(2).
    # Since B has full column rank, C is unique?
    # Actually, C represents the coordinates of each column of M in the basis B.
    # We can find C by solving the system for each column of M.
    # Or simpler: Since B is in RREF form (columns are pivot columns of MT),
    # the coefficients are directly available from the RREF process?
    
    # Let's just compute C explicitly.
    # B is n_rows x r. M is n_rows x n_cols.
    # We want C (r x n_cols) such that B @ C = M.
    # This is equivalent to C = (B^T B)^-1 B^T M? No, in GF(2) with full rank B,
    # we can perform Gaussian elimination on [B | M] to get [I | C].
    
    def solve_gf2(B, M):
        """Solve B @ C = M in GF(2). B is n x r (full col rank), M is n x k. Returns C (r x k)."""
        n, r = B.shape
        k = M.shape[1]
        
        # Augmented matrix [B | M]
        A = np.hstack((B, M))
        
        # Gaussian elimination to get B to identity
        # We assume B has full column rank r.
        # We need to find row operations to transform B to I_r (top part).
        
        pivot_cols = []
        row = 0
        for c in range(r):
            # Find pivot in column c
            pivot = -1
            for i in range(row, n):
                if A[i, c] == 1:
                    pivot = i
                    break
            if pivot == -1:
                # This shouldn't happen if B is full rank and we process correctly
                # But B might not have pivots in order 0..r-1 if we didn't permute.
                # Wait, B is constructed from RREF of MT.
                # The columns of B correspond to the pivot rows of MT.
                # Actually, let's just use the property that B is a basis.
                # We can solve by iterating.
                continue
                
            A[[row, pivot]] = A[[pivot, row]]
            
            for i in range(n):
                if i != row and A[i, c] == 1:
                    A[i] = (A[i] + A[row]) % 2
            
            pivot_cols.append(c)
            row += 1
        
        # The first r rows of the augmented matrix should now have I_r in the first r columns?
        # Not necessarily, because we didn't normalize the pivot to 1 (it is 1 in GF2)
        # and we didn't eliminate above/below properly for all columns if we skipped some.
        # Actually, the standard RREF algorithm ensures that if B is full rank,
        # we can reduce it to I_r.
        
        # Let's extract C. The last k columns of the top r rows?
        # Only if the pivot columns of B are exactly 0..r-1.
        # In our construction, B comes from RREF of MT.
        # The rows of B are the non-zero rows of RREF(MT).
        # The columns of B are the original columns of M (which are rows of MT).
        # This is getting confusing.
        
        # Alternative: Just use numpy linalg? No, GF(2).
        # Let's re-calculate C by expressing each column of M as a sum of columns of B.
        # Since B is a basis, this is unique.
        
        # We can form the matrix [B | M] and reduce B to I.
        # The corresponding operations on M will give C.
        
        # Let's do it properly.
        A = np.hstack((B, M))
        r = B.shape[1]
        
        # Perform Gauss-Jordan on the first r columns of A
        for col in range(r):
            # Find row with 1 in col
            pivot_row = -1
            for row_idx in range(r):
                if A[row_idx, col] == 1:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                # Error: B is not full rank or not independent in this order
                # This can happen if B columns are not independent? They are from RREF, so they are.
                # Maybe the order is wrong?
                pass
            
            if pivot_row != -1:
                # Swap to current row
                if pivot_row != col:
                    A[[col, pivot_row]] = A[[pivot_row, col]]
                
                # Eliminate other rows
                for row_idx in range(r):
                    if row_idx != col and A[row_idx, col] == 1:
                        A[row_idx] = (A[row_idx] + A[col]) % 2
        
        # Now the first r x r block should be Identity.
        # The next r x k block is C.
        C = A[:r, r:r+k]
        return C

    C = solve_gf2(B, M)
    
    # Now we have B (n_rows x r) and C (r x n_cols).
    # B columns are basis vectors.
    # C columns are coefficients for each output.
    
    # Synthesis:
    # 1. Compute basis wires b_0 ... b_{r-1}.
    #    b_i is XOR of inputs where B[:, i] == 1.
    #    Gate count for b_i: weight(B[:, i]) - 1.
    #    Wire index for b_i: n + 1 + i (if we number them sequentially).
    #    Wait, we need to interleave? No, we can compute all basis wires first.
    #    Inputs are 0..n-1. Zero wire is n.
    #    Basis wires start at n+1.
    
    gates = []
    wire_map = {} # index in basis -> wire id
    
    # Compute basis wires
    for i in range(r):
        cols = np.where(B[:, i] == 1)[0]
        if len(cols) == 0:
            wire_map[i] = n # Zero wire
        elif len(cols) == 1:
            wire_map[i] = cols[0] # Direct input
        else:
            # XOR the inputs
            w = cols[0]
            for j in range(1, len(cols)):
                new_wire = n + 1 + len(gates)
                gates.append([w, cols[j]])
                w = new_wire
            wire_map[i] = w
    
    # Compute outputs
    outputs = []
    for j in range(n_cols):
        coeffs = np.where(C[:, j] == 1)[0]
        if len(coeffs) == 0:
            outputs.append(n) # Zero wire
        elif len(coeffs) == 1:
            outputs.append(wire_map[coeffs[0]])
        else:
            # XOR the basis wires
            w = wire_map[coeffs[0]]
            for k in range(1, len(coeffs)):
                new_wire = n + 1 + len(gates)
                gates.append([w, wire_map[coeffs[k]]])
                w = new_wire
            outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END

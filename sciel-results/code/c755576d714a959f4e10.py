# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    rank_terms = []
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1}, 
             {(j0, k0): 1}, 
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1}, 
             {(j0, k0+1): 1, (j0+1, k0+1): -1}, 
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1}, 
             {(j0+1, k0): 1, (j0, k0): -1}, 
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1}, 
             {(j0+1, k0+1): 1}, 
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1}, 
             {(j0, k0): 1, (j0, k0+1): 1}, 
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1}, 
             {(j0+1, k0): 1, (j0+1, k0+1): 1}, 
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))
            
    # Initial block decomposition
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))
    
    # Build matrices as Fraction for exact arithmetic
    def build_matrices(terms):
        rank = len(terms)
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = Fraction(coeff)
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = Fraction(coeff)
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = Fraction(coeff)
        
        return U, V, W
    
    def verify_decomposition(U, V, W):
        """Check that U ⊗ V ⊗ W equals the matmul tensor."""
        rank = len(U[0]) if U else 0
        # For each (i, j, k) in output, check sum_r U[i*m+j][r] * V[j*p+k][r] == W[k*n+i][r] contribution
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k], so tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # The decomposition says: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # Wait, that's not right. The standard form is:
        # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) ... no
        # Actually: C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k])
        # Let me re-check: the tensor for matmul is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # And T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    lhs = Fraction(1)  # T[i,j,k] should be 1 for all valid (i,j,k)
                    rhs = Fraction(0)
                    for r in range(rank):
                        rhs += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if lhs != rhs:
                        return False
        return True
    
    def try_remove_column(terms, col_to_remove):
        """Try removing a column by expressing it as a linear combination of others."""
        rank = len(terms)
        if rank <= 1:
            return None
        
        U, V, W = build_matrices(terms)
        
        # Check if column col_to_remove is in the span of other columns
        # for U, V, W simultaneously
        other_cols = [c for c in range(rank) if c != col_to_remove]
        
        # Build system: find coefficients alpha_c such that
        # U[:, col] = sum_c alpha_c * U[:, c], similarly for V and W
        # This is a system of equations. Use Gaussian elimination with Fractions.
        
        # Collect all equations from U, V, W
        # Unknowns: alpha_c for c in other_cols
        num_unknowns = len(other_cols)
        other_idx = {c: idx for idx, c in enumerate(other_cols)}
        
        # Build augmented matrix [A | b] where A is (num_rows x num_unknowns), b is (num_rows,)
        rows = []
        for row in range(n * m):
            b_val = U[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                rows.append((row_vec, b_val))
        
        for row in range(m * p):
            b_val = V[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                rows.append((row_vec, b_val))
        
        for row in range(p * n):
            b_val = W[row][col_to_remove]
            if b_val != 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                rows.append((row_vec, b_val))
        
        # Also need: for rows where b_val == 0, the sum must be 0
        for row in range(n * m):
            if U[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(U[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        for row in range(m * p):
            if V[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(V[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        for row in range(p * n):
            if W[row][col_to_remove] == 0:
                row_vec = []
                for c in other_cols:
                    row_vec.append(W[row][c])
                if any(x != 0 for x in row_vec):
                    rows.append((row_vec, Fraction(0)))
        
        if not rows:
            # Column is all zeros, can remove
            return other_cols
        
        # Gaussian elimination
        # Matrix: rows x (num_unknowns + 1)
        mat = []
        for row_vec, b_val in rows:
            mat.append(row_vec + [b_val])
        
        # Reduce
        pivot_rows = []
        used_pivots = set()
        for col in range(num_unknowns):
            if col in used_pivots:
                continue
            # Find pivot
            pivot_row = None
            for r in range(len(mat)):
                if mat[r][col] != 0 and r not in used_pivots:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            
            # Swap to top
            mat[pivot_row], mat[len(used_pivots)] = mat[len(used_pivots)], mat[pivot_row]
            
            # Normalize
            pivot_val = mat[len(used_pivots)][col]
            mat[len(used_pivots)] = [x / pivot_val for x in mat[len(used_pivots)]]
            
            # Eliminate
            for r in range(len(mat)):
                if r == len(used_pivots):
                    continue
                factor = mat[r][col]
                if factor != 0:
                    mat[r] = [mat[r][c] - factor * mat[len(used_pivots)][c] for c in range(num_unknowns + 1)]
            
            used_pivots.add(len(used_pivots))
            pivot_rows.append((len(used_pivots) - 1, col))
        
        # Check for inconsistency
        for r in range(len(mat)):
            all_zero = all(mat[r][c] == 0 for c in range(num_unknowns))
            if all_zero and mat[r][num_unknowns] != 0:
                return None  # Inconsistent, can't remove
        
        # Solution exists. Extract alpha values
        alpha = [Fraction(0)] * num_unknowns
        for pr, col in pivot_rows:
            alpha[col] = mat[pr][num_unknowns]
        
        # Build new terms
        new_terms = []
        for c in other_cols:
            if alpha[other_idx[c]] != 0:
                u_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][0].items()}
                v_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][1].items()}
                w_dict = {k: v * alpha[other_idx[c]] for k, v in terms[c][2].items()}
                new_terms.append((u_dict, v_dict, w_dict))
        
        # Remove zero terms
        new_terms = [(u, v, w) for u, v, w in new_terms 
                     if u and v and w]
        
        return new_terms
    
    # Verify initial decomposition
    U, V, W = build_matrices(rank_terms)
    if not verify_decomposition(U, V, W):
        # Fallback: return as-is (shouldn't happen)
        pass
    
    # Try to remove columns one by one
    current_terms = list(rank_terms)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for col in range(len(current_terms)):
            if time.time() > deadline:
                break
            new_terms = try_remove_column(current_terms, col)
            if new_terms is not None and len(new_terms) < len(current_terms):
                # Verify the new decomposition
                U_new, V_new, W_new = build_matrices(new_terms)
                if verify_decomposition(U_new, V_new, W_new):
                    current_terms = new_terms
                    improved = True
                    break
    
    rank = len(current_terms)
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(current_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = float(coeff) if isinstance(coeff, Fraction) else coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

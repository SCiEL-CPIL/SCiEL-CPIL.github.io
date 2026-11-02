"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_dict.items():
            u_vec[u_idx(i, j)] += val
        for (j, k), val in v_dict.items():
            v_vec[v_idx(j, k)] += val
        for (k, i), val in w_dict.items():
            w_vec[w_idx(k, i)] += val
            
        terms.append((u_vec, v_vec, w_vec))

    def strassen_block(y_indices, z_indices, w_indices):
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices
        
        # M1 = (y00+y11)(z00+z11)
        u = {y00: 1, y11: 1}
        v = {z00: 1, z11: 1}
        w = {w00: 1, w11: 1}
        add_term(u, v, w)
        
        # M2 = (y10+y11)z00
        u = {y10: 1, y11: 1}
        v = {z00: 1}
        w = {w10: 1, w11: -1}
        add_term(u, v, w)
        
        # M3 = y00(z01-z11)
        u = {y00: 1}
        v = {z01: 1, z11: -1}
        w = {w01: 1, w11: 1}
        add_term(u, v, w)
        
        # M4 = y11(z10-z00)
        u = {y11: 1}
        v = {z10: 1, z00: -1}
        w = {w00: 1, w10: 1}
        add_term(u, v, w)
        
        # M5 = (y00+y01)z11
        u = {y00: 1, y01: 1}
        v = {z11: 1}
        w = {w00: -1, w01: 1}
        add_term(u, v, w)
        
        # M6 = (y10-y00)(z00+z01)
        u = {y10: 1, y00: -1}
        v = {z00: 1, z01: 1}
        w = {w11: 1}
        add_term(u, v, w)
        
        # M7 = (y01-y11)(z10+z11)
        u = {y01: 1, y11: -1}
        v = {z10: 1, z11: 1}
        w = {w00: 1}
        add_term(u, v, w)

    def schoolbook_2x2x1(y_indices, z_indices, w_indices):
        y00, y01, y10, y11 = y_indices
        z0, z1 = z_indices
        w0, w1 = w_indices
        
        # M1 = y00 * z0
        add_term({y00: 1}, {z0: 1}, {w0: 1})
        
        # M2 = y01 * z1
        add_term({y01: 1}, {z1: 1}, {w0: 1})
        
        # M3 = y10 * z0
        add_term({y10: 1}, {z0: 1}, {w1: 1})
        
        # M4 = y11 * z1
        add_term({y11: 1}, {z1: 1}, {w1: 1})

    # Block 1: A_L (cols 0,1) * B_1 (rows 0,1, cols 0,1) -> C (cols 0,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_2 (rows 2,3, cols 0,1) -> C (cols 0,1)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_1 (rows 0,1, cols 2,3) -> C (cols 2,3)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_2 (rows 2,3, cols 2,3) -> C (cols 2,3)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_1 (rows 0,1, col 4) -> C (col 4)
    y5 = [(0,0), (0,1), (1,0), (1,1)]
    z5 = [(0,4), (1,4)]
    w5 = [(4,0), (4,1)]
    schoolbook_2x2x1(y5, z5, w5)
    
    # Block 6: A_R (cols 2,3) * B_2 (rows 2,3, col 4) -> C (col 4)
    y6 = [(0,2), (0,3), (1,2), (1,3)]
    z6 = [(2,4), (3,4)]
    w6 = [(4,0), (4,1)]
    schoolbook_2x2x1(y6, z6, w6)
    
    # Now we have 36 terms in `terms`
    # Check for linear dependencies
    # Flatten each term to a vector of length n*m*p = 40? 
    # Wait, the tensor space is 8x20x10 = 1600.
    # But the condition is only 40 equations: sum_t U_t V_t W_t = 1 for all i,j,k.
    # Actually, the tensor T_mat is fixed.
    # We can check if the sum of a subset of terms equals T_mat.
    # T_mat[i,j,k] = 1 for all i,j,k? No.
    # T_mat[i,j,k] = 1 if we are computing C_ik?
    # The condition is sum_t U_t[i,j] V_t[j,k] W_t[k,i] = 1 for all i,j,k.
    # This is 2*4*5 = 40 equations.
    # Let's form the matrix A (40 x 36) where A[idx, t] = U_t[i,j] V_t[j,k] W_t[k,i]
    # We want to find a subset of columns that sums to the vector of ones.
    # Actually, we know the sum of ALL 36 columns is the vector of ones.
    # We want to find a linear combination c_t such that sum c_t A[:, t] = 1, and c_t is 0 or 1.
    # Since sum 1 * A[:, t] = 1, any dependency allows us to change coefficients.
    # If A is rank R, there are 36-R dependencies.
    
    # Let's build the 40x36 matrix
    A_mat = np.zeros((n*m*p, len(terms)))
    for t, (u, v, w) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    A_mat[idx, t] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                    
    # Check rank
    rank = np.linalg.matrix_rank(A_mat)
    
    # If rank < 36, we can reduce.
    # Find a basis using Gaussian elimination or SVD
    # We want to keep as many original terms as possible?
    # Actually, we just need ANY set of R terms that sums to 1.
    # If we find a basis of R terms, we can express the others in terms of the basis.
    # But we need the coefficients to be such that the sum is 1.
    # Since the sum of all 36 is 1, if we drop term k, we need to add its "shadow" to the others.
    
    # Simple strategy: Try dropping each term k.
    # Check if 1 is in the span of the remaining 35 columns.
    # Since sum of all is 1, 1 is in the span of all 36.
    # If term k is redundant, 1 is in the span of the other 35.
    
    best_terms = terms
    best_rank = len(terms)
    
    for k in range(len(terms)):
        # Create matrix without column k
        mask = [i for i in range(len(terms)) if i != k]
        A_sub = A_mat[:, mask]
        
        # Check if vector of ones is in the column space of A_sub
        # Solve A_sub x = 1
        # If a solution exists, term k is redundant.
        try:
            # Use lstsq for stability
            sol, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, np.ones(n*m*p), rcond=None)
            # Check if the solution is "good" (residual is small)
            # Since the entries are integers, the solution should be exact if it exists.
            # But due to floating point, we check residual.
            if residuals and residuals[0] < 1e-9:
                # Term k is redundant.
                # We can drop it.
                # New set of terms is terms without k.
                # We can try to drop more?
                # For now, let's just drop one and see.
                # To be safe, we can try to find the maximum subset.
                # But dropping one is a good improvement.
                new_terms = [terms[i] for i in mask]
                if len(new_terms) < best_rank:
                    best_terms = new_terms
                    best_rank = len(new_terms)
        except:
            pass
            
    # If we found a smaller set, use it.
    final_terms = best_terms
    
    # Construct U, V, W
    rank = len(final_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for t, (u, v, w) in enumerate(final_terms):
        U[:, t] = u
        V[:, t] = v
        W[:, t] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

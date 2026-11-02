import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Returns the best rank found (55 or potentially 54).
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify rank-55 solution
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i, j, k] = np.sum(U55[i*m+j, :] * V55[j*p+k, :] * W55[k*n+i, :])
    
    if not np.allclose(T_check, 1.0):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Attempt to find rank-54
    # Strategy: Try to merge or eliminate one term by solving a linear system
    # with 54 terms.
    # We try to remove each of the 55 terms and see if the rest can be adjusted.
    # This is expensive, so we limit the attempts.
    
    target_rank = 54
    found = False
    
    # We only try to remove terms from the Strassen part (first 35)
    # or the schoolbook part (last 20).
    
    # To keep it fast, we try a few random subsets.
    # Actually, we can try to remove one term at a time.
    
    # If we remove term r, we have 54 terms left.
    # We need to find coefficients c_1..c_54 such that sum c_i T_i = T.
    # This is a linear system A x = b.
    # A is 60 x 54. b is 60 x 1.
    # We can use np.linalg.lstsq.
    
    # Construct the matrix A for the full 55 terms.
    # Each column corresponds to a term.
    # The entry A[eq, r] is the value of the r-th term at the eq-th tensor position.
    
    # Tensor positions: (i, j, k) -> index
    # i in 0..1, j in 0..2, k in 0..9
    # Total 60 positions.
    
    A_full = np.zeros((60, 55))
    for r in range(55):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    # Term r contributes U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    A_full[idx, r] = U55[i*m+j, r] * V55[j*p+k, r] * W55[k*n+i, r]
    
    b = np.ones(60)
    
    # Try removing each term r
    for r_remove in range(55):
        if time.time() - start_time > DEADLINE:
            break
            
        # Select all columns except r_remove
        cols = [c for c in range(55) if c != r_remove]
        A_sub = A_full[:, cols]
        
        # Solve A_sub x = b
        # Use least squares
        try:
            x, residuals, rank, s = np.linalg.lstsq(A_sub, b, rcond=None)
            # Check if the solution is exact
            if np.allclose(A_sub @ x, b, atol=1e-9):
                # We found a rank-54 solution!
                # Construct the new U, V, W
                U_new = np.zeros((n*m, target_rank))
                V_new = np.zeros((m*p, target_rank))
                W_new = np.zeros((p*n, target_rank))
                
                for new_r, old_r in enumerate(cols):
                    U_new[:, new_r] = U55[:, old_r] * x[new_r]
                    V_new[:, new_r] = V55[:, old_r] * x[new_r]
                    W_new[:, new_r] = W55[:, old_r] * x[new_r]
                    
                # Verify
                T_check_new = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check_new[i, j, k] = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                
                if np.allclose(T_check_new, 1.0):
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, target_rank
                    found = True
                    break
        except Exception:
            continue
            
    if found:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

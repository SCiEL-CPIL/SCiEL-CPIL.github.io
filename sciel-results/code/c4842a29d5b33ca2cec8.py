import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = delta for all i,j,k
    # Actually, the condition is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_s (sum_j U[i*m+j,s]*V[j*p+k,s]) * W[k*n+i,s]
    # This must equal sum_j A[i,j]*B[j,k] for all A,B
    # So we need: for all i,k,j,j',k': the tensor matches
    
    # Better: check that for random A, B, the reconstructed C matches
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    # Reconstruct: C[i,k] = sum_s W[k*n+i, s] * (sum_j U[i*m+j, s] * V[j*p+k, s])
    C_rec = np.zeros((n, p))
    for s in range(rank):
        # u_s: (n*m,), v_s: (m*p,)
        u_s = U[:, s].reshape(n, m)
        v_s = V[:, s].reshape(m, p)
        # product_s = u_s @ v_s: (n, p)
        prod = u_s @ v_s
        C_rec += W[:, s] * prod  # W[:, s] is (p*n,), need to reshape
    
    # W[k*n+i, s] -> reshape to (p, n) then transpose to (n, p)
    # Actually W is (p*n, rank), so W[:, s] is (p*n,)
    # W[k*n + i, s] = W_reshaped[k, i, s] where W_reshaped is (p, n, rank)
    # C_rec[i, k] = sum_s W_reshaped[k, i, s] * prod_s[i, k]
    
    C_rec = np.zeros((n, p))
    W_reshaped = W.reshape(p, n, rank)
    for s in range(rank):
        u_s = U[:, s].reshape(n, m)
        v_s = V[:, s].reshape(m, p)
        prod = u_s @ v_s
        C_rec += W_reshaped[:, :, s] * prod  # W_reshaped[:,:,s] is (p,n), prod is (n,p) -> need transpose
    
    # W_reshaped[k, i, s] * prod[i, k] -> sum over i for fixed k? No.
    # C_rec[i, k] = sum_s W_reshaped[k, i, s] * prod_s[i, k]
    # So: C_rec = sum_s (W_reshaped[:, :, s].T * prod_s)
    
    C_rec = np.zeros((n, p))
    for s in range(rank):
        u_s = U[:, s].reshape(n, m)
        v_s = V[:, s].reshape(m, p)
        prod = u_s @ v_s  # (n, p)
        w_s = W_reshaped[:, :, s].T  # (n, p)
        C_rec += w_s * prod
    
    if not np.allclose(C_rec, C_true, atol=tol):
        return False
    return True

def _strassen_2x2_terms():
    """Return Strassen coefficients for 2x2 * 2x2."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h), P2 = (c+d)(e+f), P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # R00 = P1+P4-P5+P7, R01 = P3+P5, R10 = P2+P4, R11 = P1-P2+P3+P6
    
    U_coeffs = [
        {(0,0): 1, (1,1): 1},   # P1: a+d
        {(1,0): 1, (1,1): 1},   # P2: c+d
        {(0,0): 1},              # P3: a
        {(1,1): 1},              # P4: d
        {(0,0): 1, (0,1): 1},   # P5: a+b
        {(1,0): 1, (0,0): -1},  # P6: c-a
        {(0,1): 1, (1,1): -1},  # P7: b-d
    ]
    V_coeffs = [
        {(0,0): 1, (1,1): 1},   # P1: e+h
        {(0,0): 1, (0,1): 1},   # P2: e+f
        {(0,1): 1, (1,1): -1},  # P3: f-h
        {(1,0): 1, (0,0): -1},  # P4: g-e
        {(1,1): 1},              # P5: h
        {(0,0): 1, (0,1): 1},   # P6: e+f
        {(1,0): 1, (1,1): 1},   # P7: g+h
    ]
    W_coeffs = [
        {(0,0): 1, (1,1): 1},   # P1: R00+, R11+
        {(1,0): 1, (1,1): -1},  # P2: R10+, R11-
        {(0,1): 1, (1,1): 1},   # P3: R01+, R11+
        {(0,0): 1, (1,0): 1},   # P4: R00+, R10+
        {(0,0): -1, (0,1): 1},  # P5: R00-, R01+
        {(1,1): 1},              # P6: R11+
        {(0,0): 1},              # P7: R00+
    ]
    return U_coeffs, V_coeffs, W_coeffs

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_coeffs, V_coeffs, W_coeffs = _strassen_2x2_terms()
    
    col = 0
    for a_block in range(2):      # A: left (cols 0-1) or right (cols 2-3)
        for b_block in range(3):  # B: column block 0 (cols 0-1), 1 (cols 2-3), 2 (cols 4-5)
            for s in range(7):    # Strassen term
                # U: A block a_block, Strassen term s
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    U[i * m + j, col] += coeff
                
                # V: B row block a_block (rows a_block*2 to a_block*2+1), column block b_block
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    V[j * p + k, col] += coeff
                
                # W: output row i_loc, column k in block b_block
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    W[k * n + i, col] += coeff
                
                col += 1
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

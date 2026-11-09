# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Returns a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    A_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    B_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    C_coeffs = [
        [1, -1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for m_outer in range(7):
        for m_inner in range(7):
            col = m_outer * 7 + m_inner
            
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    c = j // 2
                    u_i = i % 2
                    u_j = j % 2
                    b = r * 2 + c
                    d = u_i * 2 + u_j
                    U[4*i+j][col] = A_coeffs[m_outer][b] * A_coeffs[m_inner][d]
            
            for j in range(4):
                for k in range(4):
                    c = j // 2
                    d = k // 2
                    v_j = j % 2
                    v_k = k % 2
                    b_B = c * 2 + d
                    d_B = v_j * 2 + v_k
                    V[4*j+k][col] = B_coeffs[m_outer][b_B] * B_coeffs[m_inner][d_B]
            
            for k in range(4):
                for i in range(4):
                    r = i // 2
                    d = k // 2
                    w_i = i % 2
                    w_k = k % 2
                    b_C = r * 2 + d
                    d_C = w_i * 2 + w_k
                    W[4*k+i][col] = C_coeffs[m_outer][b_C] * C_coeffs[m_inner][d_C]
    
    return {"U": U, "V": V, "W": W}

def _extend_to_4x4x5(base):
    """
    Extend a 4x4x4 decomposition to 4x4x5.
    Rank = Rank(4x4x4) + 16.
    """
    U_base = base["U"]
    V_base = base["V"]
    W_base = base["W"]
    rank_base = len(U_base[0])
    
    # New rank
    rank_new = rank_base + 16
    
    U = [row[:] + [0]*16 for row in U_base]
    V = [row[:] + [0]*16 for row in V_base]
    W = [row[:] + [0]*16 for row in W_base]
    
    # Add terms for k=4
    # C[i,4] = sum_j A[i,j] B[j,4]
    # We need 16 terms for this?
    # Term j: A[i,j] B[j,4] contributes to C[i,4].
    # For each j (0..3) and each i (0..3)?
    # No, the decomposition must be separable.
    # The part for k=4 is a matrix-vector product.
    # Rank of 4x4x1 is 16.
    # U[i,j,c] for c in 16 terms.
    # V[j,4,c] for c in 16 terms.
    # W[4,i,c] for c in 16 terms.
    
    for j in range(4):
        for i in range(4):
            c = rank_base + j * 4 + i
            U[4*i+j][c] = 1
            V[4*j+4][c] = 1
            W[4*4+i][c] = 1 if i == j else 0 # W[k*n+i] with k=4. Index 16+i.
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try Strassen Extension
    try:
        base_444 = _strassen_4x4x4()
        extended = _extend_to_4x4x5(base_444)
        
        if _verify_decomp(n, m, p, extended["U"], extended["V"], extended["W"]):
            best_result = extended
            best_rank = len(extended["U"][0])
    except Exception:
        pass
        
    return best_result
# EVOLVE-BLOCK-END

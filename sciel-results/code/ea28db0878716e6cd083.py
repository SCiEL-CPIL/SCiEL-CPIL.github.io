import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly reconstructs the all-ones tensor."""
    rank = U.shape[1]
    U_t = U.T.reshape(n, m, rank)
    V_t = V.T.reshape(m, p, rank)
    W_t = W.T.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    target = np.ones((n, m, p))
    return np.allclose(result, target, atol=1e-9)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _strassen_decomposition():
    """
    Returns the rank-7 tensor decomposition for 2x2x2 matrix multiplication.
    Matrices indexed as:
    A: [[a00, a01], [a10, a11]]
    B: [[b00, b01], [b10, b11]]
    C: [[c00, c01], [c10, c11]]
    
    Returns list of (u, v, w) where u,v,w are length-4 vectors.
    u corresponds to A coeffs (a00,a01,a10,a11), v to B coeffs, w to C coeffs.
    """
    decomps = []
    
    # M1 = (a00+a11)(b00+b11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    decomps.append((u, v, w))
    
    # M2 = (a10+a11)b00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, 0]
    decomps.append((u, v, w))
    
    # M3 = a00(b11-b00)
    u = [1, 0, 0, 0]
    v = [-1, 0, 0, 1]
    w = [1, 0, 0, 1]
    decomps.append((u, v, w))
    
    # M4 = a01(b00+b10)
    u = [0, 1, 0, 0]
    v = [1, 0, 1, 0]
    w = [0, 1, 1, 0]
    decomps.append((u, v, w))
    
    # M5 = (a00-a01)(b10+b11)
    u = [1, -1, 0, 0]
    v = [0, 0, 1, 1]
    w = [-1, 0, 0, 1]
    decomps.append((u, v, w))
    
    # M6 = (a10-a00)b01
    u = [-1, 0, 1, 0]
    v = [0, 1, 0, 0]
    w = [0, 1, 0, 1]
    decomps.append((u, v, w))
    
    # M7 = (a11-a10)(b00-b01)
    u = [0, 0, -1, 1]
    v = [1, -1, 0, 0]
    w = [1, 0, 0, -1]
    decomps.append((u, v, w))
    
    return decomps

def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n×m×p matrix multiplication.
    For 2×4×8: split A into 2×2 blocks, B into 2×2 blocks, and use rank-7 Strassen.
    """
    strassen_decomp = _strassen_decomposition()
    nm, mp, pn = n*m, m*p, p*n
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_2x2_product(a_row, a_col, b_row, b_col, c_row, c_col):
        """Add a 2×2×2 Strassen product at the given block offsets."""
        for (u, v, w) in strassen_decomp:
            u_full = np.zeros(nm)
            v_full = np.zeros(mp)
            w_full = np.zeros(pn)
            
            for k in range(4):
                r = a_row + (k // 2)
                c = a_col + (k % 2)
                u_full[r * m + c] += u[k]
                
            for k in range(4):
                r = b_row + (k // 2)
                c = b_col + (k % 2)
                v_full[r * p + c] += v[k]
                
            for k in range(4):
                r = c_row + (k // 2)
                c = c_col + (k % 2)
                w_full[c * n + r] += w[k]
                
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)
    
    # For 2×4×8:
    # A is 2×4, split into A_L (cols 0-1) and A_R (cols 2-3)
    # B is 4×8, split into 2×2 blocks
    # C is 2×8, split into 2×2 blocks
    
    # B_L (cols 0-3): P=B[0:2,0:2], Q=B[0:2,2:4], R=B[2:4,0:2], S=B[2:4,2:4]
    # C_L0 (cols 0-1) = A_L @ P + A_R @ R
    add_2x2_product(0, 0, 0, 0, 0, 0)   # A_L @ P
    add_2x2_product(0, 2, 2, 0, 0, 0)   # A_R @ R
    # C_L1 (cols 2-3) = A_L @ Q + A_R @ S
    add_2x2_product(0, 0, 0, 2, 0, 2)   # A_L @ Q
    add_2x2_product(0, 2, 2, 2, 0, 2)   # A_R @ S
    
    # B_R (cols 4-7): P'=B[0:2,4:6], Q'=B[0:2,6:8], R'=B[2:4,4:6], S'=B[2:4,6:8]
    # C_R0 (cols 4-5) = A_L @ P' + A_R @ R'
    add_2x2_product(0, 0, 0, 4, 0, 4)   # A_L @ P'
    add_2x2_product(0, 2, 2, 4, 0, 4)   # A_R @ R'
    # C_R1 (cols 6-7) = A_L @ Q' + A_R @ S'
    add_2x2_product(0, 0, 0, 6, 0, 6)   # A_L @ Q'
    add_2x2_product(0, 2, 2, 6, 0, 6)   # A_R @ S'
    
    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook fallback
    U_school, V_school, W_school, rank_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy 1: Strassen block decomposition (rank 56 for 2×4×8)
    if time.time() < deadline - 10:
        U_s, V_s, W_s, rank_s = _build_strassen_decomposition(n, m, p)
        if _verify_decomposition(U_s, V_s, W_s, n, m, p):
            if rank_s < best_rank:
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank_s
    
    # Strategy 2: Try to find a lower-rank decomposition via numerical optimization
    # The tensor is all-ones of shape (2,4,8). We want to find rank-r decomposition.
    # This is a hard problem, but for small r we can try.
    # Lower bound: rank >= max(n,m,p) = 8, but likely higher.
    # Let's try ranks from 32 up to best_rank-1 to see if we can do better than 56.
    
    # For now, return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

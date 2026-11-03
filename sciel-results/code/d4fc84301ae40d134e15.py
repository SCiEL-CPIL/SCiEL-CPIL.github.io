import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140

def _build_target_tensor(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) > tol:
        return False
    U_h = U * 2
    V_h = V * 2
    W_h = W * 2
    if not np.allclose(U_h, np.round(U_h), atol=1e-5):
        return False
    if not np.allclose(V_h, np.round(V_h), atol=1e-5):
        return False
    if not np.allclose(W_h, np.round(W_h), atol=1e-5):
        return False
    return True

def _schoolbook(n, m, p):
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
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_block(j0, j1, k0, k1, n=2, m=3, p=4):
    """
    Generate 7 rank-1 terms for the Strassen decomposition of the 2x2x2 block
    defined by A cols j0, j1 and B rows j0, j1 and B cols k0, k1 and C rows k0, k1.
    Returns list of (U_vec, V_vec, W_vec) where each is a dict or sparse representation.
    Here we return full vectors of size n*m, m*p, p*n.
    """
    terms = []
    
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # A indices: i in {0,1}, j in {j0, j1}
    # B indices: j in {j0, j1}, k in {k0, k1}
    # C indices: k in {k0, k1}, i in {0,1}
    
    # Helper to create U, V, W vectors for a term
    def make_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {(i, j): val}
        v_coeffs: dict {(j, k): val}
        w_coeffs: dict {(k, i): val}
        """
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        
        for (i, j), val in u_coeffs.items():
            U[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            W[k * n + i] = val
            
        return U, V, W

    # M1
    u = {(0, j0): 1, (1, j1): 1}
    v = {(j0, k0): 1, (j1, k1): 1}
    w = {(k0, 0): 1, (k0, 1): 1, (k1, 0): 1, (k1, 1): 1} 
    # Wait, C00 = M1 + ..., C01 = ..., C10 = ..., C11 = ...
    # M1 contributes to C00, C11?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-map carefully.
    # C indices: (k, i)
    # k0, k1 are rows of C. i0, i1 are cols of C.
    
    # M1 contributes to C(k0, i0) and C(k1, i1)?
    # No, M1 is a scalar. It is added to specific C entries.
    # C(k0, i0) += M1
    # C(k1, i1) += M1
    
    # So w_coeffs for M1: {(k0, 0): 1, (k1, 1): 1}
    
    terms.append(make_term(
        {(0, j0): 1, (1, j1): 1},
        {(j0, k0): 1, (j1, k1): 1},
        {(k0, 0): 1, (k1, 1): 1}
    ))
    
    # M2: (A10+A11)B00
    # u: (1, j0): 1, (1, j1): 1
    # v: (j0, k0): 1
    # w: C10 += M2, C11 -= M2
    # C10 is (k0, 1), C11 is (k1, 1)
    terms.append(make_term(
        {(1, j0): 1, (1, j1): 1},
        {(j0, k0): 1},
        {(k0, 1): 1, (k1, 1): -1}
    ))
    
    # M3: A00(B10-B11)
    # u: (0, j0): 1
    # v: (j1, k0): 1, (j1, k1): -1
    # w: C01 += M3, C11 += M3
    # C01 is (k1, 0), C11 is (k1, 1)
    terms.append(make_term(
        {(0, j0): 1},
        {(j1, k0): 1, (j1, k1): -1},
        {(k1, 0): 1, (k1, 1): 1}
    ))
    
    # M4: A11(B01-B00)
    # u: (1, j1): 1
    # v: (j0, k1): 1, (j0, k0): -1
    # w: C00 += M4, C10 += M4
    # C00 is (k0, 0), C10 is (k0, 1)
    terms.append(make_term(
        {(1, j1): 1},
        {(j0, k1): 1, (j0, k0): -1},
        {(k0, 0): 1, (k0, 1): 1}
    ))
    
    # M5: (A00+A01)B11
    # u: (0, j0): 1, (0, j1): 1
    # v: (j1, k1): 1
    # w: C00 -= M5, C01 += M5
    # C00 is (k0, 0), C01 is (k1, 0)
    terms.append(make_term(
        {(0, j0): 1, (0, j1): 1},
        {(j1, k1): 1},
        {(k0, 0): -1, (k1, 0): 1}
    ))
    
    # M6: (A10-A00)(B00+B01)
    # u: (1, j0): 1, (0, j0): -1
    # v: (j0, k0): 1, (j0, k1): 1
    # w: C11 += M6
    # C11 is (k1, 1)
    terms.append(make_term(
        {(1, j0): 1, (0, j0): -1},
        {(j0, k0): 1, (j0, k1): 1},
        {(k1, 1): 1}
    ))
    
    # M7: (A01-A11)(B10+B11)
    # u: (0, j1): 1, (1, j1): -1
    # v: (j1, k0): 1, (j1, k1): 1
    # w: C00 += M7
    # C00 is (k0, 0)
    terms.append(make_term(
        {(0, j1): 1, (1, j1): -1},
        {(j1, k0): 1, (j1, k1): 1},
        {(k0, 0): 1}
    ))
    
    return terms

def construct():
    n, m, p = 2, 3, 4
    T = _build_target_tensor(n, m, p)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: j in {0,1}, k in {0,1}
    terms1 = _strassen_block(0, 1, 0, 1, n, m, p)
    for U, V, W in terms1:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    # Block 2: j in {0,1}, k in {2,3}
    terms2 = _strassen_block(0, 1, 2, 3, n, m, p)
    for U, V, W in terms2:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    # Block 3: j=2, k in {0,1,2,3} (Schoolbook for 2x1x4)
    for i in range(n):
        for k in range(p):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            U[i * m + 2] = 1
            V[2 * p + k] = 1
            W[k * n + i] = 1
            U_list.append(U)
            V_list.append(V)
            W_list.append(W)
            
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    if _verify(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

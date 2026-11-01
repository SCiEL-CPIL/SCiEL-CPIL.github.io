"""
Strassen rank-7 decomposition for 2x2 matrix multiplication.
The known optimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
We construct this decomposition, verify it numerically, and return it.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return Strassen's rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    The bilinear decomposition form:
    C[i,k] = sum_c W[k*n+i, c] * m_c
    where m_c = (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
    
    With n=m=p=2:
    - U has shape (4, 7): row index = i*2+j
    - V has shape (4, 7): row index = j*2+k
    - W has shape (4, 7): row index = k*2+i (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M3 - M2 + M6
    
    factors = [
        # (u_coords, u_vals, v_coords, v_vals, w_coords, w_vals)
        # u_coords: list of (i, j), v_coords: list of (j, k), w_coords: list of (k, i)
        
        # M1 = (A00+A11)(B00+B11)
        ([ (0,0), (1,1) ], [1, 1],
         [ (0,0), (1,1) ], [1, 1],
         [ (0,0), (1,1) ], [1, 1]),
        
        # M2 = (A10+A11)B00
        ([ (1,0), (1,1) ], [1, 1],
         [ (0,0) ], [1],
         [ (0,1), (1,1) ], [1, -1]),
        
        # M3 = A00(B01-B11)
        ([ (0,0) ], [1],
         [ (0,1), (1,1) ], [1, -1],
         [ (1,0), (1,1) ], [1, 1]),
        
        # M4 = A11(B10-B00)
        ([ (1,1) ], [1],
         [ (1,0), (0,0) ], [1, -1],
         [ (0,0), (0,1) ], [1, 1]),
        
        # M5 = (A00+A01)B11
        ([ (0,0), (0,1) ], [1, 1],
         [ (1,1) ], [1],
         [ (0,0), (1,0) ], [-1, 1]),
        
        # M6 = (A10-A00)(B00+B01)
        ([ (1,0), (0,0) ], [1, -1],
         [ (0,0), (0,1) ], [1, 1],
         [ (1,1) ], [1]),
        
        # M7 = (A01-A11)(B10+B11)
        ([ (0,1), (1,1) ], [1, -1],
         [ (1,0), (1,1) ], [1, 1],
         [ (0,0) ], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
    
    # Verify numerically with multiple random tests
    rng = np.random.RandomState(42)
    for trial in range(50):
        A = rng.randint(-5, 6, (2, 2)).astype(np.float64)
        B = rng.randint(-5, 6, (2, 2)).astype(np.float64)
        C_expected = A @ B
        
        A_flat = A.flatten()  # [A00, A01, A10, A11]
        B_flat = B.flatten()  # [B00, B01, B10, B11]
        
        # Compute m_c for each c
        m = np.zeros(rank)
        for c in range(rank):
            a_c = np.dot(U[:, c], A_flat)
            b_c = np.dot(V[:, c], B_flat)
            m[c] = a_c * b_c
        
        # Compute C from decomposition
        # C_flat is indexed by k*n+i, so C_flat[k*2+i] = C[i,k]
        C_flat = np.zeros(4)
        for ik in range(4):
            C_flat[ik] = np.dot(W[ik, :], m)
        
        C_recon = np.zeros((2, 2))
        for k in range(2):
            for i in range(2):
                C_recon[i, k] = C_flat[k * 2 + i]
        
        if not np.allclose(C_recon, C_expected, atol=1e-10):
            raise ValueError(f"Decomposition failed at trial {trial}! A={A}, B={B}, expected={C_expected}, got={C_recon}")
    
    # Check deadline
    if time.time() > deadline:
        # Return immediately if close to deadline
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

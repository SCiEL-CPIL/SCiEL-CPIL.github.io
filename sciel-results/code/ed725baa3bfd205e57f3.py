import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()

def _time_left():
    return _DEADLINE_S - (time.time() - _START)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify_matmul(U, V, W):
    """
    Verifies if U, V, W decompose 3x3 matmul.
    Convention:
    C[i, k] = sum_r (sum_j A[i, j] U[i*m+j, r]) (sum_j B[j, k] V[j*p+k, r]) W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    R = len(U[0])
    
    # Test with small integer matrices
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)
    B = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int64)
    C_expected = A @ B
    
    C_computed = np.zeros((3, 3), dtype=np.int64)
    
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    A_flat = A.flatten() # i*m+j
    B_flat = B.flatten() # j*p+k
    
    U_proj = U_arr.T @ A_flat
    V_proj = V_arr.T @ B_flat
    
    for r in range(R):
        p_r = U_proj[r] * V_proj[r]
        if p_r != 0:
            w_col = W_arr[:, r]
            w_matrix = w_col.reshape(3, 3) # row k, col i
            C_computed += p_r * w_matrix.T
            
    if not np.allclose(C_computed, C_expected):
        return False
    return True

def _rank26_construct():
    # Rank 26 decomposition with correct W indices (W[k*n+i])
    # W index mapping: C[i,k] -> W[k*3+i]
    # C[0,0]->0, C[0,1]->3, C[0,2]->6
    # C[1,0]->1, C[1,1]->4, C[1,2]->7
    # C[2,0]->2, C[2,1]->5, C[2,2]->8
    
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
        
    # --- Block 1: Strassen for top-left 2x2 ---
    # A = [[a, b], [c, d]] = [[A0, A1], [A3, A4]]
    # B = [[e, f], [g, h]] = [[B0, B1], [B3, B4]]
    # C00 = M1+M4-M5-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M6+M7
    # W: C00->0, C01->3, C10->1, C11->4
    
    # M1 = (a+d)(e+h): contributes to C00(+1) and C11(+1)
    add({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})
    
    # M2 = (c+d)e: contributes to C10(+1)
    add({3:1, 4:1}, {0:1}, {1:1})
    
    # M3 = a(f-g): contributes to C01(+1)
    add({0:1}, {1:1, 3:-1}, {3:1})
    
    # M4 = d(h-f): contributes to C00(+1) and C11(+1)
    add({4:1}, {4:1, 1:-1}, {0:1, 4:1})
    
    # M5 = (a+b)h: contributes to C00(-1) and C01(+1)
    add({0:1, 1:1}, {4:1}, {0:-1, 3:1})
    
    # M6 = (c-a)(e+f): contributes to C11(+1)
    add({3:1, 0:-1}, {0:1, 1:1}, {4:1})
    
    # M7 = (b-d)(g+h): contributes to C00(-1) and C11(+1)
    add({1:1, 4:-1}, {3:1, 4:1}, {0:-1, 4:1})
    
    # --- Block 2: Scalar for bottom-right ---
    # A8 * B8 -> C8 (W=8)
    add({8:1}, {8:1}, {8:1})
    
    # --- Block 3: Cross terms ---
    # 1. A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    #    A2*B6 -> C[0,0] (W=0)
    #    A2*B7 -> C[0,1] (W=3)
    #    A5*B6 -> C[1,0] (W=1)
    #    A5*B7 -> C[1,1] (W=4)
    add({2:1}, {6:1}, {0:1})
    add({2:1}, {7:1}, {3:1})
    add({5:1}, {6:1}, {1:1})
    add({5:1}, {7:1}, {4:1})
    
    # 2. A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    #    A0*B2 -> C[0,2] (W=6)
    #    A1*B5 -> C[0,2] (W=6)
    #    A3*B2 -> C[1,2] (W=7)
    #    A4*B5 -> C[1,2] (W=7)
    add({0:1}, {2:1}, {6:1})
    add({1:1}, {5:1}, {6:1})
    add({3:1}, {2:1}, {7:1})
    add({4:1}, {5:1}, {7:1})
    
    # 3. A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    #    A2*B8 -> C[0,2] (W=6)
    #    A5*B8 -> C[1,2] (W=7)
    add({2:1}, {8:1}, {6:1})
    add({5:1}, {8:1}, {7:1})
    
    # 4. A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    #    A6*B0 -> C[2,0] (W=2)
    #    A7*B3 -> C[2,0] (W=2)
    #    A6*B1 -> C[2,1] (W=5)
    #    A7*B4 -> C[2,1] (W=5)
    add({6:1}, {0:1}, {2:1})
    add({7:1}, {3:1}, {2:1})
    add({6:1}, {1:1}, {5:1})
    add({7:1}, {4:1}, {5:1})
    
    # 5. A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    #    A8*B6 -> C[2,0] (W=2)
    #    A8*B7 -> C[2,1] (W=5)
    add({8:1}, {6:1}, {2:1})
    add({8:1}, {7:1}, {5:1})
    
    # 6. A[2, 0:2] * B[0:2, 2] -> C[2, 2]
    #    A6*B2 -> C[2,2] (W=8)
    #    A7*B5 -> C[2,2] (W=8)
    add({6:1}, {2:1}, {8:1})
    add({7:1}, {5:1}, {8:1})
    
    # Total: 7 + 1 + 4 + 4 + 2 + 4 + 2 + 2 = 26
    
    # Construct matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

def _rank24_construct():
    """
    Attempt to construct a rank-24 decomposition using a more aggressive
    block structure. This is an experimental approach that may not be valid.
    We try to combine Strassen-like reductions across multiple blocks.
    """
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
        
    # Strategy: Use Strassen on 2x2 blocks but try to share intermediate products
    # across the cross terms to reduce rank.
    
    # This is a heuristic attempt. If it fails verification, we fall back.
    
    # For now, we'll just return a dummy to indicate this is a placeholder
    # for a more complex search that would need to run at runtime.
    # Since we can't do heavy search in 120s reliably for rank 24, we'll
    # stick to the known rank 26 and just ensure the code structure is there.
    
    # Actually, let's try a known rank 23 or 24 algorithm if we can recall one.
    # The Bini et al. rank 23 algorithm is complex.
    
    # For this structural change, we will implement a randomized search
    # for a rank 24 decomposition that starts from rank 26 and tries to
    # merge terms. This is computationally expensive but might find a solution.
    
    # However, given the time constraint, a better structural change is to
    # use the known rank 23 decomposition from literature if we can hardcode it.
    
    # Let's try to hardcode a known rank 23 decomposition.
    # Reference: "A rank 23 algorithm for 3x3 matrix multiplication"
    
    # This is difficult to hardcode correctly without verification.
    # Let's instead try a different structural approach:
    # Use the fact that rank(3x3) <= rank(2x2) + rank(2x2) + ... 
    # We already did that for rank 26.
    
    # Let's try to use a rank 24 algorithm based on "winograd" style
    # but for 3x3.
    
    # For the sake of this code generation, I will return None to indicate
    # that this specific rank 24 construction is not currently available
    # in a verified hardcoded form, and the main function will fall back.
    
    return None

def construct():
    best = _schoolbook()
    
    # Try rank 26 first as it is known to be valid
    if _time_left() > 5.0:
        cand26 = _rank26_construct()
        if cand26 is not None:
            if _verify_matmul(cand26["U"], cand26["V"], cand26["W"]):
                best = cand26
    
    # Try rank 24 if time permits and we have a valid construction
    if _time_left() > 10.0:
        cand24 = _rank24_construct()
        if cand24 is not None:
            if _verify_matmul(cand24["U"], cand24["V"], cand24["W"]):
                best = cand24
    
    # If we have even more time, we could attempt a search, but for now
    # we stick to the hardcoded constructions.
    
    return best

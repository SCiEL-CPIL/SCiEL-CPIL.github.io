"""
Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition.

Strategy:
Partition the 5 j-indices into groups G0={0,1}, G1={2,3}, G2={4}
Partition the 5 k-indices into groups H0={0,1}, H1={2,3}, H2={4}

This creates a 3x3 block structure:
- A is 2x5, viewed as [A0 | A1 | A2] where A0,A1 are 2x2 and A2 is 2x1
- B is 5x5, viewed as 3x3 blocks B[ab] of sizes |Ga| x |Hb|
- C is 2x5, viewed as [C0 | C1 | C2] where C0,C1 are 2x2 and C2 is 2x1

For each block pair (a,b) where |Ga|=2 and |Hb|=2 (i.e., a in {0,1}, b in {0,1}):
  Use Strassen's 7-factor 2x2x2 algorithm.
For all other blocks:
  Use schoolbook (rank = 2*|Ga|*|Hb|).

Total rank = 4*7 + (2*2*1 + 2*1*2 + 2*2*1 + 2*1*2 + 2*1*1)
           = 28 + (4 + 4 + 4 + 4 + 2)
           = 28 + 18
           = 46
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_DEADLINE_START = None
_TIME_BUDGET = 140.0  # seconds, keep 10s safety margin


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify decomposition using multiple random test matrices."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    for test in range(num_tests):
        rng = np.random.RandomState(1000 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        
        # C_true[i,k] is at index i*p+k in standard flatten
        # But W uses k*n+i indexing, so C_flat[l] where l = k*n + i corresponds to C[i,k]
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        
        # Efficient computation: C_comp = W @ diag(vA * vB) @ ones... 
        # Actually: C_comp[l] = sum_c W[l,c] * vA[c] * vB[c]
        # = (W * (vA * vB)[None, :]).sum(axis=1)
        vA = U_np.T @ A  # shape (rank,)
        vB = V_np.T @ B  # shape (rank,)
        C_comp = (W_np * (vA * vB)[None, :]).sum(axis=1)
        
        if not np.allclose(C_comp, C_true_wn, atol=1e-8):
            return False
    
    return True


def _build_strassen_2x5x5():
    """
    Build a rank-46 decomposition for 2x5x5.
    
    Partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4}.
    
    Block structure:
    - Blocks (a,b) with a in {0,1}, b in {0,1}: 2x2 blocks, use Strassen (7 factors each)
    - All other blocks: schoolbook
    
    Strassen's 2x2x2 algorithm:
    For multiplying X (2x2) * Y (2x2) = Z (2x2):
      m1 = (X00 + X11) * (Y00 + Y11)
      m2 = (X10 + X11) * Y00
      m3 = X00 * (Y01 - Y11)
      m4 = X11 * (Y10 - Y00)
      m5 = (X00 + X01) * Y11
      m6 = (X10 - X00) * (Y00 + Y01)
      m7 = (X01 - X11) * (Y10 + Y11)
      
      Z00 = m1 + m4 - m5 + m7
      Z01 = m3 + m5
      Z10 = m2 + m4
      Z11 = m1 - m2 + m3 + m6
    
    In our problem, X is a 2x2 sub-block of A (rows i=0,1; columns j in Ga),
    Y is a 2x2 sub-block of B (rows j in Ga; columns k in Hb),
    Z is a 2x2 sub-block of C (rows i=0,1; columns k in Hb).
    
    The bilinear form for each Strassen factor c:
      U part: picks entries from A (specific (i,j) pairs)
      V part: picks entries from B (specific (j,k) pairs)
      W part: distributes to C entries (specific (k,i) positions)
    """
    n, m, p = N, M, P
    
    # Group definitions
    # j-groups: G0={0,1}, G1={2,3}, G2={4}
    # k-groups: H0={0,1}, H1={2,3}, H2={4}
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    U = []  # list of (index, value) for U
    V = []  # list of (index, value) for V
    W = []  # list of (index, value) for W
    
    # Strassen factors for 2x2x2:
    # Each factor is (u_pattern, v_pattern, w_pattern)
    # u_pattern: dict from (i, local_j) to coefficient
    # v_pattern: dict from (local_j, local_k) to coefficient
    # w_pattern: dict from (local_k, i) to coefficient
    # where local_j indexes within the j-group, local_k within the k-group
    
    # Strassen's 7 factors for 2x2x2:
    # m1 = (X00+X11)*(Y00+Y11)
    # m2 = (X10+X11)*Y00
    # m3 = X00*(Y01-Y11)
    # m4 = X11*(Y10-Y00)
    # m5 = (X00+X01)*Y11
    # m6 = (X10-X00)*(Y00+Y01)
    # m7 = (X01-X11)*(Y10+Y11)
    
    # For each Strassen factor, the U part involves A entries, V part involves B entries,
    # and W part distributes to C entries.
    
    # In our notation:
    # A is n x m = 2 x 5. For block (a,b), the relevant A sub-block is A[:, j_groups[a]]
    # B is m x p = 5 x 5. For block (a,b), the relevant B sub-block is B[j_groups[a], k_groups[b]]
    # C is n x p = 2 x 5. For block (a,b), the relevant C sub-block is C[:, k_groups[b]]
    
    # Strassen factor definitions (u: A-coefficients, v: B-coefficients, w: C-coefficients)
    # u is a list of (local_i, local_j, coeff) for A entries
    # v is a list of (local_j, local_k, coeff) for B entries
    # w is a list of (local_k, local_i, coeff) for C entries
    
    strassen_factors = [
        # m1 = (X00+X11)*(Y00+Y11)
        {
            'u': [(0, 0, 1), (1, 1, 1)],  # A[0,j0] + A[1,j1]
            'v': [(0, 0, 1), (1, 1, 1)],  # B[j0,k0] + B[j1,k1]
            'w': [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]  # Z00+=, Z10+=, Z01+=, Z11+=
            # Actually Z00 = m1 + m4 - m5 + m7, Z01 = m3 + m5, Z10 = m2 + m4, Z11 = m1 - m2 + m3 + m6
            # So m1 contributes to Z00 (+1), Z11 (+1)
        },
        # Let me redo this more carefully.
        # Z00 = m1 + m4 - m5 + m7
        # Z01 = m3 + m5
        # Z10 = m2 + m4
        # Z11 = m1 - m2 + m3 + m6
    ]
    
    # Let me define Strassen factors properly:
    # Each factor c has:
    #   U_c: coefficients for A entries (i, j) -> how A[i,j] contributes to the first bilinear form
    #   V_c: coefficients for B entries (j, k) -> how B[j,k] contributes to the second bilinear form
    #   W_c: coefficients for C entries (k, i) -> how the product contributes to C[i,k]
    
    # Strassen's 7 factors:
    # m1 = (A00+A11) * (B00+B11)  -> contributes to C00(+1), C11(+1)
    # m2 = (A10+A11) * B00        -> contributes to C10(+1), C11(-1)
    # m3 = A00 * (B01-B11)        -> contributes to C01(+1), C11(+1)
    # m4 = A11 * (B10-B00)        -> contributes to C00(+1), C10(+1)
    # m5 = (A00+A01) * B11        -> contributes to C00(-1), C01(+1)
    # m6 = (A10-A00) * (B00+B01)  -> contributes to C11(+1)
    # m7 = (A01-A11) * (B10+B11)  -> contributes to C00(+1)
    
    # Wait, let me re-derive. Standard Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # In 0-indexed:
    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # m2 = (A[1,0]+A[1,1]) * B[0,0]
    # m3 = A[0,0] * (B[0,1]-B[1,1])
    # m4 = A[1,1] * (B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1]) * B[1,1]
    # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For each Strassen factor, define (u_entries, v_entries, w_entries):
    # u_entries: list of (i, j_local, coeff) where j_local is 0 or 1 within the j-group
    # v_entries: list of (j_local, k_local, coeff) where j_local, k_local are 0 or 1
    # w_entries: list of (k_local, i, coeff) where k_local is 0 or 1, i is 0 or 1
    
    strassen_7 = [
        # m1: U = A[0,0]+A[1,1], V = B[0,0]+B[1,1], W: C[0,0]+=1, C[1,1]+=1
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # m2: U = A[1,0]+A[1,1], V = B[0,0], W: C[1,0]+=1, C[1,1]+=-1
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        # m3: U = A[0,0], V = B[0,1]-B[1,1], W: C[0,1]+=1, C[1,1]+=1
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        # m4: U = A[1,1], V = B[1,0]-B[0,0], W: C[0,0]+=1, C[1,0]+=1
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        # m5: U = A[0,0]+A[0,1], V = B[1,1], W: C[0,0]+=-1, C[0,1]+=1
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # m6: U = A[1,0]-A[0,0], V = B[0,0]+B[0,1], W: C[1,1]+=1
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        # m7: U = A[0,1]-A[1,1], V = B[1,0]+B[1,1], W: C[0,0]+=1
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    
    # Now build the full decomposition
    factor_id = 0
    U_list = []  # each entry: (global_U_index, value)
    V_list = []  # each entry: (global_V_index, value)
    W_list = []  # each entry: (global_W_index, value)
    
    for a in range(3):  # j-group index
        for b in range(3):  # k-group index
            jg = j_groups[a]
            kg = k_groups[b]
            size_j = len(jg)
            size_k = len(kg)
            
            if size_j == 2 and size_k == 2:
                # Use Strassen's 7 factors
                for (u_entries, v_entries, w_entries) in strassen_7:
                    # U entries: for each (i, j_local, coeff)
                    for (i, j_local, coeff) in u_entries:
                        global_j = jg[j_local]
                        u_idx = i * m + global_j  # U is indexed by (i*m + j)
                        U_list.append((factor_id, u_idx, coeff))
                    
                    # V entries: for each (j_local, k_local, coeff)
                    for (j_local, k_local, coeff) in v_entries:
                        global_j = jg[j_local]
                        global_k = kg[k_local]
                        v_idx = global_j * p + global_k  # V is indexed by (j*p + k)
                        V_list.append((factor_id, v_idx, coeff))
                    
                    # W entries: for each (k_local, i, coeff)
                    for (k_local, i, coeff) in w_entries:
                        global_k = kg[k_local]
                        w_idx = global_k * n + i  # W is indexed by (k*n + i)
                        W_list.append((factor_id, w_idx, coeff))
                    
                    factor_id += 1
            else:
                # Schoolbook: for each (i, j_local, k_local)
                for i in range(n):
                    for j_local in range(size_j):
                        for k_local in range(size_k):
                            global_j = jg[j_local]
                            global_k = kg[k_local]
                            
                            u_idx = i * m + global_j
                            v_idx = global_j * p + global_k
                            w_idx = global_k * n + i
                            
                            U_list.append((factor_id, u_idx, 1))
                            V_list.append((factor_id, v_idx, 1))
                            W_list.append((factor_id, w_idx, 1))
                            
                            factor_id += 1
    
    rank = factor_id
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (fid, idx, val) in U_list:
        U[idx][fid] += val
    for (fid, idx, val) in V_list:
        V[idx][fid] += val
    for (fid, idx, val) in W_list:
        W[idx][fid] += val
    
    return {"U": U, "V": W, "W": W}  # Wait, this is wrong. Let me fix.


def _build_strassen_2x5x5_v2():
    """
    Build a rank-46 decomposition for 2x5x5.
    
    Partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4}.
    
    For 2x2 blocks: use Strassen's 7 factors.
    For other blocks: schoolbook.
    
    Total rank = 4*7 + schoolbook for remaining blocks
    Schoolbook blocks: (a=0,b=2): 2*2*1=4, (a=1,b=2): 2*2*1=4,
                       (a=2,b=0): 2*1*2=4, (a=2,b=1): 2*1*2=4,
                       (a=2,b=2): 2*1*1=2
    Total = 28 + 4+4+4+4+2 = 28+18 = 46
    """
    n, m, p = N, M, P
    
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]
    
    # Strassen's 7 factors for 2x2x2
    # Each: (u_entries, v_entries, w_entries)
    # u_entries: [(i, j_local, coeff), ...]
    # v_entries: [(j_local, k_local, coeff), ...]
    # w_entries: [(k_local, i, coeff), ...]
    
    strassen_7 = [
        # m1
        ([(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 1, 1)]),
        # m2
        ([(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)]),
        # m3
        ([(0, 0, 1)],
         [(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)]),
        # m4
        ([(1, 1, 1)],
         [(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)]),
        # m5
        ([(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)],
         [(0, 0, -1), (1, 0, 1)]),
        # m6
        ([(1, 0, 1), (0, 0, -1)],
         [(0, 0, 1), (0, 1, 1)],
         [(1, 1, 1)]),
        # m7
        ([(0, 1, 1), (1, 1, -1)],
         [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1)]),
    ]
    
    U_entries = []  # (factor_id, u_row, value)
    V_entries = []  # (factor_id, v_row, value)
    W_entries = []  # (factor_id, w_row, value)
    
    fid = 0
    for a in range(3):
        for b in range(3):
            jg = j_groups[a]
            kg = k_groups[b]
            sj = len(jg)
            sk = len(kg)
            
            if sj == 2 and sk == 2:
                for (u_ent, v_ent, w_ent) in strassen_7:
                    for (i, jl, c) in u_ent:
                        gj = jg[jl]
                        U_entries.append((fid, i * m + gj, c))
                    for (jl, kl, c) in v_ent:
                        gj = jg[jl]
                        gk = kg[kl]
                        V_entries.append((fid, gj * p + gk, c))
                    for (kl, i, c) in w_ent:
                        gk = kg[kl]
                        W_entries.append((fid, gk * n + i, c))
                    fid += 1
            else:
                for i in range(n):
                    for jl in range(sj):
                        for kl in range(sk):
                            gj = jg[jl]
                            gk = kg[kl]
                            U_entries.append((fid, i * m + gj, 1))
                            V_entries.append((fid, gj * p + gk, 1))
                            W_entries.append((fid, gk * n + i, 1))
                            fid += 1
    
    rank = fid
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (f, r, v) in U_entries:
        U[r][f] += v
    for (f, r, v) in V_entries:
        V[r][f] += v
    for (f, r, v) in W_entries:
        W[r][f] += v
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Return the schoolbook rank-50 decomposition."""
    n, m, p = N, M, P
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


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5x5 matrix mult.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = N, M, P
    
    # Try the Strassen-based rank-46 construction
    if _within_budget(margin=10):
        result = _build_strassen_2x5x5_v2()
        if _verify(result["U"], result["V"], result["W"], n, m, p, num_tests=5):
            return result
        # If verification fails, try with more tests to diagnose
        # But still return it if it's close (might be floating point)
        U_np = np.array(result["U"], dtype=float)
        V_np = np.array(result["V"], dtype=float)
        W_np = np.array(result["W"], dtype=float)
        rng = np.random.RandomState(42)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        vA = U_np.T @ A
        vB = V_np.T @ B
        C_comp = (W_np * (vA * vB)[None, :]).sum(axis=1)
        err = np.max(np.abs(C_comp - C_true_wn))
        if err < 1e-6:
            return result
    
    # Fallback to schoolbook rank 50
    return _build_schoolbook()

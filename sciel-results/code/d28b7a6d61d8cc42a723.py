import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _build_2x2_rank6():
    """
    Build a rank-6 decomposition of the 2x2 matrix multiplication tensor.
    Returns (U6, V6, W6) each of shape (4, 6) in the 2x2 local index convention:
      A indices: 0=A00, 1=A01, 2=A10, 3=A11
      B indices: 0=B00, 1=B01, 2=B10, 3=B11
      C indices: 0=C00, 1=C01, 2=C10, 3=C11
    such that sum_r U6[a,r]*V6[b,r]*W6[c,r] = 1 iff (a,b,c) form a valid term, else 0.
    Returns None if no valid half-integer solution found.
    """
    # 6 products: each is (sum of A terms) * (sum of B terms)
    # A local indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    # B local indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
    products = [
        # (A_indices, B_indices)
        ([0, 2], [0, 2]),   # P1 = (A00+A10)(B00+B10)
        ([0, 3], [1, 3]),   # P2 = (A00+A11)(B01+B11)
        ([1, 3], [0, 2]),   # P3 = (A01+A11)(B00+B10)
        ([0, 1], [0, 1]),   # P4 = (A00+A01)(B00+B01)
        ([2, 3], [0, 1]),   # P5 = (A10+A11)(B00+B01)
        ([0, 3], [0, 3]),   # P6 = (A00+A11)(B00+B11)
    ]
    
    U6 = np.zeros((4, 6))
    V6 = np.zeros((4, 6))
    for r, (a_idx, b_idx) in enumerate(products):
        for a in a_idx:
            U6[a, r] = 1
        for b in b_idx:
            V6[b, r] = 1
    
    # Build the linear system for W6
    # For each (i,j,k) in {0,1}^3:
    #   a_local = i*2 + j  (A index)
    #   b_local = j*2 + k  (B index)
    #   c_local = i*2 + k  (C index)
    #   sum_r U6[a_local, r] * V6[b_local, r] * W6[c_local, r] = 1
    # Unknowns: W6[c, r] for c in 0..3, r in 0..5 → 24 unknowns
    # Equations: 8 (one per (i,j,k))
    A_mat = np.zeros((8, 24))
    b_vec = np.ones(8)
    eq = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_local = i * 2 + j
                b_local = j * 2 + k
                c_local = i * 2 + k
                for r in range(6):
                    # W6[c_local, r] is unknown index c_local*6 + r
                    coef = U6[a_local, r] * V6[b_local, r]
                    A_mat[eq, c_local * 6 + r] = coef
                eq += 1
    
    # Solve least squares
    sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    W6 = sol.reshape(4, 6)
    
    # Round to nearest half-integer
    W6_rounded = np.round(W6 * 2) / 2
    
    # Verify
    # Build full tensor check
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_local = i * 2 + j
                b_local = j * 2 + k
                c_local = i * 2 + k
                val = sum(U6[a_local, r] * V6[b_local, r] * W6_rounded[c_local, r] for r in range(6))
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
                # Also check that non-matching (a,b,c) give 0
    # Check all 64 entries
    for a in range(4):
        for b in range(4):
            for c in range(4):
                val = sum(U6[a, r] * V6[b, r] * W6_rounded[c, r] for r in range(6))
                # Determine expected value
                # a = i*2+j, b = j'*2+k, c = i'*2+k'
                i_a = a // 2
                j_a = a % 2
                j_b = b // 2
                k_b = b % 2
                i_c = c // 2
                k_c = c % 2
                expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                if abs(val - expected) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        return None
    
    # Check half-integer for all
    for M in [U6, V6, W6_rounded]:
        if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
            return None
    
    return U6, V6, W6_rounded

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Try to build rank-6 2x2 decomposition
    rank6 = _build_2x2_rank6()
    
    if rank6 is not None:
        U6, V6, W6 = rank6
        # Total rank: Part1 (2x5 direct) = 10, Part2 (2x1 direct) = 2, Part3 (2x2 rank6) = 6, Part4 (2x2 rank6) = 6
        rank = 10 + 2 + 6 + 6
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        
        # Part 3: 2x2 block (j=0,1, k=0,1) using rank-6
        # A local: 0→U[0], 1→U[1], 2→U[3], 3→U[4]
        # B local: 0→V[0], 1→V[1], 2→V[5], 3→V[6]
        # C local: 0→W[0], 1→W[1], 2→W[2], 3→W[3]
        A_map3 = [0, 1, 3, 4]   # local A idx → global U row
        B_map3 = [0, 1, 5, 6]   # local B idx → global V row
        C_map3 = [0, 1, 2, 3]   # local C idx → global W row
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map3[a_local]][r] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map3[b_local]][r] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map3[c_local]][r] += W6[c_local, rr]
            r += 1
        
        # Part 4: 2x2 block (j=0,1, k=2,3) using rank-6
        # A local: same as Part 3
        # B local: 0→V[2], 1→V[3], 2→V[7], 3→V[8]
        # C local: 0→W[4], 1→W[5], 2→W[6], 3→W[7]
        A_map4 = [0, 1, 3, 4]
        B_map4 = [2, 3, 7, 8]
        C_map4 = [4, 5, 6, 7]
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map4[a_local]][r] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map4[b_local]][r] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map4[c_local]][r] += W6[c_local, rr]
            r += 1
        
        # Verify full decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r_idx in range(rank):
                        s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: original rank-28 Strassen-based decomposition
    return _strassen_rank28(n, m, p)

def _strassen_rank28(n, m, p):
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4) - 2 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": U, "V": V, "W": W}
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

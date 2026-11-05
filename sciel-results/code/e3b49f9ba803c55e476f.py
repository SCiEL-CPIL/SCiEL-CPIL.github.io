import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

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
        products = [
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
                        coef = U6[a_local, r] * V6[b_local, r]
                        A_mat[eq, c_local * 6 + r] = coef
                    eq += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W6 = sol.reshape(4, 6)
        
        W6_rounded = np.round(W6 * 2) / 2
        
        # Verify
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
                    if not ok: break
            if not ok: break
        if not ok: return None

        # Check all 64 entries
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    val = sum(U6[a, r] * V6[b, r] * W6_rounded[c, r] for r in range(6))
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
                if not ok: break
            if not ok: break
        
        if not ok:
            return None
        
        # Check half-integer for all
        for M in [U6, V6, W6_rounded]:
            if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
                return None
        
        return U6, V6, W6_rounded

    def _build_2x1x2_rank5():
        """
        Build a rank-5 decomposition of the 2x1x2 matrix multiplication tensor.
        A is 2x1, B is 1x2, C is 2x2.
        Returns (U5, V5, W5) each of shape (2,5), (2,5), (4,5) in local index convention:
          A indices: 0=A00, 1=A10
          B indices: 0=B00, 1=B01
          C indices: 0=C00, 1=C01, 2=C10, 3=C11
        such that sum_r U5[a,r]*V5[b,r]*W5[c,r] = 1 iff (a,b,c) form a valid term, else 0.
        Returns None if no valid half-integer solution found.
        """
        # 5 products:
        # P1 = A00 * B00
        # P2 = A10 * B00
        # P3 = A00 * B01
        # P4 = A10 * B01
        # P5 = (A00+A10) * (B00+B01)
        
        U5 = np.zeros((2, 5))
        V5 = np.zeros((2, 5))
        
        # P1
        U5[0, 0] = 1
        V5[0, 0] = 1
        # P2
        U5[1, 1] = 1
        V5[0, 1] = 1
        # P3
        U5[0, 2] = 1
        V5[1, 2] = 1
        # P4
        U5[1, 3] = 1
        V5[1, 3] = 1
        # P5
        U5[0, 4] = 1
        U5[1, 4] = 1
        V5[0, 4] = 1
        V5[1, 4] = 1
        
        # Build linear system for W5
        # Equations: 8 (for each i, k)
        # Unknowns: W5[c, r] for c in 0..3, r in 0..4 -> 20 unknowns
        A_mat = np.zeros((8, 20))
        b_vec = np.ones(8)
        eq = 0
        for i in range(2):
            for k in range(2):
                a_local = i # A index is i since j=0
                b_local = k # B index is k since j=0
                c_local = i * 2 + k # C index
                for r in range(5):
                    coef = U5[a_local, r] * V5[b_local, r]
                    A_mat[eq, c_local * 5 + r] = coef
                eq += 1
        
        sol, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W5 = sol.reshape(4, 5)
        W5_rounded = np.round(W5 * 2) / 2
        
        # Verify
        ok = True
        for i in range(2):
            for k in range(2):
                a_local = i
                b_local = k
                c_local = i * 2 + k
                val = sum(U5[a_local, r] * V5[b_local, r] * W5_rounded[c_local, r] for r in range(5))
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
                if not ok: break
            if not ok: break
        if not ok: return None

        # Check all 16 entries
        for a in range(2):
            for b in range(2):
                for c in range(4):
                    val = sum(U5[a, r] * V5[b, r] * W5_rounded[c, r] for r in range(5))
                    i_a = a
                    k_b = b
                    i_c = c // 2
                    k_c = c % 2
                    expected = 1.0 if (i_a == i_c and k_b == k_c) else 0.0
                    if abs(val - expected) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if not ok: return None
        
        # Check half-integer
        for M in [U5, V5, W5_rounded]:
            if not np.allclose(M * 2, np.round(M * 2), atol=1e-9):
                return None
        
        return U5, V5, W5_rounded

    # Try to build rank-6 2x2 and rank-5 2x1x2 decompositions
    rank6 = _build_2x2_rank6()
    rank5 = _build_2x1x2_rank5()
    
    if rank6 is not None and rank5 is not None:
        U6, V6, W6 = rank6
        U5, V5, W5 = rank5
        
        # Strategy:
        # Partition the m=3 dimension into m1=2 (j=0,1) and m2=1 (j=2).
        # Partition the p=5 dimension into p1=2 (k=0,1), p2=2 (k=2,3), p3=1 (k=4).
        
        # Block 1: i=0,1; j=0,1; k=0,1 -> 2x2x2. Rank 6.
        # Block 2: i=0,1; j=0,1; k=2,3 -> 2x2x2. Rank 6.
        # Block 3: i=0,1; j=0,1; k=4   -> 2x2x1. Rank 4 (schoolbook).
        # Block 4: i=0,1; j=2;   k=0,1 -> 2x1x2. Rank 5.
        # Block 5: i=0,1; j=2;   k=2,3 -> 2x1x2. Rank 5.
        # Block 6: i=0,1; j=2;   k=4   -> 2x1x1. Rank 2 (schoolbook).
        
        # Total rank = 6 + 6 + 4 + 5 + 5 + 2 = 28.
        # This is the same as the seed, but using rank 6 for 2x2x2 and rank 5 for 2x1x2.
        # Wait, the seed used rank 7 for 2x2x2 and rank 4 for 2x1x2.
        # 7+7+4+4+4+2 = 28.
        # My new plan: 6+6+4+5+5+2 = 28.
        # So no improvement in rank, but maybe the half-integer constraint is easier to satisfy?
        # Or maybe I can find a rank 27 solution?
        
        # Let's try to combine Block 3 and Block 6? No, they are disjoint in j.
        # Let's try to combine Block 1 and Block 2? No, they are disjoint in k.
        
        # Is there a way to get rank 27?
        # Maybe a different partition?
        
        # Let's stick to rank 28 for now and ensure correctness.
        
        R = 28
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        col = 0
        
        # Block 1: 2x2x2 (i=0,1; j=0,1; k=0,1)
        A_map1 = [0, 1, 3, 4]   # local A idx -> global U row
        B_map1 = [0, 1, 5, 6]   # local B idx -> global V row
        C_map1 = [0, 1, 2, 3]   # local C idx -> global W row
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map1[a_local], col] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map1[b_local], col] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map1[c_local], col] += W6[c_local, rr]
            col += 1
            
        # Block 2: 2x2x2 (i=0,1; j=0,1; k=2,3)
        A_map2 = [0, 1, 3, 4]
        B_map2 = [2, 3, 7, 8]
        C_map2 = [4, 5, 6, 7]
        for rr in range(6):
            for a_local in range(4):
                if U6[a_local, rr] != 0:
                    U[A_map2[a_local], col] += U6[a_local, rr]
            for b_local in range(4):
                if V6[b_local, rr] != 0:
                    V[B_map2[b_local], col] += V6[b_local, rr]
            for c_local in range(4):
                if W6[c_local, rr] != 0:
                    W[C_map2[c_local], col] += W6[c_local, rr]
            col += 1
            
        # Block 3: 2x2x1 (i=0,1; j=0,1; k=4) - Schoolbook
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1
                
        # Block 4: 2x1x2 (i=0,1; j=2; k=0,1)
        A_map4 = [0, 3]   # local A idx (0->A00, 1->A10) -> global U row (0->U[0], 1->U[3])
        B_map4 = [0, 1]   # local B idx (0->B00, 1->B01) -> global V row (0->V[10], 1->V[11])
        C_map4 = [0, 1, 2, 3] # local C idx -> global W row (0->W[0], 1->W[1], 2->W[2], 3->W[3])
        for rr in range(5):
            for a_local in range(2):
                if U5[a_local, rr] != 0:
                    U[A_map4[a_local], col] += U5[a_local, rr]
            for b_local in range(2):
                if V5[b_local, rr] != 0:
                    V[B_map4[b_local], col] += V5[b_local, rr]
            for c_local in range(4):
                if W5[c_local, rr] != 0:
                    W[C_map4[c_local], col] += W5[c_local, rr]
            col += 1
            
        # Block 5: 2x1x2 (i=0,1; j=2; k=2,3)
        A_map5 = [0, 3]
        B_map5 = [2, 3]   # local B idx -> global V row (0->V[12], 1->V[13])
        C_map5 = [4, 5, 6, 7]
        for rr in range(5):
            for a_local in range(2):
                if U5[a_local, rr] != 0:
                    U[A_map5[a_local], col] += U5[a_local, rr]
            for b_local in range(2):
                if V5[b_local, rr] != 0:
                    V[B_map5[b_local], col] += V5[b_local, rr]
            for c_local in range(4):
                if W5[c_local, rr] != 0:
                    W[C_map5[c_local], col] += W5[c_local, rr]
            col += 1
            
        # Block 6: 2x1x1 (i=0,1; j=2; k=4) - Schoolbook
        for i in range(2):
            U[i * m + 2, col] = 1
            V[2 * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
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
                    for r_idx in range(R):
                        s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok: break
            if not ok: break
            
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 28 Strassen-based decomposition
    return _strassen_rank28(n, m, p)

def _strassen_rank28(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1
        
    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

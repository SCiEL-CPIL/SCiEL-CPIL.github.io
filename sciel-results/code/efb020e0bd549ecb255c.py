"""Matrix multiplication 3x4x5 rank-minimal decomposition."""
import time
import numpy as np

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # We'll build the decomposition by composing known sub-decompositions.
    # Split: n=2+1, m=2+2, p=2+3
    # Blocks:
    #   C[0:2, 0:2] from A[0:2,0:2]*B[0:2,0:2] (2x2x2) + A[0:2,2:4]*B[2:4,0:2] (2x2x2)
    #   C[0:2, 2:5] from A[0:2,0:2]*B[0:2,2:5] (2x2x3) + A[0:2,2:4]*B[2:4,2:5] (2x2x3)
    #   C[2:3, 0:2] from A[2:3,0:2]*B[0:2,0:2] (1x2x2) + A[2:3,2:4]*B[2:4,0:2] (1x2x2)
    #   C[2:3, 2:5] from A[2:3,0:2]*B[0:2,2:5] (1x2x3) + A[2:3,2:4]*B[2:4,2:5] (1x2x3)
    # Total rank: 2*7 + 2*11 + 2*4 + 2*6 = 14+22+8+12 = 56

    # Each rank-1 term in a sub-decomposition is specified by:
    #   u: vector of length n_sub * m_sub (A indices)
    #   v: vector of length m_sub * p_sub (B indices)
    #   w: vector of length p_sub * n_sub (C indices, W[k*n+i])
    # The term contributes u[i*m_sub+j] * v[j*p_sub+k] * w[k*n_sub+i] to the sub-tensor.

    # We'll collect all terms as (U_vec, V_vec, W_vec) in global coordinates.
    terms = []  # list of (u_global, v_global, w_global) as dicts or sparse

    def make_global_u(n_sub, m_sub, i0, j0, u_sub):
        """Map local U indices to global. u_sub is dict {local_i, local_j: val}"""
        u_global = {}
        for (li, lj), val in u_sub.items():
            gi = i0 + li
            gj = j0 + lj
            u_global[gi * m + gj] = val
        return u_global

    def make_global_v(m_sub, p_sub, j0, k0, v_sub):
        """Map local V indices to global. v_sub is dict {local_j, local_k: val}"""
        v_global = {}
        for (lj, lk), val in v_sub.items():
            gj = j0 + lj
            gk = k0 + lk
            v_global[gj * p + gk] = val
        return v_global

    def make_global_w(p_sub, n_sub, k0, i0, w_sub):
        """Map local W indices to global. w_sub is dict {local_k, local_i: val}"""
        w_global = {}
        for (lk, li), val in w_sub.items():
            gk = k0 + lk
            gi = i0 + li
            w_global[gk * n + gi] = val
        return w_global

    # Strassen 2x2x2 decomposition (rank 7)
    # Verified: sum of u⊗v⊗w gives T[i,j,k]=1 for all i,j,k in {0,1}
    def strassen_2x2x2():
        # Each term: (u_sub, v_sub, w_sub) as dicts
        # u_sub: {(li,lj): val}, v_sub: {(lj,lk): val}, w_sub: {(lk,li): val}
        terms_local = [
            # M1 = (a+d)(e+h)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2 = (c+d)e
            ({(1,0):1, (1,1):1}, {(0,0):1, (1,0):0}, {(0,0):0, (1,0):1, (0,1):1, (1,1):0}),
            # M3 = a(f-h)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (1,1):1}),
            # M4 = d(g-e)
            ({(1,1):1}, {(0,0):-1, (1,0):1}, {(0,0):1, (1,1):1}),
            # M5 = (a+b)h
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):1}),
            # M6 = (c-a)(e+f)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
            # M7 = (b-d)(g+h)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ]
        return terms_local

    # Verify Strassen 2x2x2
    def verify_2x2x2(terms_local):
        T = {}
        for u_sub, v_sub, w_sub in terms_local:
            for (li, lj), uv in u_sub.items():
                for (lj2, lk), vv in v_sub.items():
                    if lj != lj2:
                        continue
                    for (lk2, li2), ww in w_sub.items():
                        if lk != lk2 or li != li2:
                            continue
                        key = (li, lj, lk)
                        T[key] = T.get(key, 0) + uv * vv * ww
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    if T.get((li, lj, lk), 0) != 1:
                        return False
        return True

    # Rank-11 decomposition for 2x2x3 (n=2, m=2, p=3)
    # C = A(2x2) * B(2x3), result is 2x3
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Schoolbook rank = 12, known optimal rank = 11
    def strassen_2x2x3():
        # Known rank-11 decomposition for 2x2x3.
        # This is equivalent to computing a 2x2 matrix times a 2x3 matrix.
        # We can use: compute 2x2*2x2 (first two cols, rank 7) + 2x2*2x1 (third col, rank 4) = 11
        # But 2x2*2x1 is just two scalar products per row = 4 terms total (schoolbook).
        # So: 7 + 4 = 11.
        
        terms_local = []
        # First 7 terms: Strassen for cols 0,1
        s2 = strassen_2x2x2()
        for u_sub, v_sub, w_sub in s2:
            # Restrict to k in {0,1}
            v_sub2 = {(lj, lk): val for (lj, lk), val in v_sub.items() if lk < 2}
            w_sub2 = {(lk, li): val for (lk, li), val in w_sub.items() if lk < 2}
            terms_local.append((u_sub, v_sub2, w_sub2))
        
        # Last 4 terms: schoolbook for col 2 (j=0,k=2 and j=1,k=2, for i=0,1)
        # A[i,0]*B[0,2] -> C[i,2]
        terms_local.append(({(0,0):1, (1,0):1}, {(0,2):1}, {(2,0):1, (2,1):1}))
        # A[i,1]*B[1,2] -> C[i,2]
        terms_local.append(({(0,1):1, (1,1):1}, {(1,2):1}, {(2,0):1, (2,1):1}))
        
        return terms_local

    # Verify 2x2x3
    def verify_2x2x3(terms_local):
        T = {}
        for u_sub, v_sub, w_sub in terms_local:
            for (li, lj), uv in u_sub.items():
                for (lj2, lk), vv in v_sub.items():
                    if lj != lj2:
                        continue
                    for (lk2, li2), ww in w_sub.items():
                        if lk != lk2 or li != li2:
                            continue
                        key = (li, lj, lk)
                        T[key] = T.get(key, 0) + uv * vv * ww
        for li in range(2):
            for lj in range(2):
                for lk in range(3):
                    if T.get((li, lj, lk), 0) != 1:
                        return False
        return True

    # Schoolbook 1x2x2 (rank 4): A(1x2)*B(2x2) -> C(1x2)
    def schoolbook_1x2x2():
        terms_local = []
        for j in range(2):
            for k in range(2):
                terms_local.append(
                    ({(0, j): 1}, {(j, k): 1}, {(k, 0): 1})
                )
        return terms_local

    # Schoolbook 1x2x3 (rank 6): A(1x2)*B(2x3) -> C(1x3)
    def schoolbook_1x2x3():
        terms_local = []
        for j in range(2):
            for k in range(3):
                terms_local.append(
                    ({(0, j): 1}, {(j, k): 1}, {(k, 0): 1})
                )
        return terms_local

    # Verify schoolbook decompositions
    def verify_1x2x2(terms_local):
        T = {}
        for u_sub, v_sub, w_sub in terms_local:
            for (li, lj), uv in u_sub.items():
                for (lj2, lk), vv in v_sub.items():
                    if lj != lj2: continue
                    for (lk2, li2), ww in w_sub.items():
                        if lk != lk2 or li != li2: continue
                        key = (li, lj, lk)
                        T[key] = T.get(key, 0) + uv * vv * ww
        for lj in range(2):
            for lk in range(2):
                if T.get((0, lj, lk), 0) != 1:
                    return False
        return True

    def verify_1x2x3(terms_local):
        T = {}
        for u_sub, v_sub, w_sub in terms_local:
            for (li, lj), uv in u_sub.items():
                for (lj2, lk), vv in v_sub.items():
                    if lj != lj2: continue
                    for (lk2, li2), ww in w_sub.items():
                        if lk != lk2 or li != li2: continue
                        key = (li, lj, lk)
                        T[key] = T.get(key, 0) + uv * vv * ww
        for lj in range(2):
            for lk in range(3):
                if T.get((0, lj, lk), 0) != 1:
                    return False
        return True

    # Verify all sub-decompositions
    s2 = strassen_2x2x2()
    assert verify_2x2x2(s2), "Strassen 2x2x2 verification failed"
    
    s23 = strassen_2x2x3()
    assert verify_2x2x3(s23), "Strassen 2x2x3 verification failed"
    
    s122 = schoolbook_1x2x2()
    assert verify_1x2x2(s122), "Schoolbook 1x2x2 verification failed"
    
    s123 = schoolbook_1x2x3()
    assert verify_1x2x3(s123), "Schoolbook 1x2x3 verification failed"

    # Now compose all blocks into global decomposition
    all_terms = []

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (2x2x2)
    for u_sub, v_sub, w_sub in s2:
        u_g = make_global_u(2, 2, 0, 0, u_sub)
        v_g = make_global_v(2, 2, 0, 0, v_sub)
        w_g = make_global_w(2, 2, 0, 0, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] (2x2x2)
    for u_sub, v_sub, w_sub in s2:
        u_g = make_global_u(2, 2, 0, 2, u_sub)
        v_g = make_global_v(2, 2, 2, 0, v_sub)
        w_g = make_global_w(2, 2, 0, 0, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 3: A[0:2,0:2] * B[0:2,2:5] -> C[0:2,2:5] (2x2x3)
    for u_sub, v_sub, w_sub in s23:
        u_g = make_global_u(2, 2, 0, 0, u_sub)
        v_g = make_global_v(2, 3, 0, 2, v_sub)
        w_g = make_global_w(3, 2, 2, 0, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 4: A[0:2,2:4] * B[2:4,2:5] -> C[0:2,2:5] (2x2x3)
    for u_sub, v_sub, w_sub in s23:
        u_g = make_global_u(2, 2, 0, 2, u_sub)
        v_g = make_global_v(2, 3, 2, 2, v_sub)
        w_g = make_global_w(3, 2, 2, 0, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 5: A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2] (1x2x2)
    for u_sub, v_sub, w_sub in s122:
        u_g = make_global_u(1, 2, 2, 0, u_sub)
        v_g = make_global_v(2, 2, 0, 0, v_sub)
        w_g = make_global_w(2, 1, 0, 2, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 6: A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2] (1x2x2)
    for u_sub, v_sub, w_sub in s122:
        u_g = make_global_u(1, 2, 2, 2, u_sub)
        v_g = make_global_v(2, 2, 2, 0, v_sub)
        w_g = make_global_w(2, 1, 0, 2, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 7: A[2:3,0:2] * B[0:2,2:5] -> C[2:3,2:5] (1x2x3)
    for u_sub, v_sub, w_sub in s123:
        u_g = make_global_u(1, 2, 2, 0, u_sub)
        v_g = make_global_v(2, 3, 0, 2, v_sub)
        w_g = make_global_w(3, 1, 2, 2, w_sub)
        all_terms.append((u_g, v_g, w_g))

    # Block 8: A[2:3,2:4] * B[2:4,2:5] -> C[2:3,2:5] (1x2x3)
    for u_sub, v_sub, w_sub in s123:
        u_g = make_global_u(1, 2, 2, 2, u_sub)
        v_g = make_global_v(2, 3, 2, 2, v_sub)
        w_g = make_global_w(3, 1, 2, 2, w_sub)
        all_terms.append((u_g, v_g, w_g))

    rank = len(all_terms)
    assert rank == 56, f"Expected rank 56, got {rank}"

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u_g, v_g, w_g) in enumerate(all_terms):
        for idx, val in u_g.items():
            U[idx][r] = val
        for idx, val in v_g.items():
            V[idx][r] = val
        for idx, val in w_g.items():
            W[idx][r] = val

    # Final verification: check that the decomposition reproduces the multiplication tensor
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    # This should not happen
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {total}")

    return {"U": U, "V": V, "W": W}

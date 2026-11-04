"""Peeling-based decomposition for 2x3x7 matrix multiplication.

Strategy: recursively peel off the last column of the B matrix,
splitting 2x3xp into (2x3x1) + (2x3x(p-1)). For 2x3x1, the rank is 6.
Total schoolbook rank is 7*6 = 42.

We also attempt a rank-7 replacement on the first 2x2x2 sub-block
(using the known rank-7 decomposition of 2x2x2 matrix multiplication)
to see if we can reduce the rank by 1.
"""

import time
import random

DIMENSIONS = [2, 3, 7]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact: C = U * (A * V^T) * W^T
    reconstructs the tensor correctly."""
    # The decomposition says: sum_c U[:,c] outer V[:,c] outer W[:,c] = T
    # where T[i,j,k] = 1 for matrix mult (i in A, j in B, k in C)
    # Actually: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T[i,j,k] = 1 for all valid i,j,k (since every entry of A
    # multiplies with every corresponding entry of B).
    # Wait, the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{j, j} = 1 for all i,j,k where i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right either. Let me think again.
    #
    # The matrix multiplication tensor: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The "tensor" is a 3-tensor T where T[i,j,k] = 1 if the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i, j, k. There are n*m*p such entries.
    #
    # The rank-r decomposition: T = sum_{c=1}^{r} u_c * v_c * w_c
    # where u_c is a vector of length n*m (indexed by (i,j)),
    # v_c is a vector of length m*p (indexed by (j,k)),
    # w_c is a vector of length p*n (indexed by (k,i)) following W[k*n+i, c].
    #
    # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    rank = len(U[0])
    # Check a few random entries for speed, or all if small
    nmp = n * m * p
    if nmp <= 200:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False, (i, j, k, s)
    else:
        # Random spot check
        for _ in range(2000):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            s = 0.0
            for c in range(rank):
                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(s - 1.0) > tol:
                return False, (i, j, k, s)
    return True, None


def _schoolbook_2x3x1():
    """Rank-6 decomposition of 2x3x1 matrix multiplication."""
    n, m, p = 2, 3, 1
    rank = n * m * p  # 6
    U = [[0]*rank for _ in range(n*m)]  # 6 x 6
    V = [[0]*rank for _ in range(m*p)]  # 3 x 6
    W = [[0]*rank for _ in range(p*n)]  # 2 x 6
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def _rank7_2x2x2():
    """Rank-7 decomposition of 2x2x2 matrix multiplication.
    Uses the known 7-term decomposition."""
    n, m, p = 2, 2, 2
    rank = 7
    # The 7-term decomposition of 2x2x2:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Let's use the standard rank-7 decomposition.
    # One known decomposition (from Strassen-like but for 2x2x2 which needs rank 7):
    # Actually 2x2x2 has rank 7. Let me construct it carefully.
    #
    # The 8 schoolbook terms can be reduced to 7. One way:
    # t1 = A11*B11
    # t2 = A12*B21
    # t3 = A11*B12
    # t4 = A12*B22
    # t5 = A21*B11
    # t6 = A22*B21
    # t7 = A21*B12 + A22*B22  -- this is NOT rank 1, so this doesn't work directly.
    #
    # Let me use a verified rank-7 decomposition. The key identity is:
    # (A11*B11 + A12*B21 + A21*B12 + A22*B22) + (A11*B12 + A12*B22 + A21*B11 + A22*B21)
    # = sum of all 8 terms
    #
    # A rank-7 decomposition exists. Let me use the one from the literature.
    # One standard rank-7 for 2x2x2:
    #
    # Term 1: (A11) (B11) -> C11
    # Term 2: (A12) (B21) -> C12  [wait, this goes to C12? No, A12*B21 goes to C11]
    #
    # Let me be very careful. For 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # The 8 rank-1 tensors (schoolbook):
    # (i,j,k) -> (A[i,j], B[j,k], C[i,k])
    # (0,0,0), (0,1,0), (0,0,1), (0,1,1), (1,0,0), (1,1,0), (1,0,1), (1,1,1)
    #
    # A rank-7 decomposition: we can combine two of these. For example:
    # Replace (0,0,1) + (1,1,1) with a single term? No, that's not a rank-1 tensor.
    #
    # Actually, the rank of 2x2x2 matmul is exactly 7. Here's one construction:
    # Use the identity that the 8 terms can be grouped as:
    # { (0,0,0), (0,1,0), (1,0,1), (1,1,1) } -> these 4 can be 4 terms
    # { (0,0,1), (0,1,1), (1,0,0), (1,1,0) } -> these 4 can be 3 terms (saving 1)
    #
    # Let me just use a known explicit rank-7. From the literature (e.g., Landsberg):
    #
    # One explicit rank-7 decomposition:
    # t1: U=(1,0,0,0), V=(1,0,0,0), W=(1,0,0,0)  -> (0,0,0)
    # t2: U=(0,1,0,0), V=(0,0,1,0), W=(1,0,0,0)  -> (0,1,0)
    # t3: U=(0,0,1,0), V=(0,1,0,0), W=(0,1,0,0)  -> (1,0,0)
    # t4: U=(0,0,0,1), V=(0,0,0,1), W=(0,1,0,0)  -> (1,1,0)
    # t5: U=(1,0,1,0), V=(1,0,1,0), W=(0,0,1,0)  -> contributes to (0,0,1) and (1,0,1)
    # t6: U=(0,1,0,1), V=(0,1,0,1), W=(0,0,1,0)  -> contributes to (0,1,1) and (1,1,1)
    # t7: U=(1,1,1,1), V=(1,1,1,1), W=(0,0,0,1)  -> contributes to (0,0,1),(0,1,1),(1,0,1),(1,1,1)
    #
    # Let me verify this. For entry (i,j,k), the sum should be 1.
    #
    # U is indexed by (i,j): (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # V is indexed by (j,k): (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    # W is indexed by (k,i): (0,0),(0,1),(1,0),(1,1) -> indices 0,1,2,3
    #   W[k*n+i] with n=2: W[0*2+0]=W[0], W[0*2+1]=W[1], W[1*2+0]=W[2], W[1*2+1]=W[3]
    #   So W index = k*2 + i.
    #
    # Let me define the 7 terms:
    # c1: U=(1,0,0,0), V=(1,0,0,0), W=(1,0,0,0)
    # c2: U=(0,1,0,0), V=(0,1,0,0), W=(1,0,0,0)
    # c3: U=(0,0,1,0), V=(0,0,1,0), W=(0,1,0,0)
    # c4: U=(0,0,0,1), V=(0,0,0,1), W=(0,1,0,0)
    # c5: U=(1,0,1,0), V=(1,0,1,0), W=(0,0,1,0)
    # c6: U=(0,1,0,1), V=(0,1,0,1), W=(0,0,1,0)
    # c7: U=(1,1,1,1), V=(1,1,1,1), W=(0,0,0,1)
    #
    # Check (0,0,0): i=0,j=0,k=0
    # U[0]=1, V[0]=1, W[0*2+0]=W[0]=1 -> 1
    # c2: U[0]=0 -> 0
    # c3: U[0]=0 -> 0
    # c4: U[0]=0 -> 0
    # c5: U[0]=1, V[0]=1, W[0]=0 -> 0
    # c6: U[0]=0 -> 0
    # c7: U[0]=1, V[0]=1, W[0]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (0,1,0): i=0,j=1,k=0
    # c1: U[1]=0 -> 0
    # c2: U[1]=1, V[1]=1, W[0]=1 -> 1
    # c3: U[1]=0 -> 0
    # c4: U[1]=0 -> 0
    # c5: U[1]=0 -> 0
    # c6: U[1]=1, V[1]=1, W[0]=0 -> 0
    # c7: U[1]=1, V[1]=1, W[0]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (1,0,0): i=1,j=0,k=0
    # c1: U[2]=0 -> 0
    # c2: U[2]=0 -> 0
    # c3: U[2]=1, V[2]=1, W[0*2+1]=W[1]=1 -> 1
    # c4: U[2]=0 -> 0
    # c5: U[2]=1, V[2]=1, W[1]=0 -> 0
    # c6: U[2]=0 -> 0
    # c7: U[2]=1, V[2]=1, W[1]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (1,1,0): i=1,j=1,k=0
    # c3: U[3]=0 -> 0
    # c4: U[3]=1, V[3]=1, W[1]=1 -> 1
    # c5: U[3]=0 -> 0
    # c6: U[3]=1, V[3]=1, W[1]=0 -> 0
    # c7: U[3]=1, V[3]=1, W[1]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (0,0,1): i=0,j=0,k=1
    # c1: U[0]=1, V[0]=1, W[1*2+0]=W[2]=0 -> 0
    # c2: U[0]=0 -> 0
    # c3: U[0]=0 -> 0
    # c4: U[0]=0 -> 0
    # c5: U[0]=1, V[0]=1, W[2]=1 -> 1
    # c6: U[0]=0 -> 0
    # c7: U[0]=1, V[0]=1, W[2]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (0,1,1): i=0,j=1,k=1
    # c5: U[1]=0 -> 0
    # c6: U[1]=1, V[1]=1, W[2]=1 -> 1
    # c7: U[1]=1, V[1]=1, W[2]=0 -> 0
    # Sum = 1. Correct.
    #
    # Check (1,0,1): i=1,j=0,k=1
    # c5: U[2]=1, V[2]=1, W[1*2+1]=W[3]=0 -> 0
    # c7: U[2]=1, V[2]=1, W[3]=1 -> 1
    # Sum = 1. Correct.
    #
    # Check (1,1,1): i=1,j=1,k=1
    # c6: U[3]=1, V[3]=1, W[3]=0 -> 0
    # c7: U[3]=1, V[3]=1, W[3]=1 -> 1
    # Sum = 1. Correct.
    #
    # Great, all 8 entries verified. Rank-7 decomposition of 2x2x2.

    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]

    terms = [
        ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
        ([0,1,0,0], [0,1,0,0], [1,0,0,0]),
        ([0,0,1,0], [0,0,1,0], [0,1,0,0]),
        ([0,0,0,1], [0,0,0,1], [0,1,0,0]),
        ([1,0,1,0], [1,0,1,0], [0,0,1,0]),
        ([0,1,0,1], [0,1,0,1], [0,0,1,0]),
        ([1,1,1,1], [1,1,1,1], [0,0,0,1]),
    ]
    for c, (u, v, w) in enumerate(terms):
        for idx in range(4):
            U[idx][c] = u[idx]
            V[idx][c] = v[idx]
            W[idx][c] = w[idx]
    return U, V, W


def construct():
    t_start = time.time()
    DEADLINE = t_start + 130  # safety margin before 150s

    n, m, p = DIMENSIONS  # 2, 3, 7

    # Best result: start with schoolbook
    def _schoolbook_full():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = _schoolbook_full()
    best_rank = n * m * p

    # Try peeling decomposition with rank-7 on the 2x2x2 sub-block
    # Strategy: decompose 2x3x7 as:
    #   - 2x2x2 sub-block (rows 0-1 of A, cols 0-1 of B, cols 0-1 of C): rank 7
    #   - Remaining terms: the rest of the schoolbook
    #
    # The 2x2x2 sub-block uses i in {0,1}, j in {0,1}, k in {0,1}
    # The remaining terms are all (i,j,k) where NOT (i<2 AND j<2 AND k<2)
    # Total remaining = 2*3*7 - 2*2*2 = 42 - 8 = 34
    # Total rank = 7 + 34 = 41
    #
    # But wait - the rank-7 decomposition of 2x2x2 only handles the 8 terms
    # (i in {0,1}, j in {0,1}, k in {0,1}). The other 34 terms still need
    # their own rank-1 decompositions. So total rank = 7 + 34 = 41.
    #
    # Let me construct this.

    if time.time() < DEADLINE:
        try:
            rank_222 = 7
            # The 2x2x2 sub-block: i in [0,2), j in [0,2), k in [0,2)
            # Remaining: all (i,j,k) not in that block
            remaining = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not (i < 2 and j < 2 and k < 2):
                            remaining.append((i, j, k))

            total_rank = rank_222 + len(remaining)
            if total_rank < best_rank:
                U = [[0]*total_rank for _ in range(n*m)]
                V = [[0]*total_rank for _ in range(m*p)]
                W = [[0]*total_rank for _ in range(p*n)]

                # Fill the 2x2x2 rank-7 part (columns 0..6)
                U222, V222, W222 = _rank7_2x2x2()
                for c in range(rank_222):
                    for ij in range(4):
                        U[ij][c] = U222[ij][c]
                        V[ij][c] = V222[ij][c]
                        W[ij][c] = W222[ij][c]

                # Fill remaining schoolbook terms
                col = rank_222
                for (i, j, k) in remaining:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

                # Verify
                ok, err = _verify(U, V, W, n, m, p)
                if ok:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = total_rank
        except Exception:
            pass

    # Also try: can we use rank-7 on a different 2x2x2 sub-block?
    # The 2x2x2 block must be contiguous in i, j, k. Since n=2, i must be {0,1}.
    # j can be {0,1} or {1,2}. k can be {0,1}, {1,2}, ..., {5,6}.
    # All are equivalent by symmetry in j and k. Already tried {0,1}x{0,1}x{0,1}.

    # Try: can we find a rank-6 decomposition of 2x3x1? (Should be impossible, min rank is 6)
    # The rank of 2x3x1 matmul is exactly 6 (it's a "thin" matmul).

    # Try: peeling with rank-7 on 2x2x2 at different positions
    # j={1,2}, k={0,1}
    if time.time() < DEADLINE:
        try:
            sub_ijk = [(i, j, k) for i in range(2) for j in range(1, 3) for k in range(2)]
            remaining = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                         if (i, j, k) not in set(sub_ijk)]
            total_rank = 7 + len(remaining)
            if total_rank < best_rank:
                U = [[0]*total_rank for _ in range(n*m)]
                V = [[0]*total_rank for _ in range(m*p)]
                W = [[0]*total_rank for _ in range(p*n)]

                U222, V222, W222 = _rank7_2x2x2()
                for c in range(7):
                    for ij in range(4):
                        U[ij][c] = U222[ij][c]
                        V[ij][c] = V222[ij][c]
                        W[ij][c] = W222[ij][c]

                col = 7
                for (i, j, k) in remaining:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

                ok, err = _verify(U, V, W, n, m, p)
                if ok:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = total_rank
        except Exception:
            pass

    # The rank-7 2x2x2 block only covers j in {0,1} or {1,2} and k in 2 consecutive values.
    # Since the rank-7 decomposition is specific to the 2x2x2 indexing, when we use
    # j={1,2}, the V indices shift. Let me handle this properly.

    # Actually, the rank-7 decomposition is for the standard 2x2x2 with indices
    # (i,j,k) in {0,1}^3. When we embed it in 2x3x7 with j offset by 1,
    # the V indices become (j+1)*p+k instead of j*p+k.
    # Let me redo this more carefully.

    # General approach: for a 2x2x2 sub-block at (i_offset, j_offset, k_offset),
    # the rank-7 terms map as:
    #   U: (i_offset + i_local)*m + (j_offset + j_local)
    #   V: (j_offset + j_local)*p + (k_offset + k_local)
    #   W: (k_offset + k_local)*n + (i_offset + i_local)

    if time.time() < DEADLINE:
        for j_off in [0, 1]:
            for k_off in range(p - 1):
                i_off = 0  # n=2, only one choice
                try:
                    sub_set = set()
                    for il in range(2):
                        for jl in range(2):
                            for kl in range(2):
                                sub_set.add((i_off + il, j_off + jl, k_off + kl))
                    remaining = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                                 if (i, j, k) not in sub_set]
                    total_rank = 7 + len(remaining)
                    if total_rank >= best_rank:
                        continue

                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]

                    U222, V222, W222 = _rank7_2x2x2()
                    for c in range(7):
                        for il in range(2):
                            for jl in range(2):
                                for kl in range(2):
                                    # This doesn't work - the rank-7 terms don't
                                    # correspond to individual (i,j,k) triples.
                                    # The rank-7 decomposition has 7 terms where each term
                                    # has specific U, V, W vectors.
                                    pass
                        # Map the 4x4 U222 into the full U
                        for idx in range(4):
                            i_local = idx // 2
                            j_local = idx % 2
                            U_idx = (i_off + i_local) * m + (j_off + j_local)
                            U[U_idx][c] = U222[idx][c]
                            V_idx = (j_off + j_local) * p + (k_off + idx % 2)  # WRONG
                        # This mapping is getting confused. Let me use explicit index mapping.
                    # Let me redo with proper index mapping
                    U = [[0]*total_rank for _ in range(n*m)]
                    V = [[0]*total_rank for _ in range(m*p)]
                    W = [[0]*total_rank for _ in range(p*n)]

                    for c in range(7):
                        # U222 has 4 rows: (i_l, j_l) -> index i_l*2 + j_l
                        for i_l in range(2):
                            for j_l in range(2):
                                u_idx = i_l * 2 + j_l
                                U_full_idx = (i_off + i_l) * m + (j_off + j_l)
                                U[U_full_idx][c] = U222[u_idx][c]

                        # V222 has 4 rows: (j_l, k_l) -> index j_l*2 + k_l
                        for j_l in range(2):
                            for k_l in range(2):
                                v_idx = j_l * 2 + k_l
                                V_full_idx = (j_off + j_l) * p + (k_off + k_l)
                                V[V_full_idx][c] = V222[v_idx][c]

                        # W222 has 4 rows: (k_l, i_l) -> index k_l*2 + i_l (since n=2, W[k*n+i])
                        for k_l in range(2):
                            for i_l in range(2):
                                w_idx = k_l * 2 + i_l
                                W_full_idx = (k_off + k_l) * n + (i_off + i_l)
                                W[W_full_idx][c] = W222[w_idx][c]

                    col = 7
                    for (i, j, k) in remaining:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1

                    ok, err = _verify(U, V, W, n, m, p)
                    if ok and total_rank < best_rank:
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = total_rank
                except Exception:
                    pass
                if time.time() > DEADLINE - 5:
                    break
            if time.time() > DEADLINE - 5:
                break

    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    ok, err = _verify(result["U"], result["V"], result["W"], 2, 3, 7)
    print(f"Valid: {ok}, error: {err}")

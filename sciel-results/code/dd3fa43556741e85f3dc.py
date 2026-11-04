"""Optimized block-partition 4x5x6 matrix multiplication decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Try multiple block partition strategies and pick the one with lowest rank.
    # Strategy: partition the three dimensions into blocks, then for each
    # (i_block, j_block, k_block) triple, use the minimal-rank decomposition
    # for that rectangular sub-problem.

    def get_block_rank(ni, mj, pk):
        """Return the rank for multiplying an ni x mj matrix by an mj x pk matrix."""
        # Schoolbook rank = ni * mj * pk
        schoolbook = ni * mj * pk
        # For 2x2x2, Strassen gives rank 7 vs schoolbook 8
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        # For 2x3x4: use Kronecker product of Strassen(2x2x2) and schoolbook(1x3x2)
        # rank = 7 * 6 = 42... but we can do better with a custom approach
        # Actually, let's use a more direct approach:
        # 2x3x4 can be seen as 2x(2+1)x(2+2)
        # Split into: 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x1x4 (rank 8) + 2x1x2 (rank 4)
        # But that's not right either. Let's just use schoolbook for now unless we find better.
        # For 2x2x3: split as 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11 vs schoolbook 12
        if ni == 2 and mj == 2 and pk == 3:
            return 11  # Strassen(2x2x2) + schoolbook(2x2x1)
        if ni == 2 and mj == 3 and pk == 2:
            return 11  # schoolbook(2x1x2) + Strassen(2x2x2)
        if ni == 3 and mj == 2 and pk == 2:
            return 11  # schoolbook(1x2x2) + Strassen(2x2x2)
        # For 1xk x kxl: rank = l (each output element needs one multiply)
        if ni == 1:
            return pk  # each of pk outputs needs 1 scalar multiply
        if pk == 1:
            return ni  # each of ni outputs needs 1 scalar multiply
        if mj == 1:
            return ni * pk  # outer product structure
        # For other cases, use schoolbook
        return schoolbook

    def build_decomposition(partition_i, partition_j, partition_k):
        """
        Build U, V, W for a given partition of dimensions.
        partition_i: list of (start, end) for A's rows (total n)
        partition_j: list of (start, end) for A's cols / B's rows (total m)
        partition_k: list of (start, end) for B's cols / C's cols (total p)
        """
        total_rank = 0
        for (i0, i1) in partition_i:
            for (j0, j1) in partition_j:
                for (k0, k1) in partition_k:
                    ni = i1 - i0
                    mj = j1 - j0
                    pk = k1 - k0
                    total_rank += get_block_rank(ni, mj, pk)

        rank = total_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        r_offset = 0

        for (i0, i1) in partition_i:
            for (j0, j1) in partition_j:
                for (k0, k1) in partition_k:
                    ni = i1 - i0
                    mj = j1 - j0
                    pk = k1 - k0
                    blk_rank = get_block_rank(ni, mj, pk)

                    # Add the block decomposition
                    if ni == 2 and mj == 2 and pk == 2:
                        # Strassen 2x2x2
                        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
                        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
                        def w_idx(k, i): return (k0 + k) * n + (i0 + i)

                        # M1 = (A00+A11)(B00+B11)
                        U[u_idx(0,0)][r_offset] += 1; U[u_idx(1,1)][r_offset] += 1
                        V[v_idx(0,0)][r_offset] += 1; V[v_idx(1,1)][r_offset] += 1
                        W[w_idx(0,0)][r_offset] += 1; W[w_idx(1,1)][r_offset] += 1
                        r_offset += 1
                        # M2 = (A10+A11)B00
                        U[u_idx(1,0)][r_offset] += 1; U[u_idx(1,1)][r_offset] += 1
                        V[v_idx(0,0)][r_offset] += 1
                        W[w_idx(0,1)][r_offset] += 1; W[w_idx(1,1)][r_offset] += -1
                        r_offset += 1
                        # M3 = A00(B01-B11)
                        U[u_idx(0,0)][r_offset] += 1
                        V[v_idx(0,1)][r_offset] += 1; V[v_idx(1,1)][r_offset] += -1
                        W[w_idx(1,0)][r_offset] += 1; W[w_idx(1,1)][r_offset] += 1
                        r_offset += 1
                        # M4 = A11(B10-B00)
                        U[u_idx(1,1)][r_offset] += 1
                        V[v_idx(1,0)][r_offset] += 1; V[v_idx(0,0)][r_offset] += -1
                        W[w_idx(0,0)][r_offset] += 1; W[w_idx(0,1)][r_offset] += 1
                        r_offset += 1
                        # M5 = (A00+A01)B11
                        U[u_idx(0,0)][r_offset] += 1; U[u_idx(0,1)][r_offset] += 1
                        V[v_idx(1,1)][r_offset] += 1
                        W[w_idx(0,0)][r_offset] += -1; W[w_idx(1,0)][r_offset] += 1
                        r_offset += 1
                        # M6 = (A10-A00)(B00+B01)
                        U[u_idx(1,0)][r_offset] += 1; U[u_idx(0,0)][r_offset] += -1
                        V[v_idx(0,0)][r_offset] += 1; V[v_idx(0,1)][r_offset] += 1
                        W[w_idx(1,1)][r_offset] += 1
                        r_offset += 1
                        # M7 = (A01-A11)(B10+B11)
                        U[u_idx(0,1)][r_offset] += 1; U[u_idx(1,1)][r_offset] += -1
                        V[v_idx(1,0)][r_offset] += 1; V[v_idx(1,1)][r_offset] += 1
                        W[w_idx(0,0)][r_offset] += 1
                        r_offset += 1
                    elif ni == 2 and mj == 2 and pk == 3:
                        # Split into 2x2x2 (Strassen) + 2x2x1 (schoolbook)
                        # First 2x2x2 part (cols 0-1 of B)
                        def u_idx223(i, j): return (i0 + i) * m + (j0 + j)
                        def v_idx223(j, k): return (j0 + j) * p + (k0 + k)
                        def w_idx223(k, i): return (k0 + k) * n + (i0 + i)

                        # Strassen for first 2x2x2
                        # M1 = (A00+A11)(B00+B11)
                        U[u_idx223(0,0)][r_offset] += 1; U[u_idx223(1,1)][r_offset] += 1
                        V[v_idx223(0,0)][r_offset] += 1; V[v_idx223(1,1)][r_offset] += 1
                        W[w_idx223(0,0)][r_offset] += 1; W[w_idx223(1,1)][r_offset] += 1
                        r_offset += 1
                        # M2 = (A10+A11)B00
                        U[u_idx223(1,0)][r_offset] += 1; U[u_idx223(1,1)][r_offset] += 1
                        V[v_idx223(0,0)][r_offset] += 1
                        W[w_idx223(0,1)][r_offset] += 1; W[w_idx223(1,1)][r_offset] += -1
                        r_offset += 1
                        # M3 = A00(B01-B11)
                        U[u_idx223(0,0)][r_offset] += 1
                        V[v_idx223(0,1)][r_offset] += 1; V[v_idx223(1,1)][r_offset] += -1
                        W[w_idx223(1,0)][r_offset] += 1; W[w_idx223(1,1)][r_offset] += 1
                        r_offset += 1
                        # M4 = A11(B10-B00)
                        U[u_idx223(1,1)][r_offset] += 1
                        V[v_idx223(1,0)][r_offset] += 1; V[v_idx223(0,0)][r_offset] += -1
                        W[w_idx223(0,0)][r_offset] += 1; W[w_idx223(0,1)][r_offset] += 1
                        r_offset += 1
                        # M5 = (A00+A01)B11
                        U[u_idx223(0,0)][r_offset] += 1; U[u_idx223(0,1)][r_offset] += 1
                        V[v_idx223(1,1)][r_offset] += 1
                        W[w_idx223(0,0)][r_offset] += -1; W[w_idx223(1,0)][r_offset] += 1
                        r_offset += 1
                        # M6 = (A10-A00)(B00+B01)
                        U[u_idx223(1,0)][r_offset] += 1; U[u_idx223(0,0)][r_offset] += -1
                        V[v_idx223(0,0)][r_offset] += 1; V[v_idx223(0,1)][r_offset] += 1
                        W[w_idx223(1,1)][r_offset] += 1
                        r_offset += 1
                        # M7 = (A01-A11)(B10+B11)
                        U[u_idx223(0,1)][r_offset] += 1; U[u_idx223(1,1)][r_offset] += -1
                        V[v_idx223(1,0)][r_offset] += 1; V[v_idx223(1,1)][r_offset] += 1
                        W[w_idx223(0,0)][r_offset] += 1
                        r_offset += 1

                        # Schoolbook for 2x2x1 (col 2 of B)
                        for ii in range(2):
                            for jj in range(2):
                                U[(i0+ii)*m + (j0+jj)][r_offset] = 1
                                V[(j0+jj)*p + (k0+2)][r_offset] = 1
                                W[(k0+2)*n + (i0+ii)][r_offset] = 1
                                r_offset += 1

                    elif ni == 2 and mj == 3 and pk == 2:
                        # Split into 2x1x2 (schoolbook) + 2x2x2 (Strassen)
                        def u_idx232(i, j): return (i0 + i) * m + (j0 + j)
                        def v_idx232(j, k): return (j0 + j) * p + (k0 + k)
                        def w_idx232(k, i): return (k0 + k) * n + (i0 + i)

                        # Schoolbook for 2x1x2 (row 0 of B)
                        for ii in range(2):
                            for kk in range(2):
                                U[(i0+ii)*m + (j0+0)][r_offset] = 1
                                V[(j0+0)*p + (k0+kk)][r_offset] = 1
                                W[(k0+kk)*n + (i0+ii)][r_offset] = 1
                                r_offset += 1

                        # Strassen for 2x2x2 (rows 1-2 of B)
                        # M1 = (A01+A12)(B10+B21)
                        U[u_idx232(0,1)][r_offset] += 1; U[u_idx232(1,2)][r_offset] += 1
                        V[v_idx232(1,0)][r_offset] += 1; V[v_idx232(2,1)][r_offset] += 1
                        W[w_idx232(0,0)][r_offset] += 1; W[w_idx232(1,1)][r_offset] += 1
                        r_offset += 1
                        # M2 = (A11+A12)B10
                        U[u_idx232(1,1)][r_offset] += 1; U[u_idx232(1,2)][r_offset] += 1
                        V[v_idx232(1,0)][r_offset] += 1
                        W[w_idx232(0,1)][r_offset] += 1; W[w_idx232(1,1)][r_offset] += -1
                        r_offset += 1
                        # M3 = A01(B21-B11)
                        U[u_idx232(0,1)][r_offset] += 1
                        V[v_idx232(2,1)][r_offset] += 1; V[v_idx232(1,1)][r_offset] += -1
                        W[w_idx232(1,0)][r_offset] += 1; W[w_idx232(1,1)][r_offset] += 1
                        r_offset += 1
                        # M4 = A12(B20-B10)
                        U[u_idx232(1,2)][r_offset] += 1
                        V[v_idx232(2,0)][r_offset] += 1; V[v_idx232(1,0)][r_offset] += -1
                        W[w_idx232(0,0)][r_offset] += 1; W[w_idx232(0,1)][r_offset] += 1
                        r_offset += 1
                        # M5 = (A01+A02)B21
                        U[u_idx232(0,1)][r_offset] += 1; U[u_idx232(0,2)][r_offset] += 1
                        V[v_idx232(2,1)][r_offset] += 1
                        W[w_idx232(0,0)][r_offset] += -1; W[w_idx232(1,0)][r_offset] += 1
                        r_offset += 1
                        # M6 = (A11-A01)(B10+B11)
                        U[u_idx232(1,1)][r_offset] += 1; U[u_idx232(0,1)][r_offset] += -1
                        V[v_idx232(1,0)][r_offset] += 1; V[v_idx232(1,1)][r_offset] += 1
                        W[w_idx232(1,1)][r_offset] += 1
                        r_offset += 1
                        # M7 = (A02-A12)(B20+B21)
                        U[u_idx232(0,2)][r_offset] += 1; U[u_idx232(1,2)][r_offset] += -1
                        V[v_idx232(2,0)][r_offset] += 1; V[v_idx232(2,1)][r_offset] += 1
                        W[w_idx232(0,0)][r_offset] += 1
                        r_offset += 1

                    elif ni == 3 and mj == 2 and pk == 2:
                        # Split into 1x2x2 (schoolbook) + 2x2x2 (Strassen)
                        def u_idx322(i, j): return (i0 + i) * m + (j0 + j)
                        def v_idx322(j, k): return (j0 + j) * p + (k0 + k)
                        def w_idx322(k, i): return (k0 + k) * n + (i0 + i)

                        # Schoolbook for 1x2x2 (row 0 of A)
                        for jj in range(2):
                            for kk in range(2):
                                U[(i0+0)*m + (j0+jj)][r_offset] = 1
                                V[(j0+jj)*p + (k0+kk)][r_offset] = 1
                                W[(k0+kk)*n + (i0+0)][r_offset] = 1
                                r_offset += 1

                        # Strassen for 2x2x2 (rows 1-2 of A)
                        # M1 = (A10+A21)(B00+B11)
                        U[u_idx322(1,0)][r_offset] += 1; U[u_idx322(2,1)][r_offset] += 1
                        V[v_idx322(0,0)][r_offset] += 1; V[v_idx322(1,1)][r_offset] += 1
                        W[w_idx322(0,1)][r_offset] += 1; W[w_idx322(1,2)][r_offset] += 1
                        r_offset += 1
                        # M2 = (A20+A21)B00
                        U[u_idx322(2,0)][r_offset] += 1; U[u_idx322(2,1)][r_offset] += 1
                        V[v_idx322(0,0)][r_offset] += 1
                        W[w_idx322(0,2)][r_offset] += 1; W[w_idx322(1,2)][r_offset] += -1
                        r_offset += 1
                        # M3 = A10(B01-B11)
                        U[u_idx322(1,0)][r_offset] += 1
                        V[v_idx322(0,1)][r_offset] += 1; V[v_idx322(1,1)][r_offset] += -1
                        W[w_idx322(1,1)][r_offset] += 1; W[w_idx322(1,2)][r_offset] += 1
                        r_offset += 1
                        # M4 = A21(B10-B00)
                        U[u_idx322(2,1)][r_offset] += 1
                        V[v_idx322(1,0)][r_offset] += 1; V[v_idx322(0,0)][r_offset] += -1
                        W[w_idx322(0,1)][r_offset] += 1; W[w_idx322(0,2)][r_offset] += 1
                        r_offset += 1
                        # M5 = (A10+A11)B11
                        U[u_idx322(1,0)][r_offset] += 1; U[u_idx322(1,1)][r_offset] += 1
                        V[v_idx322(1,1)][r_offset] += 1
                        W[w_idx322(0,1)][r_offset] += -1; W[w_idx322(1,1)][r_offset] += 1
                        r_offset += 1
                        # M6 = (A20-A10)(B00+B01)
                        U[u_idx322(2,0)][r_offset] += 1; U[u_idx322(1,0)][r_offset] += -1
                        V[v_idx322(0,0)][r_offset] += 1; V[v_idx322(0,1)][r_offset] += 1
                        W[w_idx322(1,2)][r_offset] += 1
                        r_offset += 1
                        # M7 = (A11-A21)(B10+B11)
                        U[u_idx322(1,1)][r_offset] += 1; U[u_idx322(2,1)][r_offset] += -1
                        V[v_idx322(1,0)][r_offset] += 1; V[v_idx322(1,1)][r_offset] += 1
                        W[w_idx322(0,1)][r_offset] += 1
                        r_offset += 1

                    else:
                        # Schoolbook: each (i,j,k) triple gets one rank-1 term
                        for ii in range(ni):
                            for jj in range(mj):
                                for kk in range(pk):
                                    U[(i0+ii)*m + (j0+jj)][r_offset] = 1
                                    V[(j0+jj)*p + (k0+kk)][r_offset] = 1
                                    W[(k0+kk)*n + (i0+ii)][r_offset] = 1
                                    r_offset += 1

        if r_offset != rank:
            raise ValueError(f"Rank mismatch: expected {rank}, got {r_offset}")

        return U, V, W, rank

    # Generate candidate partitions
    def gen_partitions(dim):
        """Generate all ways to partition a dimension of size dim into consecutive blocks."""
        results = []
        def helper(start, remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            for size in range(1, remaining + 1):
                current.append((start, start + size))
                helper(start + size, remaining - size, current)
                current.pop()
        helper(0, dim, [])
        return results

    # For efficiency, limit partition complexity
    # n=4: partitions like [4], [2,2], [1,3], [3,1], [1,1,2], etc.
    # m=5: partitions like [5], [2,3], [3,2], [1,4], [4,1], [1,1,3], etc.
    # p=6: partitions like [6], [2,4], [4,2], [3,3], [2,2,2], etc.

    # Limit to partitions with at most 3 blocks per dimension for search efficiency
    def gen_limited_partitions(dim, max_blocks=3):
        results = []
        def helper(start, remaining, current):
            if remaining == 0:
                if len(current) <= max_blocks:
                    results.append(list(current))
                return
            for size in range(1, remaining + 1):
                current.append((start, start + size))
                helper(start + size, remaining - size, current)
                current.pop()
        helper(0, dim, [])
        return results

    pi_candidates = gen_limited_partitions(n, 3)
    pj_candidates = gen_limited_partitions(m, 3)
    pk_candidates = gen_limited_partitions(p, 3)

    best_rank = float('inf')
    best_result = None

    for pi in pi_candidates:
        for pj in pj_candidates:
            for pk in pk_candidates:
                # Quick rank estimate
                est_rank = 0
                for (i0, i1) in pi:
                    for (j0, j1) in pj:
                        for (k0, k1) in pk:
                            est_rank += get_block_rank(i1-i0, j1-j0, k1-k0)
                if est_rank >= best_rank:
                    continue
                if time.time() - start_time > DEADLINE:
                    break

                try:
                    U, V, W, rank = build_decomposition(pi, pj, pk)
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W, rank)
                except Exception:
                    continue
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break

    if best_result is None:
        # Fallback: pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    U, V, W, rank = best_result

    # Verify
    if time.time() - start_time < DEADLINE - 5:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        A_test = np.random.randint(-2, 3, (n, m))
        B_test = np.random.randint(-2, 3, (m, p))
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r in range(rank):
            A_part = U_np[:, r] @ A_test.reshape(-1) if False else None
            # Manual computation
            a_vec = np.zeros(n)
            b_vec = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    a_vec[i] += U_np[i*m+j, r] * A_test[i, j]
            for k in range(p):
                for j in range(m):
                    b_vec[k] += V_np[j*p+k, r] * B_test[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_np[k*n+i, r] * a_vec[i] * b_vec[k]

        if not np.allclose(C_computed, C_expected):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][r] = 1
                        V[j*p+k][r] = 1
                        W[k*n+i][r] = 1
                        r += 1

    return {"U": U, "V": V, "W": W}

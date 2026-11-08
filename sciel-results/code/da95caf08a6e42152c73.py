"""Matrix multiplication 3x5x6: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term: (u_pat, v_pat, w_pat) where patterns map local indices to values.
        U: (i,j) -> val, V: (j,k) -> val, W: (k,i) -> val
        """
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # M2 = (A10+A11)B00 -> C10+1, C11-1
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}))
        # M3 = A00(B01-B11) -> C01+1, C11+1
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}))
        # M4 = A11(B10-B00) -> C00+1, C10+1
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}))
        # M5 = (A00+A01)B11 -> C00-1, C01+1
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}))
        # M6 = (A10-A00)(B00+B01) -> C11+1
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # M7 = (A01-A11)(B10+B11) -> C00+1
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    STRASSEN = strassen_2x2x2_terms()

    def verify(U, V, W, rank):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_decomposition(i_groups, j_groups, k_groups, use_strassen=True):
        """
        Build decomposition from partition.
        i_groups: list of lists of row indices
        j_groups: list of lists of col indices  
        k_groups: list of lists of output col indices
        Returns (U, V, W, rank) or None if invalid.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if use_strassen and si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if use_strassen and si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in STRASSEN:
                    for (li, lj), val_u in u_pat.items():
                        U[ig[li] * m + jg[lj]][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        V[jg[lj] * p + kg[lk]][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        W[kg[lk] * n + ig[li]][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            U[ig[li] * m + jg[lj]][col] = 1
                            V[jg[lj] * p + kg[lk]][col] = 1
                            W[kg[lk] * n + ig[li]][col] = 1
                            col += 1

        return U, V, W, total_rank

    def gen_set_partitions(size):
        """Generate all set partitions of {0, 1, ..., size-1} as lists of frozensets."""
        if size == 0:
            yield []
            return
        if size == 1:
            yield [frozenset([0])]
            return
        # Recursive: pick a partition of {0..size-2}, then decide where to put (size-1)
        for part in gen_set_partitions(size - 1):
            # Option 1: put (size-1) in a new block
            yield part + [frozenset([size - 1])]
            # Option 2: put (size-1) in each existing block
            for i in range(len(part)):
                new_part = list(part)
                new_part[i] = new_part[i] | frozenset([size - 1])
                yield new_part

    def find_best_partition():
        """
        Search over all set partitions of rows, cols, and output cols.
        For each combination, compute the rank and pick the minimum.
        """
        # Generate all set partitions and convert frozensets to sorted lists
        i_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(n)]
        j_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(m)]
        k_parts = [tuple(sorted(g) for g in part) for part in gen_set_partitions(p)]

        best_rank = n * m * p
        best_result = None

        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    # Quick rank estimate without full build
                    est_rank = 0
                    for ig in ip:
                        for jg in jp:
                            for kg in kp:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    est_rank += 7
                                else:
                                    est_rank += si * sj * sk
                    if est_rank >= best_rank:
                        continue
                    # Convert tuples to lists for build_decomposition
                    ip_list = [list(ig) for ig in ip]
                    jp_list = [list(jg) for jg in jp]
                    kp_list = [list(kg) for kg in kp]
                    U, V, W, rank = build_decomposition(ip_list, jp_list, kp_list, use_strassen=True)
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)

        return best_result, best_rank

    def schoolbook():
        """Fallback schoolbook decomposition."""
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

    # Start with schoolbook
    best_result = schoolbook()
    best_rank = n * m * p

    # Search over all partitions
    if time.time() < deadline - 20:
        found, found_rank = find_best_partition()
        if found is not None and found_rank < best_rank:
            best_result = {"U": found[0], "V": found[1], "W": found[2]}
            best_rank = found_rank

    # Verify the best result
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        # Fallback to schoolbook
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify_numpy(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u_pattern, v_pattern, w_pattern) where patterns are
        dicts mapping local index tuples to integer values.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def generate_all_partitions(dim):
        """Generate all ways to partition indices 0..dim-1 into contiguous groups of size 1 or 2."""
        if dim <= 0:
            return [[]]
        if dim == 1:
            return [[[0]]]
        if dim == 2:
            return [[[0, 1]], [[0], [1]]]
        if dim == 3:
            return [
                [[0, 1], [2]],
                [[0], [1, 2]],
                [[0], [1], [2]],
            ]
        if dim == 4:
            results = []
            def gen(idx, current):
                if idx >= dim:
                    results.append(current[:])
                    return
                # Try group of size 1
                current.append([idx])
                gen(idx+1, current)
                current.pop()
                # Try group of size 2
                if idx+1 < dim:
                    current.append([idx, idx+1])
                    gen(idx+2, current)
                    current.pop()
            gen(0, [])
            return results
        if dim == 5:
            results = []
            def gen(idx, current):
                if idx >= dim:
                    results.append(current[:])
                    return
                # Try group of size 1
                current.append([idx])
                gen(idx+1, current)
                current.pop()
                # Try group of size 2
                if idx+1 < dim:
                    current.append([idx, idx+1])
                    gen(idx+2, current)
                    current.pop()
            gen(0, [])
            return results
        if dim == 6:
            results = []
            def gen(idx, current):
                if idx >= dim:
                    results.append(current[:])
                    return
                # Try group of size 1
                current.append([idx])
                gen(idx+1, current)
                current.pop()
                # Try group of size 2
                if idx+1 < dim:
                    current.append([idx, idx+1])
                    gen(idx+2, current)
                    current.pop()
            gen(0, [])
            return results
        # Fallback for larger dims
        return [[[i] for i in range(dim)]]

    def compute_rank_for_partition(i_groups, j_groups, k_groups):
        """Compute the total rank for a given partition using Strassen for 2x2x2 blocks."""
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total_rank += 7
                    else:
                        total_rank += si * sj * sk
        return total_rank

    def build_decomposition_for_partition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = compute_rank_for_partition(i_groups, j_groups, k_groups)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen for this 2x2x2 block
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                U[gi * m + gj][col] = val_u
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                V[gj * p + gk][col] = val_v
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                W[gk * n + gi][col] = val_w
                            col += 1
                    else:
                        # Schoolbook for this sub-block
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Generate all possible partitions
    i_partitions = generate_all_partitions(n)  # n=3
    j_partitions = generate_all_partitions(m)  # m=5
    k_partitions = generate_all_partitions(p)  # p=6

    # Find the minimum rank partition
    min_rank = best_rank
    best_partition = None
    
    for i_g in i_partitions:
        for j_g in j_partitions:
            for k_g in k_partitions:
                r = compute_rank_for_partition(i_g, j_g, k_g)
                if r < min_rank:
                    min_rank = r
                    best_partition = (i_g, j_g, k_g)
    
    # Build the best partition
    if best_partition is not None:
        i_g, j_g, k_g = best_partition
        result = build_decomposition_for_partition(i_g, j_g, k_g)
        rank = result.pop("rank")
        
        # Verify
        if verify_numpy(result["U"], result["V"], result["W"], rank):
            best_result = result
            best_rank = rank
        else:
            # Fallback to schoolbook
            pass
    else:
        # Use schoolbook
        pass

    return best_result
# EVOLVE-BLOCK-END

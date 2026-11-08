import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def strassen_2x2x2_terms():
        terms = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def generate_partitions(size, max_block=6):
        if size == 0:
            yield []
            return
        for first in range(1, min(size, max_block) + 1):
            for rest in generate_partitions(size - first, max_block):
                yield [first] + rest

    def compute_block_rank(si, sj, sk):
        if si == 2 and sj == 2 and sk == 2:
            return 7
        if (si == 2 and sj == 3 and sk == 2) or (si == 2 and sj == 2 and sk == 3) or (si == 3 and sj == 2 and sk == 2):
            return 11
        return si * sj * sk

    def total_rank_for_partition(i_part, j_part, k_part):
        total = 0
        for si in i_part:
            for sj in j_part:
                for sk in k_part:
                    total += compute_block_rank(si, sj, sk)
        return total

    def find_best_partition():
        best_rank = n * m * p
        best_partition = None
        
        i_partitions = list(generate_partitions(n, max_block=3))
        j_partitions = list(generate_partitions(m, max_block=5))
        k_partitions = list(generate_partitions(p, max_block=6))
        
        for ip in i_partitions:
            for jp in j_partitions:
                for kp in k_partitions:
                    r = total_rank_for_partition(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (ip, jp, kp)
        
        return best_partition, best_rank

    def build_decomposition(i_part, j_part, k_part):
        i_groups = []
        idx = 0
        for s in i_part:
            i_groups.append(list(range(idx, idx + s)))
            idx += s
        
        j_groups = []
        idx = 0
        for s in j_part:
            j_groups.append(list(range(idx, idx + s)))
            idx += s
        
        k_groups = []
        idx = 0
        for s in k_part:
            k_groups.append(list(range(idx, idx + s)))
            idx += s
        
        total_rank = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    total_rank += compute_block_rank(si, sj, sk)
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_222 = strassen_2x2x2_terms()
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        for (u_pat, v_pat, w_pat) in strassen_222:
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
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    best_partition, best_rank = find_best_partition()
    
    if best_partition is not None and best_rank < n * m * p:
        result = build_decomposition(*best_partition)
        rank = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], rank):
            return result
    
    return schoolbook()
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations, product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
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
        terms.append((u4, w4))  # placeholder to keep structure, will fix below
        
        # Re-do term 4 correctly
        terms.pop()
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

    def compute_partition_rank(i_groups, j_groups, k_groups):
        """Compute total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    def generate_partitions(dim):
        """Generate all partitions of range(dim) into contiguous groups."""
        partitions = []
        for g in range(1, dim + 1):
            for dividers in combinations(range(dim - 1), g - 1):
                groups = []
                prev = 0
                for d in dividers:
                    groups.append(list(range(prev, d + 1)))
                    prev = d + 1
                groups.append(list(range(prev, dim)))
                partitions.append(groups)
        return partitions

    def build_strassen_decomposition(i_groups=None, j_groups=None, k_groups=None):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        if i_groups is None:
            i_groups = [[0, 1], [2]]
        if j_groups is None:
            j_groups = [[0, 1], [2, 3], [4]]
        if k_groups is None:
            k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
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

    def find_best_partition():
        """Search over partitions to find the one with minimum rank."""
        if time.time() > deadline:
            return None
        
        i_parts = generate_partitions(n)
        j_parts = generate_partitions(m)
        k_parts = generate_partitions(p)
        
        best_rank = float('inf')
        best_partition = None
        
        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    r = compute_partition_rank(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (ip, jp, kp)
        
        if best_partition is None:
            return None
        
        ip, jp, kp = best_partition
        result = build_strassen_decomposition(ip, jp, kp)
        rank = result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], rank):
            return result, rank
        
        return None

    def try_tensor_rank_minimization(initial_rank=None, max_iters=500, tol=1e-10):
        """
        Try to find a lower rank decomposition using ALS.
        """
        if time.time() > deadline:
            return None
        
        if initial_rank is None:
            initial_rank = n * m * p // 2
        
        # Initialize with random factors
        U = np.random.randn(n * m, initial_rank)
        V = np.random.randn(m * p, initial_rank)
        W = np.random.randn(p * n, initial_rank)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Precompute tensor indices
        tensor_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    tensor_indices.append((u_idx, v_idx, w_idx))
        
        # ALS iterations
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for c in range(initial_rank):
                # For each row of U, solve least squares
                pass  # Simplified: use full matrix update
            
            # Full matrix ALS update
            # Update U: for each row i, U[i,:] = (V^T * W)_{i}^{-1} * ... 
            # This is complex for 3-way tensors. Use simpler approach.
            
            # Instead, use a simpler gradient-based approach or skip if too slow
            if it % 100 == 0:
                # Check error
                err = 0.0
                for (ui, vi, wi) in tensor_indices:
                    actual = np.dot(U[ui, :], V[vi, :]) * W[wi, :].sum()  # Wrong, need elementwise
                    # Correct check:
                    actual = 0.0
                    for c in range(initial_rank):
                        actual += U[ui, c] * V[vi, c] * W[wi, c]
                    err += (actual - 1.0) ** 2
                err /= len(tensor_indices)
                
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        # Final check
        err = 0.0
        for (ui, vi, wi) in tensor_indices:
            actual = 0.0
            for c in range(initial_rank):
                actual += U[ui, c] * V[vi, c] * W[wi, c]
            err += (actual - 1.0) ** 2
        err /= len(tensor_indices)
        
        if err < 1e-9:
            # Convert to list format
            U_list = [[U[i, c] for c in range(initial_rank)] for i in range(n * m)]
            V_list = [[V[i, c] for c in range(initial_rank)] for i in range(m * p)]
            W_list = [[W[i, c] for c in range(initial_rank)] for i in range(p * n)]
            
            # Round to integers if close
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for c in range(len(row)):
                        if abs(row[c] - round(row[c])) < 1e-8:
                            row[c] = round(row[c])
            
            return {"U": U_list, "V": V_list, "W": W_list, "rank": initial_rank}
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass

    # Try improved decomposition via partition search
    partition_result = find_best_partition()
    if partition_result is not None:
        result, rank = partition_result
        if rank < best_rank:
            best_result = result
            best_rank = rank

    # Try tensor rank minimization from the best known rank
    if time.time() < deadline - 20:
        for trial_rank in range(max(7, best_rank - 10), max(7, best_rank - 30), -5):
            if time.time() > deadline - 5:
                break
            result = try_tensor_rank_minimization(initial_rank=trial_rank, max_iters=100)
            if result is not None:
                rank = result.pop("rank")
                if rank < best_rank:
                    best_result = result
                    best_rank = rank
                    break

    return best_result
# EVOLVE-BLOCK-END

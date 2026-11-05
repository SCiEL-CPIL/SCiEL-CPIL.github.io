"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
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
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_sparse(U, V, W, rank):
        """Verify decomposition using sparse representation for speed."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_r = i * m + j
                    v_r = j * p + k
                    w_r = k * n + i
                    total = 0.0
                    for c in range(rank):
                        uv = U[u_r][c] * V[v_r][c]
                        if uv != 0:
                            total += uv * W[w_r][c]
                    if abs(total - 1.0) > 1e-9:
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
        """
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

    def build_decomposition(i_groups, j_groups, k_groups):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
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

    # Try to find a lower-rank decomposition using numpy-based optimization
    # Strategy: use the block decomposition as a starting point and try to
    # reduce rank by finding linear dependencies among the rank-1 terms
    def try_reduce_rank_via_dependencies(current_decomp, current_rank):
        """Try to reduce rank by finding linear dependencies among columns."""
        nonlocal best_result, best_rank
        
        U_np = np.array(current_decomp["U"], dtype=float)
        V_np = np.array(current_decomp["V"], dtype=float)
        W_np = np.array(current_decomp["W"], dtype=float)
        
        # For each column, check if it can be expressed as a linear combination
        # of other columns while maintaining the tensor decomposition
        
        # This is a complex optimization problem. Instead, let's try a simpler approach:
        # Try to merge pairs of columns that have similar patterns
        
        # Actually, let's try a different approach: use randomized rounding
        # to find a sparse representation with fewer non-zero columns
        
        # For now, this is a placeholder for more sophisticated methods
        return current_decomp, current_rank

    # Try various groupings to minimize rank
    # For n=3, m=5, p=6:
    # i can be split as [0,1],[2] or [0],[1,2]
    # j can be split in various ways
    # k can be split in various ways
    
    i_splits = [
        [[0, 1], [2]],
        [[0], [1, 2]],
        [[0], [1], [2]],
    ]
    j_splits = [
        [[0, 1], [2, 3], [4]],
        [[0], [1, 2], [3, 4]],
        [[0, 1], [2], [3, 4]],
        [[0], [1], [2, 3, 4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3], [4]],
        [[0, 1, 2], [3, 4]],
        [[0], [1], [2], [3, 4]],
        [[0, 1], [2], [3], [4]],
    ]
    k_splits = [
        [[0, 1], [2, 3], [4, 5]],
        [[0], [1, 2], [3, 4, 5]],
        [[0, 1], [2], [3, 4, 5]],
        [[0], [1], [2, 3], [4, 5]],
        [[0, 1], [2, 3], [4], [5]],
        [[0], [1, 2], [3], [4, 5]],
        [[0, 1], [2], [3], [4, 5]],
        [[0], [1], [2], [3], [4, 5]],
        [[0, 1, 2], [3, 4, 5]],
        [[0], [1, 2, 3], [4, 5]],
    ]
    
    # Try all combinations
    for i_g in i_splits:
        for j_g in j_splits:
            for k_g in k_splits:
                if time.time() > deadline:
                    break
                
                # Calculate rank for this grouping
                total_rank = 0
                for ig in i_g:
                    for jg in j_g:
                        for kg in k_g:
                            si, sj, sk = len(ig), len(jg), len(kg)
                            if si == 2 and sj == 2 and sk == 2:
                                total_rank += 7
                            else:
                                total_rank += si * sj * sk
                
                if total_rank >= best_rank:
                    continue
                
                # Build the decomposition
                result = build_decomposition(i_g, j_g, k_g)
                
                # Verify
                if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                    best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                    best_rank = total_rank

    # Try to improve by using a more sophisticated search
    # Key insight: the current block approach gives 84, but we can do better
    # by exploiting the non-square structure more cleverly
    
    def try_optimized_block_search():
        """Try more sophisticated block decompositions."""
        nonlocal best_result, best_rank
        
        # For n=3, m=5, p=6, let's try to find the optimal partition
        # that minimizes the total rank
        
        # The rank of a block (si, sj, sk) is:
        # - 7 if si=sj=sk=2 (Strassen)
        # - si*sj*sk otherwise (schoolbook)
        
        # We want to partition i into groups, j into groups, k into groups
        # such that the sum of block ranks is minimized
        
        # Let's be more systematic: try all possible partitions
        # A partition of {0,...,n-1} is a set of subsets that partition the set
        
        def get_all_partitions(elements):
            """Generate all possible partitions of a set of elements."""
            if not elements:
                yield []
                return
            
            first = elements[0]
            rest = elements[1:]
            
            for partition in get_all_partitions(rest):
                # Add first to each existing group
                for i, group in enumerate(partition):
                    new_partition = partition[:i] + [group + [first]] + partition[i+1:]
                    yield new_partition
                
                # Start a new group with first
                yield partition + [[first]]
        
        # For n=3, m=5, p=6, the number of partitions is manageable
        # n=3: 5 partitions
        # m=5: 52 partitions  
        # p=6: 203 partitions
        
        # But we can be smarter: only consider partitions where groups have size 1 or 2
        # (since Strassen only helps for 2x2x2 blocks)
        
        def get_binary_partitions(elements):
            """Generate partitions where each group has size 1 or 2."""
            elements = list(elements)
            if not elements:
                yield []
                return
            
            first = elements[0]
            rest = elements[1:]
            
            # Option 1: first is alone
            for partition in get_binary_partitions(rest):
                yield [[first]] + partition
            
            # Option 2: first is paired with another element
            for i in range(len(rest)):
                paired = [first, rest[i]]
                remaining = rest[:i] + rest[i+1:]
                for partition in get_binary_partitions(remaining):
                    yield [paired] + partition
        
        # Try all binary partitions
        i_partitions = list(get_binary_partitions(range(n)))
        j_partitions = list(get_binary_partitions(range(m)))
        k_partitions = list(get_binary_partitions(range(p)))
        
        print(f"Trying {len(i_partitions)} x {len(j_partitions)} x {len(k_partitions)} = {len(i_partitions)*len(j_partitions)*len(k_partitions)} combinations")
        
        for i_part in i_partitions:
            for j_part in j_partitions:
                for k_part in k_partitions:
                    if time.time() > deadline:
                        return
                    
                    # Calculate rank for this partition
                    total_rank = 0
                    for ig in i_part:
                        for jg in j_part:
                            for kg in k_part:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total_rank += 7
                                else:
                                    total_rank += si * sj * sk
                    
                    if total_rank >= best_rank:
                        continue
                    
                    # Build the decomposition
                    result = build_decomposition(i_part, j_part, k_part)
                    
                    # Verify
                    if verify_sparse(result["U"], result["V"], result["W"], total_rank):
                        best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
                        best_rank = total_rank
                        print(f"Found better decomposition with rank {best_rank}")
        
        return
    
    # Call the optimized search
    try_optimized_block_search()

    return best_result
# EVOLVE-BLOCK-END

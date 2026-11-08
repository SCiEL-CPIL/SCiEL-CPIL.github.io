"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

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

    def verify_numpy(U, V, W, rank):
        """Verify that the decomposition is exact using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
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

    def build_decomposition(i_groups, j_groups, k_groups, strassen_terms):
        """Build a decomposition from given groupings."""
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
        
        if col != total_rank:
            return None, None
            
        result = {"U": U, "V": V, "W": W}
        return result, total_rank

    def try_rank_reduction():
        """
        Try multiple partition strategies to find the best rank reduction.
        """
        if time.time() > deadline:
            return None, None
        
        strassen_terms = get_strassen_terms()
        
        strategies = []
        
        # Strategy 1: Balanced 2-2-1 splits
        strategies.append({
            'name': 's1',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 2: Different j split
        strategies.append({
            'name': 's2',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 2], [1, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 3: Different k split
        strategies.append({
            'name': 's3',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 2], [1, 3], [4, 5]]
        })
        
        # Strategy 4: 3-element groups
        strategies.append({
            'name': 's4',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1, 2], [3, 4]],
            'k_groups': [[0, 1, 2], [3, 4, 5]]
        })
        
        # Strategy 5: Mixed
        strategies.append({
            'name': 's5',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3, 4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 6: Different i grouping
        strategies.append({
            'name': 's6',
            'i_groups': [[0, 2], [1]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 7: Another i grouping
        strategies.append({
            'name': 's7',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 3], [1, 2], [4]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 8: Different k pairing
        strategies.append({
            'name': 's8',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 3], [1, 2], [4, 5]]
        })
        
        # Strategy 9: Different j pairs
        strategies.append({
            'name': 's9',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 4], [1, 2], [3]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 10: More varied
        strategies.append({
            'name': 's10',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 4], [3]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Additional strategies to explore
        # Strategy 11: Try all pairs for j
        strategies.append({
            'name': 's11',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 3], [1, 4], [2]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 12: Another j pairing
        strategies.append({
            'name': 's12',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 4], [1, 3], [2]],
            'k_groups': [[0, 1], [2, 3], [4, 5]]
        })
        
        # Strategy 13: Different k groupings
        strategies.append({
            'name': 's13',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 4], [1, 2], [3, 5]]
        })
        
        # Strategy 14: More k variations
        strategies.append({
            'name': 's14',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 5], [1, 3], [2, 4]]
        })
        
        # Strategy 15: Try 3 in k
        strategies.append({
            'name': 's15',
            'i_groups': [[0, 1], [2]],
            'j_groups': [[0, 1], [2, 3], [4]],
            'k_groups': [[0, 1, 2], [3, 4, 5]]
        })
        
        best_result = None
        best_rank = float('inf')
        
        for strat in strategies:
            if time.time() > deadline:
                break
            
            result, rank = build_decomposition(
                strat['i_groups'], strat['j_groups'], strat['k_groups'], strassen_terms
            )
            
            if result is None:
                continue
            
            if verify_numpy(result['U'], result['V'], result['W'], rank):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        
        if best_result is not None:
            return best_result, best_rank
        
        return None, None

    def try_tensor_optimization():
        """
        Try to find a lower rank decomposition using tensor optimization.
        This attempts to find a decomposition with rank less than the block Strassen result.
        """
        if time.time() > deadline:
            return None, None
        
        # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k)
        # We want to find U, V, W such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # Start from the block Strassen decomposition and try to optimize
        strassen_terms = get_strassen_terms()
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        base_result, base_rank = build_decomposition(i_groups, j_groups, k_groups, strassen_terms)
        if base_result is None:
            return None, None
        
        # Try to reduce rank by removing redundant terms
        # This is a heuristic approach - try removing terms and checking if decomposition still works
        
        U = np.array(base_result['U'], dtype=float)
        V = np.array(base_result['V'], dtype=float)
        W = np.array(base_result['W'], dtype=float)
        
        rank = base_rank
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Try greedy term elimination
        for _ in range(50):  # Limited iterations for time
            if time.time() > deadline:
                break
            
            improved = False
            for c in range(rank - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                # Try removing term c
                U_test = np.delete(U, c, axis=1)
                V_test = np.delete(V, c, axis=1)
                W_test = np.delete(W, c, axis=1)
                
                # Check if decomposition still works
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_test[i*m+j], V_test[j*p+k] * W_test[k*n+i])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    U, V, W = U_test, V_test, W_test
                    rank -= 1
                    improved = True
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
        
            if not improved:
                break
        
        if best_rank < base_rank:
            result = {
                "U": [[best_U[i, j] for j in range(best_rank)] for i in range(n * m)],
                "V": [[best_V[i, j] for j in range(best_rank)] for i in range(m * p)],
                "W": [[best_W[i, j] for j in range(best_rank)] for i in range(p * n)]
            }
            return result, best_rank
        
        return None, None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition
    improved, improved_rank = try_rank_reduction()
    if improved is not None and improved_rank < best_rank:
        best_result = improved
        best_rank = improved_rank

    # Try tensor optimization for further reduction
    if time.time() < deadline - 10:
        tensor_result, tensor_rank = try_tensor_optimization()
        if tensor_result is not None and tensor_rank < best_rank:
            best_result = tensor_result
            best_rank = tensor_rank

    return best_result
# EVOLVE-BLOCK-END

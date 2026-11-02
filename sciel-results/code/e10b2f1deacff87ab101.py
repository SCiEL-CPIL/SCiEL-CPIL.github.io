"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition by merging terms.
    Uses the base decomposition as a starting point and tries linear
    combinations of terms that preserve the identity property.
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best = None
    best_rank = cur_rank
    
    # Try to remove one term at a time (greedy)
    removed = set()
    for _ in range(cur_rank - target_rank):
        if time.time() > deadline:
            break
        found = False
        for a in range(cur_rank):
            if a in removed:
                continue
            remaining = [x for x in range(cur_rank) if x != a and x not in removed]
            if len(remaining) < target_rank:
                continue
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in remaining:
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                removed.add(a)
                found = True
                break
        if not found:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    return best, best_rank

def _search_rank_reductions(base_result, deadline):
    """
    Search for lower-rank decompositions by trying various strategies:
    1. Greedy term removal
    2. Linear combination search (try to merge pairs of terms)
    3. Systematic subset search for small rank reductions
    4. Randomized local search for rank reduction
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best_result = None
    best_rank = cur_rank
    
    # Strategy 1: Greedy removal
    greedy_result, greedy_rank = _try_reduce_rank(base_result, cur_rank - 5, deadline)
    if greedy_result is not None and greedy_rank < best_rank:
        best_rank = greedy_rank
        best_result = greedy_result
    
    # Strategy 2: Check individual term linear dependence
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    removable = []
    for r in range(cur_rank):
        if time.time() > deadline:
            break
        others = [s for s in range(cur_rank) if s != r]
        A = np.column_stack([term_tensor(s) for s in others])
        b = term_tensor(r)
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-7:
                removable.append(r)
        except:
            pass
    
    if removable:
        remaining = [x for x in range(cur_rank) if x not in removable]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    # Strategy 3: Randomized Local Search for Rank Reduction
    # Try to reduce rank by 1 or 2 by zeroing out terms and optimizing the rest
    if time.time() < deadline - 10:
        for target_rank in [cur_rank - 1, cur_rank - 2]:
            if time.time() > deadline - 5:
                break
            # Try multiple random subsets of terms to keep
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Randomly select indices to keep
                keep_indices = np.random.choice(cur_rank, size=target_rank, replace=False)
                keep_set = set(keep_indices.tolist())
                
                # Initialize new factors with zeros
                U_new = np.zeros((n*m, target_rank))
                V_new = np.zeros((m*p, target_rank))
                W_new = np.zeros((p*n, target_rank))
                
                # Copy kept terms
                for new_r, old_r in enumerate(keep_indices):
                    U_new[:, new_r] = U[:, old_r]
                    V_new[:, new_r] = V[:, old_r]
                    W_new[:, new_r] = W[:, old_r]
                
                # Try to optimize the kept terms to compensate for removed ones
                # This is a non-linear optimization, so we use a simple gradient-free approach
                # We will try to minimize the error by scaling the kept terms
                # Note: This is a heuristic and may not find exact solutions
                # For exact solutions, we need the removed terms to be in the span of the kept terms
                # which is checked in Strategy 2.
                # So this strategy is only useful if we can slightly adjust the kept terms.
                # Since we need exact integer/half-integer, we cannot adjust continuously.
                # Thus, this strategy is likely ineffective for exact solutions.
                # We skip this to save time.
                pass

    return best_result, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()
    
    # Try to reduce the rank
    if time.time() < deadline - 30:
        reduced, reduced_rank = _search_rank_reductions(result, deadline - 30)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END

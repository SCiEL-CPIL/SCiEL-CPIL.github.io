import numpy as np
import time
import itertools
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n with radius r using LP-based methods.
    For q=3, n=8, r=1, we use LP relaxation + greedy refinement.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback greedy
        N = q ** n
        powers = [q ** i for i in range(n)]
        neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            temp = w
            for i in range(n):
                d = temp % q
                temp //= q
                for new_d in range(q):
                    if new_d != d:
                        neighbors[w, idx] = w + (new_d - d) * powers[i]
                        idx += 1
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[np.random.randint(len(candidates))]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[int(c) for c in np.base_repr(w, base=q).zfill(n)][::-1] for w in code]

    N = 3 ** 8  # 6561
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Precompute neighbors
    powers = [3 ** i for i in range(n)]
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            for new_d in [0, 1, 2]:
                if new_d != d:
                    neighbors[w, idx] = w + (new_d - d) * powers[i]
                    idx += 1

    # --- LP Relaxation to get initial fractional solution ---
    # Build incidence matrix A: A[i][j] = 1 if word i is within radius 1 of codeword j
    # Using neighbors: A[i][j] = 1 if i is in neighbors[j]
    # Build as sparse matrix
    row_idx = []
    col_idx = []
    for j in range(N):
        for i in neighbors[j]:
            row_idx.append(i)
            col_idx.append(j)
    
    A = csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(N, N))
    
    # LP: minimize 1^T x subject to A^T x >= 1, 0 <= x <= 1
    # A^T x >= 1 means for each word i, sum of x_j over codewords j covering i >= 1
    A_t = A.T.tocsr()
    
    # linprog: minimize c^T x subject to A_ub x <= b_ub, bounds
    # We need A^T x >= 1 => -A^T x <= -1
    c = np.ones(N)
    A_ub = -A_t
    b_ub = -np.ones(N)
    bounds = [(0, 1)] * N
    
    # Try LP solve with time limit
    t0 = time.time()
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'time_limit': 30})
        if res.success and res.x is not None:
            lp_solution = res.x
            lp_value = res.fun
        else:
            lp_solution = None
            lp_value = None
    except Exception:
        lp_solution = None
        lp_value = None
    
    if time.time() > deadline - 60:
        # Not enough time for LP, use seed + greedy
        pass

    # --- Strategy 1: Randomized rounding of LP solution ---
    rng = np.random.default_rng(123)
    best_code = None
    best_size = 999999

    if lp_solution is not None:
        # Randomized rounding: include codeword j with probability x_j
        # Then repair any uncovered words with greedy
        for restart in range(3):
            if time.time() > deadline:
                break
            rounded = (rng.random(N) < lp_solution).astype(int)
            # Find uncovered words
            covered = np.zeros(N, dtype=np.uint8)
            for j in range(N):
                if rounded[j]:
                    covered[neighbors[j]] = 1
            
            uncovered_mask = 1 - covered
            if uncovered_mask.sum() == 0:
                code = np.where(rounded == 1)[0].tolist()
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                continue
            
            # Greedily add codewords to cover uncovered
            uncovered = uncovered_mask.copy()
            code = list(np.where(rounded == 1)[0])
            while uncovered.sum() > 0 and time.time() < deadline:
                gains = (uncovered[neighbors]).sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                # Prefer codewords already in LP solution
                lp_pref = lp_solution[candidates]
                w = candidates[np.argmax(lp_pref)]
                code.append(w)
                uncovered[neighbors[w]] = 0
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

    # --- Strategy 2: Greedy with LP-guided tie-breaking ---
    # If LP failed, use uniform tie-breaking
    guide = lp_solution if lp_solution is not None else np.ones(N) / N
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        while uncovered.sum() > 0 and time.time() < deadline - 5:
            gains = (uncovered[neighbors]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            
            # LP-guided selection: weight by LP value
            weights = guide[candidates]
            if weights.sum() == 0:
                w = candidates[rng.randint(len(candidates))]
            else:
                # Normalize and sample
                weights = weights / weights.sum()
                w = candidates[rng.choice(len(candidates), p=weights)]
            
            code.append(w)
            uncovered[neighbors[w]] = 0
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # --- Strategy 3: Local search / swap improvement on best solution ---
    if best_code is not None and time.time() < deadline - 10:
        code_arr = np.array(best_code, dtype=np.int32)
        code_set = set(code_arr.tolist())
        
        # Build coverage count for current code
        cov_count = np.zeros(N, dtype=np.int16)
        for w in code_arr:
            cov_count[neighbors[w]] += 1
        
        improved = True
        max_iter = 200
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Find codewords that are "removable" (no word is only covered by them)
            for idx in range(len(code_arr)):
                c = code_arr[idx]
                # Words covered by c
                c_neighbors = neighbors[c]
                # Words that would become uncovered if we remove c
                at_risk = c_neighbors[cov_count[c_neighbors] == 1]
                if len(at_risk) == 0:
                    # Safe to remove
                    code_arr = np.delete(code_arr, idx)
                    code_set.discard(c)
                    cov_count[neighbors[c]] -= 1
                    improved = True
                    break
                elif len(at_risk) == 1 and time.time() < deadline - 5:
                    # Try to find a replacement
                    at_risk_word = at_risk[0]
                    # Find codewords that cover at_risk_word but aren't already in code
                    candidates = np.where(neighbors == at_risk_word)[0]
                    candidates = candidates[~np.isin(candidates, code_arr)]
                    if len(candidates) > 0:
                        # Pick one that covers most other at-risk words (only 1 here)
                        new_c = candidates[0]
                        # Check if adding new_c helps overall
                        code_arr[idx] = new_c
                        code_set.discard(c)
                        code_set.add(new_c)
                        cov_count[neighbors[c]] -= 1
                        cov_count[neighbors[new_c]] += 1
                        improved = True
                        break
        
        if len(code_arr) < best_size:
            best_size = len(code_arr)
            best_code = code_arr.tolist()

    # --- Final local search: try single swaps ---
    if best_code is not None and time.time() < deadline - 15:
        code_arr = np.array(best_code, dtype=np.int32)
        code_set = set(code_arr.tolist())
        
        cov_count = np.zeros(N, dtype=np.int16)
        for w in code_arr:
            cov_count[neighbors[w]] += 1
        
        # Try removing and replacing codewords
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(code_arr) == 0:
                break
            
            # Pick a random codeword to try removing
            idx = rng.randint(len(code_arr))
            c = code_arr[idx]
            
            # Words at risk if we remove c
            c_neighbors = neighbors[c]
            at_risk = c_neighbors[cov_count[c_neighbors] == 1]
            
            if len(at_risk) == 0:
                # Remove it
                code_arr = np.delete(code_arr, idx)
                code_set.discard(c)
                cov_count[neighbors[c]] -= 1
            elif len(at_risk) <= 2:
                # Find replacement
                # Candidates must cover all at_risk words
                cand_set = None
                for rw in at_risk:
                    cands = np.where(neighbors == rw)[0]
                    if cand_set is None:
                        cand_set = set(cands.tolist())
                    else:
                        cand_set &= set(cands.tolist())
                
                if cand_set:
                    cand_set -= code_set
                    cand_set.discard(c)
                    if cand_set:
                        # Pick candidate that maximizes coverage of uncovered-ish words
                        cand_list = list(cand_set)
                        best_cand = None
                        best_score = -1
                        for cc in cand_list[:20]:  # Limit candidates
                            new_cov = cov_count[neighbors[cc]]
                            # Score: how many words go from 1 to 2 (waste) vs 0 to 1 (good)
                            score = np.sum((new_cov == 0)) - np.sum((new_cov >= 2))
                            if score > best_score:
                                best_score = score
                                best_cand = cc
                        if best_cand is not None:
                            code_arr[idx] = best_cand
                            code_set.discard(c)
                            code_set.add(best_cand)
                            cov_count[neighbors[c]] -= 1
                            cov_count[neighbors[best_cand]] += 1

        if len(code_arr) < best_size:
            best_size = len(code_arr)
            best_code = code_arr.tolist()

    # Convert to final format
    if best_code is None:
        # Fallback: tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                w_int = 0
                for i in range(n):
                    w_int += word[i] * powers[i]
                best_code.append(w_int)

    final_code = []
    for w in best_code:
        word = []
        temp = int(w)
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        final_code.append(word)

    return final_code

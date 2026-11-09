"""MILP-based approach for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails. Gives 81 words for q=3,n=6."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword (exact check)."""
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    words_arr = np.array(words_list, dtype=np.int8)
    code_arr = np.array(code, dtype=np.int8)
    
    diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    covered = (dists <= radius).any(axis=1)
    return covered.all()


def _build_coverage_matrix(all_words_arr, q, n, radius):
    """Build the coverage matrix A where A[j,i]=1 if word i covers word j."""
    N = len(all_words_arr)
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        w = all_words_arr[i]
        if radius == 1:
            A[i, i] = 1.0
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    nw = w.copy()
                    nw[pos] = val
                    ni = _word_to_idx(nw.tolist(), q, n)
                    A[ni, i] = 1.0
        else:
            dists = np.abs(all_words_arr - w).sum(axis=1)
            A[dists <= radius, i] = 1.0
    return A


def _solve_milp(A, ubound, time_limit=60.0):
    """Solve the set cover MILP: min sum(x) s.t. A^T x >= 1, 0<=x<=1, x binary."""
    N = A.shape[1]
    c = np.ones(N)  # minimize sum of selected words
    # Constraints: A^T x >= 1 (each word must be covered)
    A_con = A.T  # Shape (N, N): row j says word j is covered by sum_i A[j,i]*x_i
    lb = np.ones(N)
    ub = np.full(N, ubound)
    constraints = LinearConstraint(A_con, lb, ub)
    bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
    integrality = np.ones(N)
    
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "presolve": True}
    )
    
    if result.status != 0 and result.status != 1:  # 0=optimal, 1=time limit
        return None
    
    x = result.x
    if x is None:
        return None
    
    # Get indices of selected words
    selected = np.where(x > 0.5)[0]
    return selected.tolist()


def _local_search_polish(code, all_words_arr, A, q, n, radius, time_limit=30.0, seed=42):
    """Local search to further reduce code size by removing redundant words."""
    rng = np.random.default_rng(seed)
    start = time.time()
    N = len(all_words_arr)
    
    code_idx = []
    word_to_idx = {}
    for i in range(N):
        word_to_idx[tuple(all_words_arr[i])] = i
        
    for cw in code:
        idx = word_to_idx.get(tuple(cw))
        if idx is not None:
            code_idx.append(idx)
            
    counts = np.zeros(N, dtype=np.int32)
    for ci in code_idx:
        counts += A[:, ci].astype(np.int32)
        
    current_size = len(code_idx)
    best_size = current_size
    best_code_idx = list(code_idx)
    
    def try_reduce(cod_idx, counts):
        removed = 0
        changed = True
        while changed:
            if (time.time() - start) > time_limit:
                break
            changed = False
            for ci in range(len(cod_idx)):
                if (time.time() - start) > time_limit:
                    break
                new_counts = counts - A[:, cod_idx[ci]].astype(np.int32)
                if np.all(new_counts > 0):
                    cod_idx = [x for i, x in enumerate(cod_idx) if i != ci]
                    counts = new_counts
                    removed += 1
                    changed = True
                    break
        return cod_idx, counts, removed

    code_idx, counts, _ = try_reduce(code_idx, counts)
    current_size = len(code_idx)
    best_size = current_size
    best_code_idx = list(code_idx)
    
    no_improve = 0
    max_no_improve = 50
    
    while (time.time() - start) < time_limit:
        if no_improve > max_no_improve:
            code_idx = list(best_code_idx)
            counts = np.zeros(N, dtype=np.int32)
            for ci in code_idx:
                counts += A[:, ci].astype(np.int32)
            current_size = len(code_idx)
            no_improve = 0
            continue

        improved = False
        for ci in range(len(code_idx)):
            if (time.time() - start) > time_limit:
                break
            if improved:
                break
                
            removed_idx = code_idx[ci]
            vulnerable = (counts == 1) & (A[:, removed_idx] > 0)
            
            if not np.any(vulnerable):
                code_idx, counts, _ = try_reduce(code_idx, counts)
                current_size = len(code_idx)
                if current_size < best_size:
                    best_size = current_size
                    best_code_idx = list(code_idx)
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
            
            # Try random replacements
            candidates = rng.choice(N, size=min(100, N), replace=False)
            for cand in candidates:
                if cand == removed_idx:
                    continue
                if np.all(A[cand, vulnerable]):
                    trial_idx = list(code_idx)
                    trial_idx[ci] = cand
                    trial_counts = counts - A[:, removed_idx].astype(np.int32) + A[:, cand].astype(np.int32)
                    if np.all(trial_counts > 0):
                        trial_idx, trial_counts, _ = try_reduce(trial_idx, trial_counts)
                        if len(trial_idx) < current_size:
                            code_idx = trial_idx
                            counts = trial_counts
                            current_size = len(code_idx)
                            if current_size < best_size:
                                best_size = current_size
                                best_code_idx = list(code_idx)
                                no_improve = 0
                            else:
                                no_improve += 1
                            improved = True
                            break
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
        
        if not improved:
            no_improve += 1

    return [all_words_arr[i].tolist() for i in best_code_idx]


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    words_list = [_idx_to_word(i, n, q) for i in range(total)]
    all_words_arr = np.array(words_list, dtype=np.int8)
    
    # Step 1: Get initial upper bound from tetracode
    initial_code = _direct_sum_tetracode(q, n, radius)
    if not _verify_code(initial_code, q, n, radius):
        initial_code = words_list
    
    ubound = len(initial_code)
    
    # Step 2: Build coverage matrix
    A = _build_coverage_matrix(all_words_arr, q, n, radius)
    
    # Step 3: Solve MILP
    best_code = initial_code
    best_size = ubound
    
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10
    if remaining > 20:
        try:
            milp_time = min(remaining, 80.0)
            selected = _solve_milp(A, ubound, time_limit=milp_time)
            if selected is not None and len(selected) < best_size:
                candidate_code = [all_words_arr[i].tolist() for i in selected]
                if _verify_code(candidate_code, q, n, radius):
                    best_code = candidate_code
                    best_size = len(selected)
        except Exception:
            pass
    
    # Step 4: Local search polish
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 10 and best_size > 71:
        try:
            polish_time = min(remaining, 30.0)
            polished = _local_search_polish(best_code, all_words_arr, A, q, n, radius, 
                                            time_limit=polish_time, seed=42)
            if _verify_code(polished, q, n, radius) and len(polished) < best_size:
                best_code = polished
                best_size = len(polished)
        except Exception:
            pass
    
    # Step 5: Another local search pass with different seed if time remains
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 5
    if remaining > 10 and best_size > 71:
        try:
            polish_time = min(remaining, 20.0)
            polished2 = _local_search_polish(best_code, all_words_arr, A, q, n, radius, 
                                             time_limit=polish_time, seed=123)
            if _verify_code(polished2, q, n, radius) and len(polished2) < best_size:
                best_code = polished2
                best_size = len(polished2)
        except Exception:
            pass

    return best_code

# EVOLVE-BLOCK-END

"""Kissing number in dimension 11 - candidate pool + greedy/local search."""
import time
import numpy as np
from itertools import combinations

D = 11
DEADLINE = 140.0  # safety margin before 150s kill


def _valid_set(V):
    """Check if all pairwise distances >= max norm in V."""
    if len(V) == 0:
        return True
    norms2 = np.sum(V * V, axis=1)
    max_n2 = norms2.max()
    # All pairwise squared distances must be >= max_n2
    # dist2(i,j) = |vi|^2 + |vj|^2 - 2*vi.vj
    # We need |vi|^2 + |vj|^2 - 2*vi.vj >= max_n2 for all i != j
    # Equivalently: 2*vi.vj <= |vi|^2 + |vj|^2 - max_n2
    G = V @ V.T
    n = len(V)
    idx = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = norms2[i] + norms2[j] - 2 * G[i, j]
            if d2 < max_n2:
                return False
    return True


def _greedy_build(candidates, norms2, G, max_n2_limit):
    """Greedy: add vectors one by one if they maintain validity.
    We track the current max norm squared and check new vector against all existing."""
    selected = []
    sel_norms2 = []
    sel_G = []  # dot products of selected with candidates (recomputed each add)
    
    # Sort candidates by norm squared ascending (prefer smaller norms first)
    order = np.argsort(norms2)
    
    for idx in order:
        v = candidates[idx]
        n2 = norms2[idx]
        
        # New max norm squared if we add this
        new_max_n2 = max(max_n2_limit, n2)
        
        # Check: for every existing selected vector s,
        # dist2(v, s) >= new_max_n2
        # dist2 = n2 + sel_n2 - 2 * dot
        # So: 2 * dot <= n2 + sel_n2 - new_max_n2
        ok = True
        if selected:
            dots = candidates[idx] @ np.array(selected)
            for k, s in enumerate(selected):
                d2 = n2 + sel_norms2[k] - 2 * dots[k]
                if d2 < new_max_n2:
                    ok = False
                    break
        
        if ok:
            selected.append(v)
            sel_norms2.append(n2)
    
    return selected


def _local_search(V, candidates, norms2, G, max_iters=5000):
    """Try to swap or add vectors to improve the set size."""
    n = len(V)
    V_arr = np.array(V)
    norms2_arr = np.sum(V_arr * V_arr, axis=1)
    max_n2 = norms2_arr.max()
    
    best = list(V)
    best_n = n
    
    for iteration in range(max_iters):
        # Try adding each candidate
        added = False
        for c_idx in range(len(candidates)):
            v = candidates[c_idx]
            n2_c = norms2[c_idx]
            new_max_n2 = max(max_n2, n2_c)
            
            # Check against all in best
            dots = v @ np.array(best)
            ok = True
            for k in range(len(best)):
                d2 = n2_c + np.sum(best[k] * best[k]) - 2 * dots[k]
                if d2 < new_max_n2:
                    ok = False
                    break
            
            if ok:
                best.append(v)
                best_n += 1
                added = True
                # Recompute max_n2
                max_n2 = new_max_n2
                break
        
        if not added:
            # Try swaps: remove one, add one
            swapped = False
            for rem in range(len(best)):
                for c_idx in range(len(candidates)):
                    v = candidates[c_idx]
                    n2_c = norms2[c_idx]
                    
                    # New set without best[rem]
                    remaining = best[:rem] + best[rem+1:]
                    if not remaining:
                        new_max_n2 = n2_c
                    else:
                        rem_norms2 = [np.sum(x * x) for x in remaining]
                        new_max_n2 = max(max(rem_norms2), n2_c)
                    
                    # Check v against remaining
                    ok = True
                    for k in range(len(remaining)):
                        d2 = n2_c + np.sum(remaining[k] * remaining[k]) - 2 * (v @ remaining[k])
                        if d2 < new_max_n2:
                            ok = False
                            break
                    
                    # Also check remaining pairs are still valid (they were before, removing one doesn't hurt)
                    if ok and len(remaining) + 1 > best_n - 1:
                        # Only accept if it doesn't decrease size (swap keeps size same, but we want growth)
                        # Actually swap keeps size same; we only do this if it enables future growth
                        # For now, skip pure swaps to keep it simple
                        pass
        
        if time.time() > DEADLINE - 5:
            break
    
    return best


def construct():
    start_time = time.time()
    
    # Build candidate pool
    candidates = []
    
    # 1. All single-unit vectors ±e_i (22 vectors)
    for i in range(D):
        v = [0] * D
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)
    
    # 2. All pair vectors ±±e_i ±e_j (220 vectors) - the seed set
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # 3. Some triple vectors with small coefficients
    #    e_i + e_j + e_k, e_i + e_j - e_k, etc. (norm^2 = 3)
    #    These are valid with pair vectors if distance checks out
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # 4. Vectors with a 2 in one position (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    
    # 5. Vectors with 2 and 1 (norm^2 = 5)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = sj
                candidates.append(v)
                v2 = [0] * D
                v2[i] = sj
                v2[j] = 2 * si
                candidates.append(v2)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared and Gram matrix
    norms2 = np.sum(candidates * candidates, axis=1)
    G = candidates @ candidates.T
    
    # Greedy build: sort by norm, add if valid
    # Key constraint: all pairwise distances >= max_norm
    # Start with smallest norm vectors
    order = np.argsort(norms2)
    
    selected = []
    sel_idx = []
    
    for idx in order:
        if time.time() > DEADLINE - 10:
            break
        
        v = candidates[idx]
        n2 = norms2[idx]
        
        # New max norm squared
        if selected:
            cur_max_n2 = max(max(norms2[sel_idx]), n2)
        else:
            cur_max_n2 = n2
        
        # Check v against all selected
        ok = True
        for s_idx in sel_idx:
            d2 = n2 + norms2[s_idx] - 2 * G[idx, s_idx]
            if d2 < cur_max_n2:
                ok = False
                break
        
        if ok:
            selected.append(idx)
            sel_idx.append(idx)
    
    # Convert to array
    result = candidates[selected]
    
    # Try local search to add more
    # For efficiency, work with the selected set
    if time.time() < DEADLINE - 15 and len(selected) < n_cand:
        # Try adding unselected candidates
        unselected = [i for i in range(n_cand) if i not in set(selected)]
        
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            sel_set = set(selected)
            cur_max_n2 = max(norms2[selected]) if selected else 0
            
            for c_idx in unselected:
                if c_idx in sel_set:
                    continue
                n2_c = norms2[c_idx]
                new_max_n2 = max(cur_max_n2, n2_c)
                
                ok = True
                for s_idx in selected:
                    d2 = n2_c + norms2[s_idx] - 2 * G[c_idx, s_idx]
                    if d2 < new_max_n2:
                        ok = False
                        break
                
                if ok:
                    selected.append(c_idx)
                    sel_set.add(c_idx)
                    cur_max_n2 = new_max_n2
                    unselected.remove(c_idx)
                    improved = True
                    break
    
    result = np.array(candidates[selected], dtype=np.int64)
    
    # Final validation
    if len(result) > 1:
        res_norms2 = np.sum(result * result, axis=1)
        max_n2 = res_norms2.max()
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = res_norms2[i] + res_norms2[j] - 2 * (result[i] @ result[j])
                if d2 < max_n2:
                    # Shouldn't happen, but safety
                    return np.array(candidates[:220], dtype=np.int64)
    
    return result


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} vectors")
    if len(result) > 1:
        norms2 = np.sum(result * result, axis=1)
        max_n2 = norms2.max()
        min_d2 = min(
            norms2[i] + norms2[j] - 2 * (result[i] @ result[j])
            for i in range(len(result)) for j in range(i+1, len(result))
        )
        print(f"Max norm^2: {max_n2}, Min dist^2: {min_d2}")
        print(f"Valid: {min_d2 >= max_n2}")

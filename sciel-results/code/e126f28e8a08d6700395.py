import itertools
import random
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    N = 4096

    def compute_all_sums(chosen):
        """Compute all unordered triple sums (with repetition) for a sorted list."""
        m = len(chosen)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(chosen[i] + chosen[j] + chosen[k])
        return sums

    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    def check_add_fast(v, chosen, sums):
        """
        Fast check if v can be added. Uses numpy for the O(n^2) pair part.
        Returns (ok, new_sums_to_add) or (False, None).
        """
        m = len(chosen)
        new_sums = set()
        
        # (v, v, v)
        new_sums.add(3 * v)
        
        if m == 0:
            if 3 * v in sums:
                return False, None
            return True, new_sums
        
        chosen_arr = np.array(chosen, dtype=np.int64)
        
        # (v, v, u) for each u
        new_sums.update((2 * v + chosen_arr).tolist())
        
        # (v, u, u) for each u
        new_sums.update((v + 2 * chosen_arr).tolist())
        
        # (v, u, w) for u <= w: vectorized
        # Generate all pairs u_i + w_j where i <= j
        # Use broadcasting: chosen_arr[:, None] + chosen_arr[None, :]
        pair_sums = chosen_arr[:, None] + chosen_arr[None, :]  # m x m
        # Only take upper triangle (i <= j)
        mask = np.triu(np.ones((m, m), dtype=bool))
        valid_pairs = pair_sums[mask]
        triple_sums_v = v + valid_pairs
        new_sums.update(triple_sums_v.tolist())
        
        # Check conflicts
        if new_sums & sums:
            return False, None
        return True, new_sums

    def greedy_build_from_base(base_set, order, deadline):
        """Greedy add from a base set following the given order."""
        chosen = sorted(base_set)
        sums = compute_all_sums(chosen)
        chosen_set = set(chosen)
        
        for v in order:
            if time.time() > deadline:
                break
            if v in chosen_set:
                continue
            if v > N:
                continue
            ok, ns = check_add_fast(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
                chosen_set.add(v)
        
        return chosen

    def base3_sparse():
        """
        Start with a sparse base-3-like set that is provably B_3[1].
        Use a_i = c * 3^i with small c to stay in range.
        """
        # Powers of 3: 1, 3, 9, 27, 81, 243, 729, 2187
        # 3^8 = 6561 > 4096
        # We can use multipliers: a_i = d_i * 3^i where d_i in {1,2}
        # This gives unique base-3-like representations for triple sums
        
        candidates = []
        for i in range(8):
            for d in [1, 2]:
                val = d * (3 ** i)
                if val <= N:
                    candidates.append(val)
        
        # Sort and greedily build
        candidates.sort()
        chosen = []
        sums = set()
        for v in candidates:
            ok, ns = check_add_fast(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        return chosen

    best = []
    
    # Strategy 1: Base-3 sparse start, then dense greedy
    base = base3_sparse()
    if is_valid(base):
        best = list(base)
    
    # Strategy 2: Multiple greedy passes with different orderings
    # Ascending order
    asc_order = list(range(1, N + 1))
    res = greedy_build_from_base(best, asc_order, DEADLINE - 20)
    if is_valid(res) and len(res) > len(best):
        best = res
    
    # Descending order
    desc_order = list(range(N, 0, -1))
    if time.time() < DEADLINE - 15:
        res = greedy_build_from_base(best, desc_order, DEADLINE - 15)
        if is_valid(res) and len(res) > len(best):
            best = res
    
    # Random orderings
    random.seed(42)
    for trial in range(10):
        if time.time() > DEADLINE - 5:
            break
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        res = greedy_build_from_base(best, rand_order, DEADLINE - 5)
        if is_valid(res) and len(res) > len(best):
            best = res
    
    # Strategy 3: Local search - try to swap elements to allow more additions
    def local_search(chosen, deadline):
        """
        Try to improve by removing one element and adding a better one,
        or by trying to add more elements after removals.
        """
        current = sorted(chosen)
        current_sums = compute_all_sums(current)
        current_set = set(current)
        best_local = list(current)
        best_score = len(current)
        
        # Try removing each element and then greedily adding
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            
            removed = current[idx]
            new_current = [x for x in current if x != removed]
            new_set = set(new_current)
            
            # Recompute sums
            new_sums = compute_all_sums(new_current)
            
            # Try to add back removed and others
            # Greedily add from ascending
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                if v in new_set:
                    continue
                ok, ns = check_add_fast(v, new_current, new_sums)
                if ok:
                    new_sums |= ns
                    new_current.append(v)
                    new_set.add(v)
            
            if is_valid(new_current) and len(new_current) > best_score:
                best_local = new_current
                best_score = len(new_current)
        
        return best_local

    if time.time() < DEADLINE - 10 and best:
        ls_result = local_search(best, DEADLINE - 5)
        if is_valid(ls_result) and len(ls_result) > len(best):
            best = ls_result
    
    # Strategy 4: Final polish - try all possible single additions
    if time.time() < DEADLINE - 2:
        best_set = set(best)
        best_sums = compute_all_sums(best)
        improved = True
        while improved and time.time() < DEADLINE - 1:
            improved = False
            for v in range(1, N + 1):
                if v in best_set:
                    continue
                ok, ns = check_add_fast(v, best, best_sums)
                if ok:
                    best_sums |= ns
                    best.append(v)
                    best_set.add(v)
                    improved = True
                    break
    
    best = sorted(set(best))
    return best

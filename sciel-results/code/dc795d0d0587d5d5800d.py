"""Rewritten program: greedy + local search + SA for odd covering with moduli dividing 51975."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0  # wall-clock limit with margin before 150s kill

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Precompute all divisors > 1 of base
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    num_divisors = len(divisors)

    # Precompute index arrays for each (m, r) pair for fast access
    # For each divisor m, precompute the list of indices for each residue r
    # This is memory heavy but fast: for m, there are m residue classes, each with base//m elements
    # Total elements across all divisors: sum_{m|base, m>1} base = num_divisors * base
    # That's about 48 * 51975 ≈ 2.5M elements, manageable

    # Actually, let's precompute coverage contributions differently
    # For each divisor m, precompute an array of length base where entry i is 1 if i % m == r for some r
    # No, we need per-residue.

    # Let's precompute for each m, a list of arrays: idx_arrays[m_idx][r] = np.array of positions
    # But that's a lot of memory. Let's use a different approach.

    # For local search, we need to quickly compute: given a set of (m, r) pairs, the coverage
    # And given removing one pair and adding another, the delta.

    # Key insight: for a given m, the residue classes partition [0, base).
    # So for a fixed m, exactly one residue class covers each position.

    # Let's precompute for each divisor m:
    #   pos_of[m][i] = i % m  (the residue class of position i)
    # This is just i % m, which we can compute on the fly.

    # Better approach: for each m, precompute a numpy array of shape (base,) where
    #   residue_of_pos[m_idx, i] = i % m
    # Then coverage by (m, r) is: residue_of_pos[m_idx, :] == r

    # But storing this for all m is num_divisors * base integers ≈ 48 * 51975 * 4 bytes ≈ 10 MB
    # That's fine.

    m_to_idx = {m: i for i, m in enumerate(divisors)}
    
    # Precompute residue arrays
    residue_arrays = []
    for m in divisors:
        arr = np.arange(base, dtype=np.int32) % m
        residue_arrays.append(arr)

    def compute_coverage(classes):
        """Compute coverage array from list of [m, r] pairs."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            mi = m_to_idx[m]
            # covered where residue_arrays[mi] == r
            covered[residue_arrays[mi] == r] = 1
        return covered

    def compute_score(covered):
        return base - np.sum(covered)

    def greedy_construct(order):
        """Greedy: process moduli in given order, each picks best residue."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            mi = m_to_idx[m]
            best_r = 0
            best_count = -1
            res_arr = residue_arrays[mi]
            for r in range(m):
                # Count newly covered positions
                mask = (res_arr == r) & (covered == 0)
                count = np.sum(mask)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[res_arr == best_r] = 1
        return classes, int(compute_score(covered))

    def local_search(classes, max_iter=5000):
        """Local search: try changing one class's residue at a time."""
        covered = compute_coverage(classes)
        current_score = int(compute_score(covered))
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick a random class to modify
            idx = random.randint(0, len(classes) - 1)
            m_old, r_old = classes[idx]
            mi = m_to_idx[m_old]
            res_arr = residue_arrays[mi]
            
            # For each candidate new residue, compute the score change
            # Score change = (positions lost from old_r that aren't covered by others) - (positions gained from new_r that weren't covered by others)
            
            # Compute coverage without this class
            # To do this efficiently, we need to know for each position, how many classes cover it
            # Let's compute a count array: count[i] = number of classes covering position i
            
            # This is expensive to recompute each iteration. Let's maintain it.
            # Actually, let's just recompute coverage without this class for each candidate.
            # For small m, this is feasible. For large m, skip.
            
            if m_old > 500:
                # Sample a few residues
                candidates = random.sample(range(m_old), min(20, m_old))
            else:
                candidates = list(range(m_old))
            
            best_new_r = r_old
            best_score = current_score
            
            # Compute coverage without this class once
            covered_no_idx = np.zeros(base, dtype=np.uint8)
            for k_idx in range(len(classes)):
                if k_idx == idx:
                    continue
                mk, rk = classes[k_idx]
                mi_k = m_to_idx[mk]
                covered_no_idx[residue_arrays[mi_k] == rk] = 1
            
            for new_r in candidates:
                if new_r == r_old:
                    continue
                
                # Coverage with new_r instead of r_old
                covered_with_new = covered_no_idx.copy()
                covered_with_new[res_arr == new_r] = 1
                score_new = int(compute_score(covered_with_new))
                
                if score_new < best_score:
                    best_score = score_new
                    best_new_r = new_r
            
            if best_new_r != r_old:
                classes[idx][1] = best_new_r
                current_score = best_score
                covered = compute_coverage(classes)
        
        return classes, current_score

    def simulated_annealing(classes, max_iter=10000):
        """Simulated annealing to escape local optima."""
        covered = compute_coverage(classes)
        current_score = int(compute_score(covered))
        best_classes = [c[:] for c in classes]
        best_score = current_score
        
        T = 30.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            if T < T_min:
                break
                
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            mi = m_to_idx[m]
            res_arr = residue_arrays[mi]
            
            new_r = random.randint(0, m - 1)
            if new_r == r_old:
                continue
            
            if m > 500:
                continue
            
            # Compute coverage without this class
            covered_no_idx = np.zeros(base, dtype=np.uint8)
            for k_idx in range(len(classes)):
                if k_idx == idx:
                    continue
                mk, rk = classes[k_idx]
                mi_k = m_to_idx[mk]
                covered_no_idx[residue_arrays[mi_k] == rk] = 1
            
            covered_with_old = covered_no_idx.copy()
            covered_with_old[res_arr == r_old] = 1
            score_old = int(compute_score(covered_with_old))
            
            covered_with_new = covered_no_idx.copy()
            covered_with_new[res_arr == new_r] = 1
            score_new = int(compute_score(covered_with_new))
            
            delta = score_new - score_old
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                classes[idx][1] = new_r
                current_score = score_new
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
            
            T *= cooling_rate
        
        return best_classes, best_score

    # Try multiple initial orderings
    orderings = []
    
    # Ordering 1: increasing
    orderings.append(divisors.copy())
    
    # Ordering 2: decreasing
    orderings.append(divisors[::-1].copy())
    
    # Ordering 3: sort by number of prime factors descending
    def omega(n):
        count = 0
        temp = n
        for p in [3, 5, 7, 11]:
            while temp % p == 0:
                count += 1
                temp //= p
        return count
    
    orderings.append(sorted(divisors, key=lambda m: (-omega(m), m)))
    
    # Ordering 4: sort by m ascending
    orderings.append(sorted(divisors, key=lambda m: m))
    
    # Random orderings
    for _ in range(15):
        perm = divisors.copy()
        random.shuffle(perm)
        orderings.append(perm)
    
    best_classes = None
    best_score = base  # worst case: nothing covered
    
    # Phase 1: Greedy construction with different orderings
    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 40:
            break
        try:
            classes, sc = greedy_construct(order)
            if sc < best_score:
                best_score = sc
                best_classes = classes
        except Exception:
            pass
    
    # Phase 2: Local search on the best result
    if best_classes is not None and time.time() - start_time < deadline - 25:
        try:
            best_classes, best_score = local_search(best_classes, max_iter=8000)
        except Exception:
            pass
    
    # Phase 3: Simulated annealing
    if best_classes is not None and time.time() - start_time < deadline - 10:
        try:
            best_classes, best_score = simulated_annealing(best_classes, max_iter=15000)
        except Exception:
            pass
    
    # Phase 4: Multiple restarts if time allows
    while time.time() - start_time < deadline - 15:
        order = divisors.copy()
        random.shuffle(order)
        
        try:
            classes, sc = greedy_construct(order)
            if sc < best_score:
                best_score = sc
                best_classes = classes
            
            if time.time() - start_time < deadline - 10:
                classes, sc = local_search(classes, max_iter=3000)
                if sc < best_score:
                    best_score = sc
                    best_classes = classes
        except Exception:
            pass
    
    return best_classes if best_classes is not None else [[3, 0]]
# EVOLVE-BLOCK-END

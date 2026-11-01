import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    total_words = q ** n
    
    # Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks using Python ints for fast bit operations
    coverage_masks = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        lst = list(w)
                        lst[pos] = nv
                        j = word_to_idx[tuple(lst)]
                        mask |= (1 << j)
        coverage_masks[i] = mask
    
    all_ones = (1 << total_words) - 1
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    # Try ILP first with scipy
    try:
        import scipy.sparse as sp
        import scipy.optimize as opt
        
        # Build constraint matrix
        rows = []
        cols = []
        data = []
        
        for i, w_i in enumerate(all_words):
            covered = [i]
            if radius == 1:
                for pos in range(n):
                    orig = w_i[pos]
                    for nv in range(q):
                        if nv != orig:
                            lst = list(w_i)
                            lst[pos] = nv
                            j = word_to_idx[tuple(lst)]
                            covered.append(j)
            for j in covered:
                rows.append(j)
                cols.append(i)
                data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
        c = np.ones(total_words)
        
        # Symmetry breaking: fix x_0 = 1
        bounds = []
        for i in range(total_words):
            if i == 0:
                bounds.append((1, 1))
            else:
                bounds.append((0, 1))
        
        b = np.ones(total_words)
        constraint = opt.LinearConstraint(-A, lb=-np.inf, ub=-b)
        integrality = np.ones(total_words)
        
        res = opt.milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected_indices = [i for i in range(total_words) if x[i] > 0.5]
            return [list(all_words[i]) for i in selected_indices]
        
    except Exception:
        pass
    
    # Fallback: improved greedy + local search with time management
    
    def greedy_cover(seed_random, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5
        uncovered = all_ones
        chosen = []
        chosen_set = set()
        indices = list(range(total_words))
        rng = random.Random(seed_random)
        rng.shuffle(indices)
        
        while uncovered and time.time() < time_limit:
            best_count = -1
            best_candidates = []
            for i in indices:
                if i in chosen_set:
                    continue
                common = coverage_masks[i] & uncovered
                cnt = popcount(common)
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [i]
                elif cnt == best_count:
                    best_candidates.append(i)
            if not best_candidates:
                break
            pick = rng.choice(best_candidates)
            chosen.append(pick)
            chosen_set.add(pick)
            uncovered &= ~coverage_masks[pick]
        
        return chosen, chosen_set
    
    def local_search(chosen, chosen_set, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(chosen)
            if k == 0:
                break
            
            masks = [coverage_masks[c] for c in chosen]
            total_cov = 0
            for m in masks:
                total_cov |= m
            
            if total_cov != all_ones:
                break
            
            # Try removal
            prefix_or = [0] * (k + 1)
            for i in range(k):
                prefix_or[i + 1] = prefix_or[i] | masks[i]
            suffix_or = [0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | masks[i]
            
            removable_indices = []
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                if without_i == all_ones:
                    removable_indices.append(i)
            
            if removable_indices:
                removed_idx = removable_indices[0]
                c = chosen.pop(removed_idx)
                chosen_set.discard(c)
                improved = True
                continue
            
            # Try 2-for-1 swap
            missing_if_remove = [0] * k
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                missing_if_remove[i] = all_ones & ~without_i
            
            important = [i for i in range(k) if missing_if_remove[i] != 0]
            
            if len(important) < 2:
                break
            
            found_swap = False
            max_pairs = 300
            pair_count = 0
            for idx_i in range(len(important)):
                if time.time() > time_limit - 1:
                    break
                i = important[idx_i]
                for idx_j in range(idx_i + 1, len(important)):
                    if pair_count >= max_pairs:
                        break
                    pair_count += 1
                    j = important[idx_j]
                    
                    or_without_ij = 0
                    for m_idx in range(k):
                        if m_idx != i and m_idx != j:
                            or_without_ij |= masks[m_idx]
                    
                    missing_ij = all_ones & ~or_without_ij
                    if missing_ij == 0:
                        if i > j: i, j = j, i
                        chosen.pop(i)
                        chosen_set.discard(chosen[i])
                        improved = True
                        found_swap = True
                        break
                    else:
                        for cp in range(total_words):
                            if cp in chosen_set:
                                continue
                            cm = coverage_masks[cp]
                            if (missing_ij & ~cm) == 0:
                                if i > j: i, j = j, i
                                word_i = chosen[i]
                                word_j = chosen[j]
                                chosen.pop(j)
                                chosen_set.discard(word_j)
                                chosen.pop(i)
                                chosen_set.discard(word_i)
                                chosen.append(cp)
                                chosen_set.add(cp)
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        return chosen, chosen_set
    
    # Multi-start with adaptive time allocation
    best_chosen = None
    best_size = total_words + 1
    
    # Phase 1: quick greedy to get a baseline
    baseline_chosen, _ = greedy_cover(42, deadline - 30)
    baseline_size = len(baseline_chosen)
    best_size = baseline_size
    best_chosen = list(baseline_chosen)
    
    # Phase 2: local search on baseline
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        local_search(best_chosen, set(best_chosen), deadline - 5)
        if len(best_chosen) < best_size:
            best_size = len(best_chosen)
            best_chosen = list(best_chosen)
    
    # Phase 3: multiple restarts with different seeds
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        seed = 100 + r * 13
        time_limit = deadline - 10 - r * 0.5  # decreasing time per restart
        
        if time.time() > time_limit - 2:
            break
        
        chosen, chosen_set = greedy_cover(seed, time_limit - 3)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
        
        # Local search if we have time
        if time.time() < deadline - 8 and len(chosen) <= best_size:
            chosen, chosen_set = local_search(chosen, chosen_set, deadline - 5)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
    
    # Phase 4: final polish on best solution
    if time.time() < deadline - 3 and best_chosen is not None:
        local_search(best_chosen, set(best_chosen), deadline - 1)
    
    if best_chosen is None:
        best_chosen, _ = greedy_cover(0, deadline - 1)
    
    return [list(all_words[i]) for i in best_chosen]

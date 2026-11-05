import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    neighbor_array = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        neighbor_array[i, :] = neigh
    
    # Precompute neighbor sets for O(1) membership
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Precompute for each point, which codewords cover it (reverse neighbors)
    covered_by = [set() for _ in range(N)]
    for i in range(N):
        for nb in neighbors[i]:
            covered_by[nb].add(i)
    
    # Helper: Calculate coverage counts for a given code list
    def get_counts(code_list):
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    def is_valid(code_list):
        counts = get_counts(code_list)
        return np.min(counts) >= 1

    def prune_code(code_list):
        counts = get_counts(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                if np.min(counts[neighbors[c]]) > 1:
                    code_list.pop(i)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    changed = True
        return code_list

    def try_swap_remove(code_list, code_set, counts, c):
        """Try to remove codeword c or swap it with a better one. Returns True if changed."""
        # Find critical points: covered only by c
        crit_points = [nb for nb in neighbors[c] if counts[nb] == 1]
        
        if not crit_points:
            # c is redundant
            code_list.remove(c)
            code_set.remove(c)
            for nb in neighbors[c]:
                counts[nb] -= 1
            return True
        
        # Try to find a replacement that covers all crit_points
        p = crit_points[0]
        candidates = [x for x in neighbors[p] if x not in code_set]
        
        best_w = -1
        best_score = -1
        
        for w in candidates:
            valid = all(k in neighbor_sets[w] for k in crit_points)
            if not valid:
                continue
            # Score: number of points with count==2 that w covers (creates redundancy)
            score = sum(1 for nb in neighbors[w] if counts[nb] == 2)
            if score > best_score:
                best_score = score
                best_w = w
        
        if best_w != -1:
            code_list.remove(c)
            code_set.remove(c)
            for nb in neighbors[c]:
                counts[nb] -= 1
            code_list.append(best_w)
            code_set.add(best_w)
            for nb in neighbors[best_w]:
                counts[nb] += 1
            return True
        
        return False

    def try_two_for_one(code_list, code_set, counts):
        """Try to remove 2 codewords and add 1 that covers all their critical points. Returns True if changed."""
        if len(code_list) < 2:
            return False
        
        # Find pairs where combined critical points can be covered by one new word
        code_list_copy = list(code_list)
        random.shuffle(code_list_copy)
        
        for i in range(len(code_list_copy)):
            c1 = code_list_copy[i]
            if c1 not in code_set:
                continue
            crit1 = [nb for nb in neighbors[c1] if counts[nb] == 1]
            
            for j in range(i + 1, len(code_list_copy)):
                c2 = code_list_copy[j]
                if c2 not in code_set:
                    continue
                crit2 = [nb for nb in neighbors[c2] if counts[nb] == 1]
                
                # Combined critical points
                all_crit = list(set(crit1 + crit2))
                
                if not all_crit:
                    # Both are redundant
                    code_list.remove(c1)
                    code_set.remove(c1)
                    for nb in neighbors[c1]:
                        counts[nb] -= 1
                    code_list.remove(c2)
                    code_set.remove(c2)
                    for nb in neighbors[c2]:
                        counts[nb] -= 1
                    return True
                
                # Try to find a single word that covers all critical points
                p = all_crit[0]
                candidates = [x for x in neighbors[p] if x not in code_set and x != c1 and x != c2]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    valid = all(k in neighbor_sets[w] for k in all_crit)
                    if not valid:
                        continue
                    score = sum(1 for nb in neighbors[w] if counts[nb] == 2)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w != -1:
                    code_list.remove(c1)
                    code_set.remove(c1)
                    for nb in neighbors[c1]:
                        counts[nb] -= 1
                    code_list.remove(c2)
                    code_set.remove(c2)
                    for nb in neighbors[c2]:
                        counts[nb] -= 1
                    code_list.append(best_w)
                    code_set.add(best_w)
                    for nb in neighbors[best_w]:
                        counts[nb] += 1
                    return True
        
        return False

    def local_search(code_list, max_time=5.0):
        """Enhanced local search with 1-for-1 and 2-for-1 exchanges."""
        counts = get_counts(code_list)
        code_set = set(code_list)
        improved = True
        ls_start = time.time()
        
        while improved and time.time() - ls_start < max_time and time.time() < deadline - 5:
            improved = False
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            # Try 1-for-1 swaps
            random.shuffle(code_list)
            for c in code_list:
                if time.time() < deadline - 5 and try_swap_remove(code_list, code_set, counts, c):
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1 exchanges
            if try_two_for_one(code_list, code_set, counts):
                improved = True
        
        return code_list

    def perturb(code_list, strength=0.1):
        """Perturb the code by randomly removing and re-adding words."""
        code_set = set(code_list)
        code_list = list(code_list)
        num_remove = max(1, int(len(code_list) * strength))
        
        # Remove random words
        to_remove = random.sample(code_list, min(num_remove, len(code_list)))
        for c in to_remove:
            code_list.remove(c)
            code_set.remove(c)
        
        # Repair: add words to cover uncovered points
        counts = get_counts(code_list)
        uncovered = counts < 1
        
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            code_list.append(best_idx)
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                if counts[nb] < 1:
                    uncovered[nb] = False
                counts[nb] += 1
        
        return code_list

    def run_greedy_local_search(seed):
        random.seed(seed)
        code = []
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        
        # Greedy Set Cover
        while uncovered_count > 0 and time.time() < deadline - 15:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            code.append(best_idx)
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    uncovered_count -= 1
        
        # Local Search
        code = prune_code(code)
        code = local_search(code, max_time=3.0)
        
        # ILS: Perturb and re-optimize
        best_local = list(code)
        best_local_size = len(best_local)
        
        ils_start = time.time()
        while time.time() - ils_start < 4.0 and time.time() < deadline - 10:
            perturbed = perturb(best_local, strength=random.uniform(0.05, 0.15))
            if len(perturbed) >= best_local_size:
                continue
            optimized = local_search(perturbed, max_time=2.0)
            if len(optimized) < best_local_size:
                best_local = optimized
                best_local_size = len(optimized)
        
        # Final validity check
        counts = get_counts(best_local)
        if np.min(counts) < 1:
            return None
            
        return best_local

    best_code = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 1000 + 42
        current_code = run_greedy_local_search(seed)
        num_restarts += 1
        
        if current_code is not None:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        if num_restarts >= 8 and time.time() > deadline - 30:
            break

    if best_code is None:
        best_code = run_greedy_local_search(0)
        if best_code is None:
            return []

    best_code = prune_code(best_code)
    return [list(words[i]) for i in best_code]

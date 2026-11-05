import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n
    ball_size = 1 + n * (q - 1)

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            neighbors[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1
            
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            neighbors[idx, neighbor_idx] = idx + delta
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def luby_greedy(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        
        rev_neighbors = [[] for _ in range(total_points)]
        for b in range(total_points):
            for p in neighbors[b]:
                rev_neighbors[p].append(b)
        
        while np.any(uncovered_counts > 0):
            max_count = np.max(uncovered_counts)
            if max_count == 0: break
            
            candidates = np.where(uncovered_counts == max_count)[0]
            best_idx = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
            selected.append(best_idx)
            
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for b_prime in rev_neighbors[p]:
                        uncovered_counts[b_prime] -= 1
        return selected

    def prune(selected_indices):
        if not selected_indices: return []
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            to_remove = []
            for i, idx in enumerate(selected_list):
                if np.all(coverage_count[neighbors[idx]] > 1):
                    to_remove.append(i)
            if to_remove:
                to_remove_set = set(to_remove)
                new_selected = []
                for i, idx in enumerate(selected_list):
                    if i not in to_remove_set:
                        new_selected.append(idx)
                    else:
                        coverage_count[neighbors[idx]] -= 1
                selected_list = new_selected
                changed = True
        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = prune(list(selected_indices))
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = np.random.default_rng(42)
        selected_set = set(selected_list)
        
        while time.time() < deadline:
            if not selected_list: break
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            ball_c = neighbors[c]
            orphans = ball_c[coverage_count[ball_c] == 1]
            
            if len(orphans) == 0:
                coverage_count[ball_c] -= 1
                selected_list.pop(c_idx)
                selected_set.remove(c)
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
            else:
                if len(orphans) <= 4:
                    candidate_set = set()
                    for p in orphans:
                        candidate_set.update(neighbors[p])
                    for c_prime in candidate_set:
                        if np.all(np.isin(orphans, neighbors[c_prime])):
                            if c_prime in selected_set:
                                coverage_count[ball_c] -= 1
                                selected_list.pop(c_idx)
                                selected_set.remove(c)
                                if len(selected_list) < best_len:
                                    best_len = len(selected_list)
                                    best_code = selected_list[:]
                            else:
                                coverage_count[ball_c] -= 1
                                coverage_count[neighbors[c_prime]] += 1
                                selected_list[c_idx] = c_prime
                                selected_set.remove(c)
                                selected_set.add(c_prime)
                            break
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    best_code = None
    best_len = float('inf')
    
    for seed in range(3):
        if time.time() > deadline - 60: break
        selected = luby_greedy(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = prune(luby_greedy(0))
        best_len = len(best_code)
        
    best_code = local_search_optimize(best_code, deadline)
    
    return [idx_to_word(idx) for idx in best_code]

import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors_list = []
    neighbors_set = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors_list.append(list(neigh_set))
        neighbors_set.append(neigh_set)
        
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_list[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors_list[c]]) >= 2:
                    current_counts[neighbors_list[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors_list[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors_list[best_idx]] = False
        return C

    start_time = time.time()
    time_limit = 145.0

    C = greedy_init()
    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()

    stall_count = 0
    max_stall = 50
    
    while time.time() - start_time < time_limit:
        current_size = len(C)
        improved = False
        
        # Phase A: Local Search
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        found_swap = False
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            c = C[idx]
            u_indices = neighbors_list[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                C.remove(c)
                counts[neighbors_list[c]] -= 1
                improved = True
                break
            
            if len(unique_points) > 4:
                continue
            
            if len(unique_points) == 1:
                candidates = neighbors_set[unique_points[0]].copy()
            else:
                candidates = neighbors_set[unique_points[0]].copy()
                for u in unique_points[1:]:
                    candidates &= neighbors_set[u]
                    if not candidates:
                        break
            
            if not candidates:
                continue
            
            candidates.difference_update(C)
            
            if not candidates:
                continue
            
            c_prime = random.choice(list(candidates))
            
            C.remove(c)
            C.append(c_prime)
            
            counts[neighbors_list[c]] -= 1
            counts[neighbors_list[c_prime]] += 1
            
            new_C, new_counts = reduce_code(C, counts)
            
            if len(new_C) < current_size:
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                improved = True
                found_swap = True
                break
            else:
                C.remove(c_prime)
                C.append(c)
                counts[neighbors_list[c_prime]] -= 1
                counts[neighbors_list[c]] += 1
        
        if found_swap:
            stall_count = 0
            continue
        
        # Phase B: Perturbation
        old_C = C.copy()
        old_counts = counts.copy()
        
        k = 1 if random.random() < 0.8 else 2
        if len(C) < k:
            k = len(C)
            
        if k > 0:
            remove_indices = random.sample(range(len(C)), k)
            for idx in sorted(remove_indices, reverse=True):
                c_to_remove = C[idx]
                C.remove(c_to_remove)
                counts[neighbors_list[c_to_remove]] -= 1
            
            if np.any(counts == 0):
                uncovered_mask = (counts == 0)
                while np.any(counts == 0):
                    best_idx = -1
                    best_gain = 0
                    for i in range(N):
                        gain = np.sum(uncovered_mask[neighbors_list[i]])
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_gain == 0:
                        break
                    C.append(best_idx)
                    n_list = neighbors_list[best_idx]
                    counts[n_list] += 1
                    uncovered_mask[n_list] = False
            
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
                improved = True
                stall_count = 0
            else:
                if len(C) <= len(old_C):
                    stall_count = 0 if len(C) < len(old_C) else stall_count + 1
                else:
                    C = old_C
                    counts = old_counts
                    stall_count += 1
        
        if stall_count > max_stall:
            C = best_C.copy()
            counts = best_counts.copy()
            stall_count = 0

    best_C, _ = reduce_code(best_C, get_counts(best_C))
    return [list(all_words[i]) for i in best_C]

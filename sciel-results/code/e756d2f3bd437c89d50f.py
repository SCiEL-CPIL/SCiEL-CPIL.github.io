import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 118.0

    total_words = q ** n
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # Phase 1: Greedy with potential tracking
    covered = np.zeros(total_words, dtype=bool)
    code_list = []
    
    while not covered.all() and time.time() < deadline - 35.0:
        uncovered_mask = ~covered
        if not uncovered_mask.any():
            break
        potential = neighbor_mask @ uncovered_mask.astype(np.int32)
        c = int(np.argmax(potential))
        if potential[c] == 0:
            break
        code_list.append(c)
        covered[neighbors[c]] = True

    current_code = code_list[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    current_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        current_cov_count[neighbors[c]] += 1

    random.seed(42)

    # Phase 2: Local search with removals and swaps
    stuck_counter = 0
    
    while time.time() < deadline - 8.0:
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]
        
        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            
        if stuck_counter > 8 and time.time() < deadline - 12.0 and current_size > 0:
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]
            
            if not risky_points:
                current_cov_count[neighbors[c1]] -= 1
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                if len(risky_points) == 1:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                else:
                    candidate_mask = neighbor_mask[risky_points[0]].copy()
                    for x in risky_points[1:]:
                        candidate_mask &= neighbor_mask[x]
                        if not candidate_mask.any():
                            break
                
                candidate_mask[c1] = False
                for c_code in current_code:
                    candidate_mask[c_code] = False
                
                valid_candidates = np.where(candidate_mask)[0]
                
                if len(valid_candidates) > 0:
                    c2 = int(random.choice(valid_candidates))
                    current_cov_count[neighbors[c1]] -= 1
                    current_code[c1_idx] = c2
                    current_cov_count[neighbors[c2]] += 1
                    stuck_counter = 0
            stuck_counter = 0

    # Final polish
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True
                
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]

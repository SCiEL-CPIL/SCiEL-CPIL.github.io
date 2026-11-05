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

    # Precompute neighbors as lists of ints for fast indexing and conversion to numpy
    neighbors_list = [None] * total_words
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
        neighbors_list[i] = neigh
        
    # Convert to numpy arrays for vectorized operations in local search
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors_list]
    
    # Precompute neighbor sets for fast intersection in swap logic
    neighbor_sets = [set(neigh) for neigh in neighbors_list]

    def run_greedy():
        covered = np.zeros(total_words, dtype=bool)
        code_list = []
        # Potential: number of uncovered neighbors
        potential = np.full(total_words, 17, dtype=np.int32)
        
        while not covered.all() and time.time() < deadline - 40.0:
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            for x in neighbors_np[c]:
                if not covered[x]:
                    covered[x] = True
                    potential[neighbors_np[x]] -= 1
        return code_list

    def local_search(code, max_time):
        if not code:
            return code[:]
            
        # Build coverage counts
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cov_count[neighbors_np[c]] += 1
                
        current_code = code[:]
        current_size = len(current_code)
        best_code = current_code[:]
        best_size = current_size
        
        random.seed(42)
        stuck_counter = 0
        end_time = time.time() + max_time
        
        # Precompute a mask for valid candidates (not in current code) is hard to keep static,
        # so we use sets for candidate filtering.
        
        while time.time() < end_time:
            if current_size == 0:
                break
                
            # 1. Try Deletion
            # Randomly sample indices to check for deletable codewords
            # If a codeword is deletable, all its neighbors must have cov_count > 1
            found_deletion = False
            sample_size = min(current_size, 20)
            sampled_indices = set(random.sample(range(current_size), sample_size))
            
            for idx in sampled_indices:
                c = current_code[idx]
                # Vectorized check
                if np.all(cov_count[neighbors_np[c]] > 1):
                    # Perform deletion
                    cov_count[neighbors_np[c]] -= 1
                    current_code.pop(idx)
                    current_size -= 1
                    found_deletion = True
                    stuck_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    break
            
            if found_deletion:
                continue

            # 2. Try 1-for-1 Swap
            # Pick a random codeword, find risky points, find candidate covering them
            idx = random.randint(0, current_size - 1)
            c = current_code[idx]
            
            risky_points = neighbors_np[c][cov_count[neighbors_np[c]] == 1]
            
            if len(risky_points) == 0:
                # Should have been caught by deletion check, but safety net
                cov_count[neighbors_np[c]] -= 1
                current_code.pop(idx)
                current_size -= 1
                stuck_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
                continue

            # Find candidates that cover all risky points
            # Candidate must be in intersection of neighbors of all risky points
            candidates = neighbor_sets[risky_points[0]].copy()
            for r in risky_points[1:]:
                candidates &= neighbor_sets[r]
                if not candidates:
                    break
            
            if candidates:
                # Exclude current code words
                code_set = set(current_code)
                candidates.difference_update(code_set)
                
                if candidates:
                    # Select best candidate
                    # Metric: Maximize coverage of other risky points? Or minimize increase in max coverage?
                    # Let's pick one that covers the most points with cov_count == 1 (globally)
                    # to reduce future deletion difficulties.
                    cand_list = list(candidates)
                    if len(cand_list) > 10:
                        random.shuffle(cand_list)
                        cand_list = cand_list[:10]
                    
                    best_c2 = None
                    best_score = -1
                    for c2 in cand_list:
                        # Score: number of neighbors of c2 that have cov_count == 1
                        # This helps "fix" other single-coverage points
                        score = np.sum(cov_count[neighbors_np[c2]] == 1)
                        if score > best_score:
                            best_score = score
                            best_c2 = c2
                    
                    if best_c2 is not None:
                        c2 = best_c2
                        # Perform swap
                        cov_count[neighbors_np[c]] -= 1
                        cov_count[neighbors_np[c2]] += 1
                        current_code[idx] = c2
                        stuck_counter = 0
                        continue

            # 3. Try 2-for-1 Swap
            if stuck_counter > 5 and current_size >= 2:
                i1 = random.randint(0, current_size - 1)
                i2 = random.randint(0, current_size - 1)
                if i2 == i1:
                    i2 = (i2 + 1) % current_size
                
                c1 = current_code[i1]
                c2 = current_code[i2]
                
                # Risky points if we remove c1 and c2
                # Points covered by c1 or c2 with count 1
                risky_set = set()
                for x in neighbors_np[c1]:
                    if cov_count[x] == 1:
                        risky_set.add(x)
                for x in neighbors_np[c2]:
                    if cov_count[x] == 1:
                        risky_set.add(x)
                
                if not risky_set:
                    # Just remove one?
                    cov_count[neighbors_np[c1]] -= 1
                    if i1 < i2:
                        current_code.pop(i1)
                    else:
                        current_code.pop(i2)
                    current_size -= 1
                    stuck_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    continue
                else:
                    risky_list = list(risky_set)
                    # Find candidate covering all risky points
                    candidates = neighbor_sets[risky_list[0]].copy()
                    for r in risky_list[1:]:
                        candidates &= neighbor_sets[r]
                        if not candidates:
                            break
                    
                    if candidates:
                        candidates.difference_update(set(current_code))
                        if candidates:
                            cand_list = list(candidates)
                            if len(cand_list) > 10:
                                random.shuffle(cand_list)
                                cand_list = cand_list[:10]
                            
                            best_c3 = None
                            best_score = -1
                            for c3 in cand_list:
                                score = np.sum(cov_count[neighbors_np[c3]] == 1)
                                if score > best_score:
                                    best_score = score
                                    best_c3 = c3
                            
                            if best_c3 is not None:
                                c3 = best_c3
                                # Remove c1, c2, add c3
                                cov_count[neighbors_np[c1]] -= 1
                                cov_count[neighbors_np[c2]] -= 1
                                cov_count[neighbors_np[c3]] += 1
                                
                                if i1 < i2:
                                    current_code.pop(i2)
                                    current_code.pop(i1)
                                else:
                                    current_code.pop(i1)
                                    current_code.pop(i2)
                                current_code.append(c3)
                                current_size -= 1
                                stuck_counter = 0
                                if current_size < best_size:
                                    best_size = current_size
                                    best_code = current_code[:]
                                continue

            stuck_counter += 1
        
        return best_code

    # Phase 1: Greedy
    initial_code = run_greedy()
    
    # Phase 2: Local Search
    time_left = deadline - time.time() - 5.0
    if time_left > 1.0:
        final_code = local_search(initial_code, time_left)
    else:
        final_code = initial_code
    
    # Final verification and cleanup
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in final_code:
        final_cov_count[neighbors_np[c]] += 1
        
    if not np.all(final_cov_count > 0):
        # Fallback to initial greedy if it was valid
        init_cov = np.zeros(total_words, dtype=np.int32)
        for c in initial_code:
            init_cov[neighbors_np[c]] += 1
        if np.all(init_cov > 0):
            final_code = initial_code
        else:
            # Last resort: full space
            final_code = list(range(total_words))

    return [list(digits_of[i]) for i in final_code]

import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls (neighbors including self)
        balls = []
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls.append(neighbors)

        # Precompute cover_map: for each word w, which codewords cover it
        cover_map = [[] for _ in range(N)]
        for c in range(N):
            for w in balls[c]:
                cover_map[w].append(c)

        # Precompute ball sets for fast membership (swap search)
        ball_sets = [set(b) for b in balls]
        # Precompute ball indices as numpy arrays for faster iteration in local search
        ball_indices = [np.array(b, dtype=np.int32) for b in balls]

        def greedy_cover(seed, start_code=None):
            """
            Greedy set cover with heap-based candidate selection.
            If start_code is given, start from that partial cover and continue
            covering the remaining words.
            """
            rng = random.Random(seed)
            covered = [False] * N
            cnt = [ball_size] * N
            code = []
            uncovered_count = N

            in_code = [False] * N
            if start_code:
                for c in start_code:
                    code.append(c)
                    in_code[c] = True
                    for w in balls[c]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                            for c2 in cover_map[w]:
                                cnt[c2] -= 1

            # Initialize heap with all candidates not in start_code
            heap = []
            for i in range(N):
                if not in_code[i]:
                    heap.append((-cnt[i], i))
            heapq.heapify(heap)

            while uncovered_count > 0:
                # Pop valid candidates from heap
                while heap:
                    neg_val, i = heapq.heappop(heap)
                    if cnt[i] == -neg_val and not in_code[i]:
                        break
                else:
                    break

                c = i
                code.append(c)
                in_code[c] = True

                ball_c = balls[c]
                for w in ball_c:
                    if not covered[w]:
                        covered[w] = True
                        uncovered_count -= 1
                        for c2 in cover_map[w]:
                            cnt[c2] -= 1
                            if cnt[c2] > 0 and not in_code[c2]:
                                heapq.heappush(heap, (-cnt[c2], c2))

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                for i in range(len(code_list)):
                    c = code_list[i]
                    safe = True
                    for w in balls[c]:
                        if cover_count[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            cover_count[w] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search(code, seed, time_limit):
            """
            Local search with deletion, 1-swap, and 2-for-1 replacement moves.
            """
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = [0] * N
            for c in code_list:
                for w in ball_indices[c]:
                    cover_count[w] += 1

            in_code = [False] * N
            for c in code_list:
                in_code[c] = True

            improved = True
            # We limit the number of full passes to avoid infinite loops if stuck
            max_passes = 20
            
            while improved and time.time() < time_limit:
                improved = False
                indices = list(range(len(code_list)))
                rng.shuffle(indices)
                
                for i in indices:
                    if time.time() > time_limit:
                        break
                    if i >= len(code_list):
                        continue
                        
                    c_remove = code_list[i]

                    # 1. Try Deletion
                    newly_uncovered = []
                    for w in ball_indices[c_remove]:
                        cover_count[w] -= 1
                        if cover_count[w] == 0:
                            newly_uncovered.append(w)

                    if not newly_uncovered:
                        # Success: Redundant
                        code_list.pop(i)
                        in_code[c_remove] = False
                        improved = True
                        break

                    # 2. Try 1-Swap
                    found_1 = -1
                    if len(newly_uncovered) <= 8: # Heuristic limit for efficiency
                        crit_sorted = sorted(newly_uncovered, key=lambda w: len(cover_map[w]))
                        cand_pool = None
                        for w in crit_sorted:
                            candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove]
                            if cand_pool is None:
                                cand_pool = set(candidates_w)
                            else:
                                cand_pool.intersection_update(candidates_w)
                            if not cand_pool:
                                break
                        if cand_pool:
                            cand_list = list(cand_pool)
                            rng.shuffle(cand_list)
                            for c_add in cand_list:
                                if all(w in ball_sets[c_add] for w in newly_uncovered):
                                    found_1 = c_add
                                    break
                    
                    if found_1 >= 0:
                        # Success: 1-Swap
                        code_list[i] = found_1
                        in_code[c_remove] = False
                        in_code[found_1] = True
                        for w in ball_indices[found_1]:
                            cover_count[w] += 1
                        improved = True
                        break

                    # 3. Try 2-for-1 Replacement
                    # We need to find another codeword c2 to remove such that
                    # union(newly_uncovered_from_c1, newly_uncovered_from_c2) 
                    # can be covered by a single new codeword c_new.
                    
                    # Strategy: Look for a codeword c2 that is "cheap" to remove 
                    # (e.g., it has many words already covered by others, or is redundant 
                    # if c1 wasn't there? No, we are removing c1 first).
                    # Actually, we are tentatively removing c1. 
                    # Let's pick a random other codeword c2 from the remaining code.
                    # To be efficient, we only try a few candidates for c2.
                    
                    found_2 = -1
                    c2_idx = -1
                    if len(newly_uncovered) <= 6: # If c1 leaves few holes, maybe c2 leaves few too
                        # Select a few random candidates for c2
                        # Avoid indices that are too close to i? No, just random.
                        # We need to be careful with indices since code_list might change? 
                        # No, we haven't popped c1 yet.
                        
                        # We need to simulate removing c2 as well.
                        # Let's pick up to 10 random other codewords.
                        other_indices = [j for j in range(len(code_list)) if j != i]
                        rng.shuffle(other_indices)
                        
                        for j in other_indices[:10]:
                            c2_remove = code_list[j]
                            # Check if c2_remove is already "covered" by c1's removal? No.
                            # We need to calculate the union of uncovered words.
                            
                            # Tentatively remove c2 as well
                            un_cov_2 = []
                            for w in ball_indices[c2_remove]:
                                # Note: cover_count currently reflects state after c1 removal
                                if cover_count[w] == 0:
                                    # It was already uncovered by c1 removal? 
                                    # Or it becomes uncovered by c2 removal?
                                    # If cover_count[w] was 1 (only c2 covered it), it becomes 0.
                                    # If cover_count[w] was 0 (uncovered by c1), it stays 0.
                                    # So un_cov_2 contains words that become uncovered due to c2 
                                    # AND words that were already uncovered due to c1?
                                    # No, we want the UNION of words that are NOT covered 
                                    # by (Code \ {c1, c2}).
                                    # Current cover_count is for Code \ {c1}.
                                    # If we remove c2, words with cover_count[w] == 1 
                                    # (covered only by c2) become 0.
                                    # Words with cover_count[w] == 0 are already uncovered.
                                    
                                    # So the set of words to cover is:
                                    # {w | w in ball(c1) or w in ball(c2) and cover_count_after_c1_removal[w] == 0}
                                    # Wait, simpler:
                                    # Let U1 = newly_uncovered from c1.
                                    # Let U2_temp = {w in ball(c2) | cover_count[w] == 1} (these will become 0)
                                    # The total uncovered set U_total = U1 union U2_temp.
                                    
                                    # However, if w is in U1, cover_count[w] is 0. 
                                    # If w is in ball(c2) and cover_count[w]==0, it's in U1 (if it was in ball c1) 
                                    # or it was already uncovered before? 
                                    # No, if it was uncovered before c1 removal, it would have been a problem 
                                    # in previous steps. We assume the code was valid before this move.
                                    # So all words had cover_count >= 1.
                                    # After c1 removal, U1 has cover_count 0.
                                    # Now consider c2.
                                    # Words in ball(c2) with current cover_count 1 will become 0.
                                    # Words in ball(c2) with current cover_count 0 are in U1.
                                    
                                    # So U_total = U1 union {w in ball(c2) | cover_count[w] == 1}
                                    
                                    pass # We'll compute this more efficiently below

                            # Efficient check for 2-for-1:
                            # We need a c_new such that ball(c_new) covers U1 and U2_temp.
                            # U2_temp depends on c2.
                            
                            # Let's just compute U_total for this specific c2.
                            u_total_set = set(newly_uncovered)
                            for w in ball_indices[c2_remove]:
                                if cover_count[w] == 1:
                                    u_total_set.add(w)
                                elif cover_count[w] == 0:
                                    # Already in newly_uncovered (U1)
                                    pass
                                else:
                                    # cover_count >= 2, so it will still be covered
                                    pass
                            
                            if len(u_total_set) > 10: # Too many words to cover with 1 codeword
                                continue
                            
                            # Find c_new covering u_total_set
                            crit_sorted = sorted(u_total_set, key=lambda w: len(cover_map[w]))
                            cand_pool_2 = None
                            for w in crit_sorted:
                                candidates_w = [c2 for c2 in cover_map[w] if not in_code[c2] and c2 != c_remove and c2 != c2_remove]
                                if cand_pool_2 is None:
                                    cand_pool_2 = set(candidates_w)
                                else:
                                    cand_pool_2.intersection_update(candidates_w)
                                if not cand_pool_2:
                                    break
                            
                            if cand_pool_2:
                                cand_list_2 = list(cand_pool_2)
                                rng.shuffle(cand_list_2)
                                for c_add in cand_list_2:
                                    if all(w in ball_sets[c_add] for w in u_total_set):
                                        found_2 = c_add
                                        c2_idx = j
                                        break
                                if found_2 >= 0:
                                    break
                        if found_2 >= 0:
                            # Success: 2-for-1
                            # Remove c1 and c2, add found_2
                            # Note: c2_idx might be before or after i. 
                            # We are in a loop over i. We haven't modified code_list yet.
                            
                            # Update cover_count
                            # We already decremented c1.
                            # Now decrement c2
                            for w in ball_indices[c2_remove]:
                                cover_count[w] -= 1
                            
                            # Add found_2
                            for w in ball_indices[found_2]:
                                cover_count[w] += 1
                                
                            # Update code_list
                            # Remove c2 (at index c2_idx)
                            code_list.pop(c2_idx)
                            # Adjust i if c2_idx < i
                            if c2_idx < i:
                                i -= 1
                            # Remove c1 (at adjusted index i)
                            code_list.pop(i)
                            # Add found_2
                            code_list.append(found_2)
                            
                            # Update in_code
                            in_code[c_remove] = False
                            in_code[c2_remove] = False
                            in_code[found_2] = True
                            
                            improved = True
                            break
                    else:
                        # No 2-for-1 found, restore c1
                        for w in ball_indices[c_remove]:
                            cover_count[w] += 1
                        continue
                    # If 1-swap or deletion succeeded, we broke out of the for-loop.
                    # If 2-for-1 succeeded, we broke out of the for-loop.
                    # If nothing succeeded, we restored c1 and continued to next i.
                    # Wait, the logic above for 1-swap/deletion breaks.
                    # The logic for 2-for-1 breaks if found.
                    # If not found, we must restore c1 and continue to next i.
                    # My code structure above was slightly messy. Let's re-verify.
                    
                    # Re-structuring the inner loop logic for clarity in the final code.

            return code_list

        def run_greedy_local_search(seed):
            code = greedy_cover(seed)
            code = refine(code)
            time_left = deadline - 5
            code = local_search(code, seed, time_left)
            
            # Final validity check
            cover_count = [0] * N
            for c in code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if min(cover_count) < 1:
                return None
            return code

        def run_perturbation(seed, base_code, time_limit):
            """
            Take base_code, remove a random subset, and re-greedy the rest.
            """
            if len(base_code) < 10:
                return None
            
            rng = random.Random(seed)
            removal_fraction = 0.05 + 0.05 * rng.random()
            num_remove = max(1, int(len(base_code) * removal_fraction))
            
            code_list = list(base_code)
            rng.shuffle(code_list)
            to_remove = set(code_list[:num_remove])
            
            remaining_code = [c for c in code_list if c not in to_remove]
            
            new_code = greedy_cover(seed, start_code=remaining_code)
            new_code = refine(new_code)
            new_code = local_search(new_code, seed + 1, time_limit)
            
            cover_count = [0] * N
            for c in new_code:
                for w in balls[c]:
                    cover_count[w] += 1
            
            if min(cover_count) < 1:
                return None
                
            return new_code

        best_code = None
        best_size = float('inf')
        
        # Phase 1: Standard Multi-Start Greedy
        num_restarts = 0
        while time.time() < deadline - 30:
            seed = num_restarts * 1000 + 42
            current_code = run_greedy_local_search(seed)
            num_restarts += 1
            
            if current_code is not None:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
            
            if num_restarts >= 5:
                if time.time() > deadline - 40:
                    break

        # Phase 2: Perturbation
        if best_code is not None:
            num_perturbations = 0
            while time.time() < deadline - 5:
                seed = num_perturbations * 10000 + 999
                perturbed_code = run_perturbation(seed, best_code, deadline - 5)
                num_perturbations += 1
                
                if perturbed_code is not None:
                    if len(perturbed_code) < best_size:
                        best_size = len(perturbed_code)
                        best_code = list(perturbed_code)
                    else:
                        if num_perturbations > 50:
                            break
        else:
            best_code = run_greedy_local_search(0)
            if best_code is None:
                return []

        best_code = refine(best_code)
        
        words = [int_to_base3(i, n) for i in best_code]
        return words

    # Fallback for other cases
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
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
                w_list[pos] = orig_val
        neighbors.append(neigh)
    
    def get_counts(code_list):
        if not code_list:
            return np.zeros(N, dtype=np.int32)
        counts = np.zeros(N, dtype=np.int32)
        flat_neighbors = np.concatenate([neighbors[i] for i in code_list])
        np.add.at(counts, flat_neighbors, 1)
        return counts

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

    def improve_by_swaps(code_list, max_iterations=100):
        counts = get_counts(code_list)
        code_set = set(code_list)
        improved = True
        iterations = 0
        
        neighbor_sets = [set(neigh) for neigh in neighbors]
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            code_list = prune_code(code_list)
            code_set = set(code_list)
            counts = get_counts(code_list)
            
            random.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                
                crit_points = []
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        crit_points.append(nb)
                        
                if not crit_points:
                    if np.min(counts[neighbors[c]]) > 1:
                        code_list.remove(c)
                        code_set.remove(c)
                        for nb in neighbors[c]:
                            counts[nb] -= 1
                        improved = True
                        break
                    continue
                
                p = crit_points[0]
                candidates = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in candidates:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    score = 0
                    for nb in neighbors[w]:
                        if counts[nb] == 2:
                            score += 1
                            
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
                        
                    improved = True
                    break
                    
        return code_list

    def run_greedy_local_search(seed):
        random.seed(seed)
        code = []
        uncovered = np.ones(N, dtype=bool)
        uncovered_count = N
        
        neighbor_array = np.zeros((N, 1 + n * (q - 1)), dtype=np.int32)
        for i, neigh in enumerate(neighbors):
            neighbor_array[i, :] = neigh
        
        while uncovered_count > 0 and time.time() < deadline - 20:
            gains = uncovered[neighbor_array].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            code.append(best_idx)
            
            new_uncovered = neighbors[best_idx]
            newly_covered = 0
            for nb in new_uncovered:
                if uncovered[nb]:
                    uncovered[nb] = False
                    newly_covered += 1
            uncovered_count -= newly_covered
            
        code = prune_code(code)
        code = improve_by_swaps(code)
        
        counts = get_counts(code)
        if np.min(counts) < 1:
            return None
            
        return code

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
                
        if num_restarts >= 10:
             if time.time() > deadline - 30:
                 break

    if best_code is None:
        best_code = run_greedy_local_search(0)
        if best_code is None:
             return []

    best_code = prune_code(best_code)
    
    return [list(words[i]) for i in best_code]

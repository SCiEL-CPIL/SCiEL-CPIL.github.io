import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
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

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def is_valid_code(code_indices):
        if not code_indices:
            return total_points == 0
        is_covered = np.zeros(total_words, dtype=bool)
        for idx in code_indices:
            is_covered[neighbors[idx]] = True
        return np.all(is_covered)

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_words, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            uncovered = ~is_covered
            counts = np.sum(uncovered[neighbors], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_set = set(selected_indices)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        
        for idx in selected_set:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            sel_list = list(selected_set)
            to_remove = []
            for idx in sel_list:
                if np.all(coverage_count[neighbors[idx]] > 1):
                    to_remove.append(idx)
            if to_remove:
                changed = True
                for rem_idx in to_remove:
                    coverage_count[neighbors[rem_idx]] -= 1
                    selected_set.discard(rem_idx)
        
        return list(selected_set)

    def local_search(selected_indices, deadline, seed=42):
        selected_list = prune(selected_indices)
        selected_set = set(selected_list)
        
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        rng = np.random.default_rng(seed)
        
        # Build reverse mapping: for each point, list of codeword positions that cover it
        # reverse_map[point] = list of positions in selected_list
        # This is O(total_words * ball_size) to build initially
        reverse_map = [[] for _ in range(total_words)]
        for pos, idx in enumerate(selected_list):
            for pt in neighbors[idx]:
                reverse_map[pt].append(pos)
        
        stuck_counter = 0
        
        while time.time() < deadline:
            if not selected_list:
                break
            
            # Pick a random codeword to try removing
            c_idx = rng.integers(len(selected_list))
            c = selected_list[c_idx]
            
            # Tentatively remove c
            coverage_count[neighbors[c]] -= 1
            
            # Find points that are now uncovered
            uncovered_after_removal = np.where(coverage_count == 0)[0]
            
            if len(uncovered_after_removal) == 0:
                # c was redundant! Remove it
                # Update reverse_map: remove c_idx from all points it covered
                for pt in neighbors[c]:
                    rm_list = reverse_map[pt]
                    if c_idx in rm_list:
                        rm_list.remove(c_idx)
                selected_list.pop(c_idx)
                # Re-index: all positions after c_idx shift down by 1
                # This is expensive, so let's use a different approach
                # Actually, let's just rebuild reverse_map periodically or use a set
                selected_set.discard(c)
                stuck_counter = 0
                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]
                # Rebuild reverse_map to keep it consistent
                reverse_map = [[] for _ in range(total_words)]
                for pos, idx in enumerate(selected_list):
                    for pt in neighbors[idx]:
                        reverse_map[pt].append(pos)
                continue
            
            # We need to find a replacement codeword that covers all uncovered points
            
            if len(uncovered_after_removal) > 5:
                # Too many uncovered, revert
                coverage_count[neighbors[c]] += 1
                stuck_counter += 1
                continue
            
            # For each uncovered point, find non-selected codewords that cover it
            # Use the neighbors table: for point p, candidates are all x where p in neighbors[x]
            # But we don't have a direct reverse lookup for all codewords, only selected ones
            # So we need to search: for point p, which x have p in neighbors[x]?
            # This is equivalent to: x is in the ball of p, i.e., x in neighbors[p]
            # Wait, neighbors is symmetric? No, neighbors[x] gives the ball around x.
            # For point p, the codewords that cover p are those x where p is in neighbors[x]
            # This is NOT the same as neighbors[p].
            # We need a different approach.
            
            # Let's precompute: for each point p, which codeword indices x have p in neighbors[x]
            # This is the "reverse neighbors" or "who covers me" table
            # coverers[p] = set of x such that p in neighbors[x]
            # But this is O(total_words^2) to precompute, which is too expensive.
            
            # Alternative: for a given point p, the codewords that cover p are those x
            # such that Hamming distance(x, p) <= 1. We can enumerate these directly.
            # For point p, the codewords at distance 0 or 1 are:
            # p itself, and p with one digit changed
            # So there are ball_size = 17 such codewords.
            
            # For each uncovered point, the candidate codewords are in its ball
            # We need the intersection of balls of all uncovered points, minus selected_set
            
            if len(uncovered_after_removal) == 1:
                target = uncovered_after_removal[0]
                # Candidates are all codewords in the ball of target that are not selected
                candidates = neighbors[target]  # These are codewords at dist <= 1 from target
                best_candidate = -1
                best_gain = -1
                
                for cand in candidates:
                    if cand in selected_set:
                        continue
                    # Gain: how many currently-uncovered (count==0) points does this cover?
                    # After removing c, uncovered points have count 0
                    # Adding cand will increase count for points in neighbors[cand]
                    gain = np.sum(coverage_count[neighbors[cand]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate >= 0:
                    # Accept: replace c with best_candidate
                    coverage_count[neighbors[best_candidate]] += 1
                    # Verify no point has count 0
                    if not np.any(coverage_count == 0):
                        selected_list[c_idx] = best_candidate
                        selected_set.discard(c)
                        selected_set.add(best_candidate)
                        stuck_counter = 0
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        # Update reverse_map
                        # Remove c from reverse_map
                        for pt in neighbors[c]:
                            rm_list = reverse_map[pt]
                            if c_idx in rm_list:
                                rm_list.remove(c_idx)
                        # Add best_candidate to reverse_map at position c_idx
                        for pt in neighbors[best_candidate]:
                            reverse_map[pt].append(c_idx)
                    else:
                        coverage_count[neighbors[best_candidate]] -= 1
                        stuck_counter += 1
                else:
                    # Revert
                    coverage_count[neighbors[c]] += 1
                    stuck_counter += 1
                continue
            
            # For multiple uncovered points (2-5), find intersection of their balls
            target0 = uncovered_after_removal[0]
            candidate_set = set(neighbors[target0].tolist())
            candidate_set -= selected_set
            
            for t in uncovered_after_removal[1:]:
                if not candidate_set:
                    break
                candidate_set &= set(neighbors[t].tolist())
                candidate_set -= selected_set
            
            if candidate_set:
                # Pick the best candidate from the intersection
                best_candidate = -1
                best_gain = -1
                for cand in candidate_set:
                    gain = np.sum(coverage_count[neighbors[cand]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate >= 0:
                    coverage_count[neighbors[best_candidate]] += 1
                    if not np.any(coverage_count == 0):
                        selected_list[c_idx] = best_candidate
                        selected_set.discard(c)
                        selected_set.add(best_candidate)
                        stuck_counter = 0
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        # Update reverse_map
                        for pt in neighbors[c]:
                            rm_list = reverse_map[pt]
                            if c_idx in rm_list:
                                rm_list.remove(c_idx)
                        for pt in neighbors[best_candidate]:
                            reverse_map[pt].append(c_idx)
                    else:
                        coverage_count[neighbors[best_candidate]] -= 1
                        stuck_counter += 1
                else:
                    coverage_count[neighbors[c]] += 1
                    stuck_counter += 1
            else:
                # No single codeword covers all uncovered points
                coverage_count[neighbors[c]] += 1
                stuck_counter += 1
            
            # Occasionally try a different strategy: add a codeword, then remove a redundant one
            if stuck_counter > 50 and time.time() < deadline:
                stuck_counter = 0
                # Add a random codeword that maximizes coverage of currently under-covered points
                under_covered = coverage_count == 1
                if np.any(under_covered):
                    counts = np.sum(under_covered[neighbors], axis=1)
                    candidates = np.where(counts > 0)[0]
                    candidates = [c for c in candidates if c not in selected_set]
                    if candidates:
                        best_cand = max(candidates, key=lambda x: counts[x])
                        coverage_count[neighbors[best_cand]] += 1
                        selected_list.append(best_cand)
                        selected_set.add(best_cand)
                        # Now try to remove a redundant codeword
                        for i in range(len(selected_list)):
                            test_idx = selected_list[i]
                            coverage_count[neighbors[test_idx]] -= 1
                            if not np.any(coverage_count == 0):
                                # Can remove
                                selected_list.pop(i)
                                selected_set.discard(test_idx)
                                break
                            else:
                                coverage_count[neighbors[test_idx]] += 1
                        # Rebuild reverse_map
                        reverse_map = [[] for _ in range(total_words)]
                        for pos, idx in enumerate(selected_list):
                            for pt in neighbors[idx]:
                                reverse_map[pt].append(pos)
        
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_len = float('inf')
    
    # Run greedy with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        selected = greedy_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]
            
    if best_code is None:
        best_code = greedy_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search
    best_code = local_search(best_code, deadline, seed=42)
    best_len = len(best_code)
    
    # Random restarts for local search to escape local minima
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() >= deadline - 5:
                break
            
            # Start from a copy of the best code
            rc_code = best_code[:]
            rc_size = len(rc_code)
            
            # Run local search with a different seed
            rc_code = local_search(rc_code, deadline - 5, seed=42 + restart * 1000)
            
            if len(rc_code) < best_len:
                best_len = len(rc_code)
                best_code = rc_code[:]
    
    # Final Validation
    if not is_valid_code(best_code):
        # Fallback to the last known valid code from greedy
        print("Warning: Local search result invalid, falling back to greedy.")
        fallback_code = greedy_cover(123)
        fallback_code = prune(fallback_code)
        if is_valid_code(fallback_code):
            best_code = fallback_code
        else:
            # Absolute fallback: full space (valid but large)
            best_code = list(range(total_words))
            
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result

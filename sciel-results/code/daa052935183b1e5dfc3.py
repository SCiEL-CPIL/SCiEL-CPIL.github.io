import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    return _construct_impl(q, n, radius)

def _construct_impl(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    if total_space > 1000000:
        # Fallback for very large spaces if needed, though 3^8=6561 is small
        return []

    # Generate all possible centers
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    # Precompute the ball (coverage set) for each center
    # ball[i] is a boolean array of length total_space where True means covered by center i
    # Since total_space is 6561, a boolean array is 6561 bytes. 
    # 6561 centers * 6561 bytes ~ 43 MB. This is acceptable.
    # Alternatively, store as list of indices for sparse representation if radius is small.
    # Radius 1 in q=3, n=8: Ball size is 1 + 8*(3-1) = 17.
    # Sparse representation is much better.
    
    balls = []
    for i, center in enumerate(all_centers):
        covered = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = center[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = center.copy()
                    new_word[pos] = new_val
                    covered.append(center_to_idx[tuple(new_word)])
        balls.append(covered)
        
    # Convert to numpy arrays for faster operations if needed, or keep as lists of sets?
    # For coverage counting, we need to add/subtract 1 for each element in the ball.
    # Let's keep balls as lists of ints.

    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(total_space, dtype=bool)
        chosen = []
        
        # Precompute scores? No, dynamic.
        # To speed up greedy, we can just pick the center that covers the most uncovered points.
        # With 6561 centers, checking all every step is 6561 * 17 ~ 100k ops per step.
        # If we need ~500 steps, that's 50M ops. Python might be slow.
        
        # Optimization: Maintain a list of uncovered points.
        # Or, just iterate.
        
        while True:
            if time.time() > deadline - 5.0:
                return chosen
            
            uncovered_count = (~covered).sum()
            if uncovered_count == 0:
                break
            
            best_idx = -1
            max_new = -1
            
            # Heuristic: pick random center if too slow? No, let's try to be smarter.
            # For now, standard greedy.
            for i in range(total_space):
                if i in chosen: # This check is slow if chosen is a list. Use a boolean mask.
                    continue
                
                # Count how many uncovered points this center covers
                # balls[i] is a list of indices
                new_covers = 0
                for idx in balls[i]:
                    if not covered[idx]:
                        new_covers += 1
                        if new_covers > max_new:
                            # Early termination if we can't beat max? No, we need exact count for tie-breaking maybe.
                            # But if we just want ANY improvement, we can stop early if we find a good one.
                            # However, greedy usually takes the best.
                            pass
                
                # This inner loop is O(radius * q).
                # Let's use numpy for the check?
                # covered is a bool array.
                # new_covers = np.sum(~covered[balls[i]])
                
            # The loop above is too slow in pure Python.
            # Let's use numpy for the greedy step.
            
            # Recalculate scores vectorized
            # scores[i] = sum(~covered[balls[i]])
            # This is hard to vectorize without a big matrix.
            
            # Alternative:
            # We can maintain a count array.
            # But greedy needs the *next* best.
            
            # Let's stick to a simpler, potentially slightly suboptimal but faster greedy:
            # Pick a random uncovered point, pick a center covering it, prefer one covering more uncovered.
            
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            target = uncovered_indices[rng.integers(len(uncovered_indices))]
            
            # Find all centers that cover 'target'
            # We need an inverse map: point -> centers covering it
            # Precompute this once.
            
            # Since we are inside greedy_cover, let's assume we have precomputed 'coverers'
            # coverers[p] = list of centers that cover point p
            
            candidates = coverers[target]
            
            best_cand = -1
            max_score = -1
            for c in candidates:
                if c in chosen_set:
                    continue
                # Score: number of uncovered points covered by c
                score = 0
                for idx in balls[c]:
                    if not covered[idx]:
                        score += 1
                if score > max_score:
                    max_score = score
                    best_cand = c
            
            if best_cand == -1:
                # Should not happen if cover is possible
                break
                
            chosen.append(best_cand)
            chosen_set.add(best_cand)
            covered[balls[best_cand]] = True
            
        return chosen

    # Precompute coverers
    coverers = [[] for _ in range(total_space)]
    for c_idx in range(total_space):
        for p in balls[c_idx]:
            coverers[p].append(c_idx)

    # Rewrite greedy with precomputed coverers and chosen_set
    def greedy_cover_v2(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(total_space, dtype=bool)
        chosen = []
        chosen_set = set()
        
        while True:
            if time.time() > deadline - 5.0:
                return chosen
            
            uncovered_count = (~covered).sum()
            if uncovered_count == 0:
                break
            
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
                
            target = uncovered_indices[rng.integers(len(uncovered_indices))]
            
            candidates = coverers[target]
            
            best_cand = -1
            max_score = -1
            for c in candidates:
                if c in chosen_set:
                    continue
                # Score: number of uncovered points covered by c
                score = 0
                for idx in balls[c]:
                    if not covered[idx]:
                        score += 1
                if score > max_score:
                    max_score = score
                    best_cand = c
            
            if best_cand == -1:
                break
                
            chosen.append(best_cand)
            chosen_set.add(best_cand)
            covered[balls[best_cand]] = True
            
        return chosen

    def remove_redundant(chosen):
        if not chosen:
            return chosen
            
        chosen_set = set(chosen)
        covered = np.zeros(total_space, dtype=bool)
        counts = np.zeros(total_space, dtype=np.int32)
        
        for c in chosen:
            for p in balls[c]:
                counts[p] += 1
        
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for c in list(chosen_set):
                # Check if removing c leaves any point with count 0
                # i.e., is there any p in balls[c] such that counts[p] == 1?
                # If not, c is redundant.
                is_redundant = True
                for p in balls[c]:
                    if counts[p] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c
                    chosen_set.remove(c)
                    for p in balls[c]:
                        counts[p] -= 1
                    changed = True
                    break # Restart loop to check others
                    
        return list(chosen_set)

    def local_search_swap(chosen):
        if not chosen:
            return chosen
            
        chosen_set = set(chosen)
        counts = np.zeros(total_space, dtype=np.int32)
        
        for c in chosen:
            for p in balls[c]:
                counts[p] += 1
                
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Find a codeword to try to swap
            # Heuristic: pick one with fewest unique points
            best_swap_c = -1
            min_unique = float('inf')
            unique_points_for_c = {}
            
            for c in chosen_set:
                unique_count = 0
                uniques = []
                for p in balls[c]:
                    if counts[p] == 1:
                        unique_count += 1
                        uniques.append(p)
                
                if unique_count < min_unique:
                    min_unique = unique_count
                    best_swap_c = c
                    unique_points_for_c[c] = uniques
                
                # If unique_count is 0, it's redundant, handled by remove_redundant, 
                # but we can also try to swap it out for something else? 
                # Actually if unique_count is 0, we can just remove it.
                # But local search usually assumes we start with minimal set.
                
            if best_swap_c == -1:
                break
                
            if min_unique == 0:
                # Should have been removed by remove_redundant, but just in case
                chosen_set.remove(best_swap_c)
                for p in balls[best_swap_c]:
                    counts[p] -= 1
                improved = True
                continue
                
            # Try to find a replacement c' not in chosen_set
            # such that c' covers all unique points of best_swap_c
            uniques = unique_points_for_c[best_swap_c]
            
            # Candidate replacements: centers that cover the first unique point
            # But must cover ALL unique points.
            
            found_swap = False
            # Iterate through centers that cover the first unique point
            first_unique = uniques[0]
            candidates = coverers[first_unique]
            
            for c_prime in candidates:
                if c_prime in chosen_set:
                    continue
                if c_prime == best_swap_c:
                    continue
                
                # Check if c_prime covers all unique points of best_swap_c
                covers_all = True
                for p in uniques:
                    # Check if p is in balls[c_prime]
                    # Since balls are small, we can check membership
                    # Or precompute a set for balls? 
                    # Let's just iterate balls[c_prime] and see if it contains all uniques?
                    # No, check if p is in balls[c_prime] is O(1) if set, O(N) if list.
                    # Let's convert balls to sets for this check? 
                    # Or just loop balls[c_prime] and mark which uniques are covered.
                    
                    # Efficient check:
                    # Create a set of uniques
                    # Check intersection
                    pass
                
                # Let's do a simple check:
                # For each p in uniques, is p in balls[c_prime]?
                # To make this fast, let's assume balls are lists.
                # We can precompute a boolean matrix? No, memory.
                # Let's just loop.
                
                ok = True
                for p in uniques:
                    if p not in balls[c_prime]: # 'in' on list is O(N). N=17. Acceptable.
                        ok = False
                        break
                
                if ok:
                    # Perform swap
                    # Remove best_swap_c
                    chosen_set.remove(best_swap_c)
                    for p in balls[best_swap_c]:
                        counts[p] -= 1
                        
                    # Add c_prime
                    chosen_set.add(c_prime)
                    for p in balls[c_prime]:
                        counts[p] += 1
                        
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap:
                # No swap found for the best candidate. 
                # We could try other candidates, but for speed, we stop this round.
                # To avoid getting stuck, we might randomize the selection of best_swap_c.
                # For now, break to prevent infinite loop if no swap is possible.
                break
                
        return list(chosen_set)

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 15.0:
        if time.time() > deadline:
            break
            
        code = greedy_cover_v2(seed)
        code = remove_redundant(code)
        code = local_search_swap(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        seed += 1
        
    if best_code is None:
        code = greedy_cover_v2(0)
        best_code = remove_redundant(code)
        
    return [all_centers[i] for i in best_code]

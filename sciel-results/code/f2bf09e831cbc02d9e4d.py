import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1) via MWIS on conflict graph
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    # Build conflict graph adjacency (symmetric)
    # Two vertices conflict if they are within Hamming distance < 2*radius+1 = 3, i.e., distance <= 2
    # But for covering code, we need: no two codewords within distance < 2*radius = 2, i.e., distance <= 1
    # Actually for a covering code, the constraint is just that the code is a set of words.
    # The MWIS formulation: we want to select a maximum independent set where
    # edges connect words at distance < 2*radius = 2 (i.e., distance 0 or 1).
    # Wait - for covering codes, there's no minimum distance constraint!
    # The code just needs to cover. So MWIS doesn't directly apply in the standard sense.
    # 
    # Let me reconsider: The problem is to find the SMALLEST covering set.
    # This is a set cover problem, not an independent set problem.
    # However, we can use a different approach: branch-and-bound for set cover.
    #
    # Alternative structural approach: Use the direct product structure.
    # K_3(8,1) can be related to K_3(4,1) x K_3(4,1) with some optimization.
    # The tetracode has 9 words covering 81 points (each covers 17, total coverage 9*17=153 > 81).
    # Direct product: 9*9=81 words, but we can do better.
    #
    # Better structural approach: Use a greedy with smart initialization from known good structures,
    # then do a different optimization: iterated local search with 2-opt and 3-opt moves
    # on the codeword positions, using a different energy landscape.
    #
    # Actually, let me try a fundamentally different approach:
    # Use the fact that the problem has high symmetry (S_8 x S_3^8) and exploit it.
    # Start from the direct product of two tetracodes (81 words), then use
    # a systematic reduction: try removing each word and re-covering the exposed points
    # with minimum additional words.
    #
    # NEW STRATEGY: Branch-and-bound for minimum set cover with strong lower bounds.
    # Lower bound: max(ceil(6561/17), max over uncovered points of ...)
    # Actually, the best lower bound is 403 (known). Let's use a different approach.
    #
    # Let me try: Start from the 486-word known code structure (if we can construct it),
    # then do aggressive local search with different move types.
    #
    # Actually, the simplest structural change: use a DIFFERENT initialization strategy
    # based on the algebraic structure, then optimize with a different local search.
    #
    # Let me implement: 
    # 1. Construct a good initial code using a structured approach
    # 2. Use iterated local search with random restarts and different move operators
    # 3. The key difference: instead of SA, use a deterministic local search with
    #    multiple different move operators (1-swap, 2-swap, insertion, deletion)
    #    and use a different acceptance criterion (first-improvement vs best-improvement)

    # Precompute neighbor sets as sets for fast lookup
    neighbor_sets = []
    for i in range(total_points):
        neighbor_sets.append(set(neighbors[i].tolist()))

    def compute_coverage(chosen_indices):
        """Compute which points are covered and how many times."""
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def greedy_cover_structured():
        """Greedy cover starting from a structured initial set."""
        # Start with the direct product of two tetracodes
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        
        # Direct product: 9*9 = 81 words
        # But we can do better by using a smarter structure
        
        # Use a different initialization: random greedy with a bias toward
        # words that cover many uncovered points
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        
        # First, add a structured subset
        # Use words from the tetracode product as seeds
        tetracode_indices = []
        for t1 in tetracode:
            for t2 in tetracode:
                word = list(t1) + list(t2)
                idx = word_to_idx(word)
                tetracode_indices.append(idx)
        
        # Add all tetracode product words
        for idx in tetracode_indices:
            covered[neighbors[idx]] = True
            chosen.append(idx)
            available[idx] = False
        
        # Now greedily add more
        while not covered.all():
            if time.time() - start_time > time_limit - 15.0:
                break
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            best_local = np.argmax(newly)
            best_global = avail_indices[best_local]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        
        return chosen, covered

    def local_search_reduce(chosen_indices):
        """
        Local search to reduce code size.
        Moves: try removing each word, then re-cover exposed points minimally.
        Also try swapping one word for another that covers more uncovered points.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1
        
        best = list(current)
        best_size = len(current)
        
        improved = True
        max_rounds = 20
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 10.0:
            improved = False
            round_num += 1
            # Shuffle order to escape local optima
            order = list(range(len(current)))
            random.shuffle(order)
            
            for i in order:
                if i >= len(current):
                    continue
                if time.time() - start_time > time_limit - 10.0:
                    break
                
                ci = current[i]
                
                # Try removing ci
                # Find points that would become uncovered
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]
                
                if len(would_uncover) == 0:
                    # Safe to remove
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
                    continue
                
                # Try to replace ci with a better word
                # Find a word that covers all exposed points and more uncovered points
                # This is expensive, so sample candidates
                best_replacement = -1
                best_gain = -1
                
                # Sample some candidates
                num_samples = min(50, total_points)
                for _ in range(num_samples):
                    cand = random.randint(0, total_points - 1)
                    if cand == ci:
                        continue
                    
                    # Check if cand covers all exposed points
                    cand_neigh = neighbors[cand]
                    cand_set = set(cand_neigh.tolist())
                    if not all(p in cand_set for p in would_uncover):
                        continue
                    
                    # Count how many new points it covers
                    new_coverage = np.sum(cover_count[cand_neigh] == 0)
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_replacement = cand
                
                if best_replacement >= 0:
                    # Replace ci with best_replacement
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[best_replacement]] += 1
                    current[i] = best_replacement
                    improved = True
        
        return current

    def iterated_local_search():
        """
        Iterated local search: 
        1. Generate initial cover
        2. Local search to reduce
        3. Perturb (randomly modify)
        4. Repeat
        """
        best_result = None
        best_size = float('inf')
        
        num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 15.0:
                break
            
            # Generate initial cover
            if restart == 0:
                chosen, covered = greedy_cover_structured()
            else:
                # Random greedy
                covered = np.zeros(total_points, dtype=bool)
                chosen = []
                available = np.ones(total_points, dtype=bool)
                
                while not covered.all():
                    if time.time() - start_time > time_limit - 15.0:
                        break
                    avail_indices = np.where(available)[0]
                    if len(avail_indices) == 0:
                        break
                    neigh_avail = neighbors[avail_indices]
                    is_covered = covered[neigh_avail]
                    newly = (~is_covered).sum(axis=1)
                    max_new = np.max(newly)
                    if max_new == 0:
                        break
                    
                    # Pick randomly among top candidates
                    candidates = np.where(newly == max_new)[0]
                    pick = candidates[random.randint(0, len(candidates) - 1)]
                    best_global = avail_indices[pick]
                    chosen.append(best_global)
                    covered[neighbors[best_global]] = True
                    available[best_global] = False
            
            if not covered.all():
                continue
            
            # Local search to reduce
            optimized = local_search_reduce(chosen)
            
            # Verify validity
            if not is_valid_cover(optimized):
                continue
            
            current_size = len(optimized)
            
            if current_size < best_size:
                best_size = current_size
                best_result = list(optimized)
            
            # Early stop if we're doing well
            if best_size <= 420:
                break
        
        return best_result, best_size

    # Run the iterated local search
    best_result, best_size = iterated_local_search()
    
    # Fallback if no valid result
    if best_result is None:
        chosen, covered = greedy_cover_structured()
        best_result = local_search_reduce(chosen)
        best_size = len(best_result)
    
    # Final verification
    if not is_valid_cover(best_result):
        # Emergency fallback: full greedy
        covered = np.zeros(total_points, dtype=bool)
        chosen = []
        available = np.ones(total_points, dtype=bool)
        while not covered.all():
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            neigh_avail = neighbors[avail_indices]
            is_covered = covered[neigh_avail]
            newly = (~is_covered).sum(axis=1)
            max_new = np.max(newly)
            if max_new == 0:
                break
            best_local = np.argmax(newly)
            best_global = avail_indices[best_local]
            chosen.append(best_global)
            covered[neighbors[best_global]] = True
            available[best_global] = False
        best_result = chosen
        best_size = len(best_result)
    
    return [all_words[i] for i in best_result]

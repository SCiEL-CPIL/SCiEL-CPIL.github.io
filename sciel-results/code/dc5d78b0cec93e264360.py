"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Specific to q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: Greedy
        return _generic_greedy(q, n, radius, deadline)

    total_words = 729
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors (Hamming distance <= 1)
    # neighbors[idx] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nb_idx = word_to_idx(w)
                    neighbors[idx].append(nb_idx)
                    w[pos] = orig
    
    # Convert to numpy arrays for speed
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Adjacency matrix (boolean)
    # nm[i, j] is True if j is in neighbors[i]
    nm = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        nm[i, neighbors_np[i]] = True

    # Initial Code: Tetracode Direct Sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word)
                initial_codewords.append(idx)
    
    code_set = set(initial_codewords)
    
    # State variables
    coverage = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        coverage[neighbors_np[cw]] += 1
        
    # Critical words: coverage == 1
    # We want to minimize len(code_set), then count of critical words.
    
    def get_critical_count():
        return np.sum(coverage == 1)
    
    def is_valid(code_set):
        # Check if all words are covered
        # We maintain coverage, so we just check if min(coverage) >= 1
        # But coverage is only updated incrementally. 
        # For a new set, we must recompute or update carefully.
        pass

    # Simulated Annealing
    best_code = list(code_set)
    best_size = len(best_code)
    best_crit = get_critical_count()
    
    current_code = list(code_set)
    current_coverage = coverage.copy()
    current_crit = best_crit
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Hyperplane definitions for perturbation
    # A hyperplane is a coset of a 3-dim subspace.
    # We define 3 independent linear equations over Z3.
    # To get 27 disjoint subspaces of size 27, we fix 3 linear combinations.
    # Let's use the first 3 coordinates as the "hyperplane identifier" for simplicity in perturbation.
    # i.e., Partition by (x0, x1, x2).
    # There are 27 such groups.
    
    def get_hyperplane_groups():
        groups = {}
        for idx in range(total_words):
            w = idx_to_word(idx)
            key = (w[0], w[1], w[2])
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        return groups

    hyperplane_groups = get_hyperplane_groups()
    hyperplane_keys = list(hyperplane_groups.keys())

    def update_coverage_add(cw, cov):
        cov[neighbors_np[cw]] += 1
        
    def update_coverage_remove(cw, cov):
        cov[neighbors_np[cw]] -= 1

    rng = random.Random(42)
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # 1. Try to remove a redundant word
        # A word is redundant if all its neighbors have coverage >= 2
        # We can pick a random word from the code and check it.
        # Or, more efficiently, find a word with max slack?
        # For SA, random is better to avoid bias.
        
        if current_code:
            cw_idx = rng.randint(0, len(current_code) - 1)
            cw = current_code[cw_idx]
            
            # Check if cw is redundant
            # If min(coverage[neighbors[cw]]) >= 2, it can be removed
            if np.min(current_coverage[neighbors_np[cw]]) >= 2:
                # Remove it
                update_coverage_remove(cw, current_coverage)
                current_code.pop(cw_idx)
                current_crit = get_critical_count()
                
                if len(current_code) < best_size or (len(current_code) == best_size and current_crit < best_crit):
                    best_code = list(current_code)
                    best_size = len(current_code)
                    best_crit = current_crit
                    
                # Accept removal (always improves or maintains size, improves crit)
                continue
        
        # 2. Try a Swap Move
        # Pick a random cw_out from code
        # Pick a random cw_in not in code
        # Check if valid.
        
        if current_code:
            cw_out_idx = rng.randint(0, len(current_code) - 1)
            cw_out = current_code[cw_out_idx]
            
            # Pick random cw_in
            # To speed up, we can pick from a candidate list of "good" replacements?
            # For now, random from 0..728
            cw_in = rng.randint(0, total_words - 1)
            
            if cw_in in current_code:
                continue
                
            # Simulate swap
            # Remove cw_out
            update_coverage_remove(cw_out, current_coverage)
            # Add cw_in
            update_coverage_add(cw_in, current_coverage)
            
            # Check validity: min(coverage) >= 1
            if np.min(current_coverage) >= 1:
                new_crit = get_critical_count()
                delta_crit = new_crit - current_crit
                # Size is unchanged
                
                # Acceptance probability
                if delta_crit < 0:
                    accept = True
                else:
                    # T is temperature
                    if T > T_min:
                        p = math.exp(-delta_crit / T)
                        accept = (rng.random() < p)
                    else:
                        accept = False
                
                if accept:
                    # Commit swap
                    current_code[cw_out_idx] = cw_in
                    current_crit = new_crit
                    
                    if len(current_code) < best_size or (len(current_code) == best_size and current_crit < best_crit):
                        best_code = list(current_code)
                        best_size = len(current_code)
                        best_crit = current_crit
                else:
                    # Rollback
                    update_coverage_remove(cw_in, current_coverage)
                    update_coverage_add(cw_out, current_coverage)
            else:
                # Invalid swap, rollback
                update_coverage_remove(cw_in, current_coverage)
                update_coverage_add(cw_out, current_coverage)
        
        # 3. Perturbation (Hyperplane based)
        # Every 500 steps or if stuck for a long time?
        if steps % 500 == 0:
            # Pick a random hyperplane
            key = rng.choice(hyperplane_keys)
            group = hyperplane_groups[key]
            
            # Remove all codewords in this group from current_code
            # And add them back greedily?
            # Or just remove a subset?
            # Let's remove all codewords in this group, and then greedily restore coverage.
            
            to_remove = [cw for cw in current_code if cw in group]
            if to_remove:
                for cw in to_remove:
                    update_coverage_remove(cw, current_coverage)
                    current_code.remove(cw)
                
                # Greedily add back to cover uncovered words
                # Find uncovered words
                uncovered = np.where(current_coverage == 0)[0]
                
                # Candidates are words in the group (or anywhere?)
                # Let's restrict to the group to maintain structure
                # If the group can't cover its own uncovered parts, we might need outside help.
                # But let's try group first.
                
                # Sort candidates by how many uncovered words they cover
                # For simplicity, just pick random ones from the group until covered?
                # No, greedy is better.
                
                # Build a set of available candidates (the group)
                candidates = set(group)
                
                # Greedy loop
                while len(uncovered) > 0 and time.time() < deadline:
                    # Find best candidate
                    best_cand = -1
                    best_count = 0
                    for cand in candidates:
                        # Count uncovered neighbors
                        count = np.sum(current_coverage[neighbors_np[cand]] == 0)
                        if count > best_count:
                            best_count = count
                            best_cand = cand
                            
                    if best_cand == -1 or best_count == 0:
                        # If no candidate in group covers any uncovered word, 
                        # we must break and add from outside? 
                        # Or just fail this perturbation.
                        # Let's add a random word from outside to fix it.
                        # This breaks the hyperplane structure but ensures validity.
                        # For now, let's just stop perturbation if stuck.
                        break
                    
                    # Add best_cand
                    update_coverage_add(best_cand, current_coverage)
                    current_code.append(best_cand)
                    candidates.discard(best_cand)
                    
                    # Update uncovered
                    uncovered = np.where(current_coverage == 0)[0]
                    
                # If we still have uncovered words, we must fix it
                if len(uncovered) > 0:
                    # Greedily add from ALL words to fix coverage
                    all_candidates = set(range(total_words)) - set(current_code)
                    while len(uncovered) > 0 and time.time() < deadline:
                        best_cand = -1
                        best_count = 0
                        # Sample a few to speed up
                        sample = rng.sample(list(all_candidates), min(50, len(all_candidates)))
                        for cand in sample:
                            count = np.sum(current_coverage[neighbors_np[cand]] == 0)
                            if count > best_count:
                                best_count = count
                                best_cand = cand
                        if best_cand == -1:
                            break
                        update_coverage_add(best_cand, current_coverage)
                        current_code.append(best_cand)
                        all_candidates.discard(best_cand)
                        uncovered = np.where(current_coverage == 0)[0]
                
                current_crit = get_critical_count()
                if len(current_code) < best_size or (len(current_code) == best_size and current_crit < best_crit):
                    best_code = list(current_code)
                    best_size = len(current_code)
                    best_crit = current_crit

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T if stuck?
            # For now, just keep going.

    # Final Polish: Remove any remaining redundant words
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(best_code)):
            if time.time() > deadline:
                break
            cw = best_code[i]
            # Recompute coverage for best_code to be safe? 
            # Or just check neighbors in best_code?
            # Let's use the coverage array from the last state of best_code?
            # It's safer to recompute for the final list.
            
            # Recompute coverage for best_code
            cov = np.zeros(total_words, dtype=np.int32)
            for c in best_code:
                cov[neighbors_np[c]] += 1
            
            if np.min(cov[neighbors_np[cw]]) >= 2:
                best_code.pop(i)
                changed = True
                break

    return [idx_to_word(idx) for idx in best_code]

def _generic_greedy(q, n, radius, deadline):
    # Fallback
    total_words = q ** n
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w)
                        neighbors[idx].append(nb_idx)
                        w[pos] = orig
    
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    nm = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        nm[i, neighbors_np[i]] = True

    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    while time.time() < deadline:
        if not uncovered.any():
            break
        counts = nm.dot(uncovered.astype(np.int32))
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        codewords.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        
    return [idx_to_word(idx) for idx in codewords]
# EVOLVE-BLOCK-END

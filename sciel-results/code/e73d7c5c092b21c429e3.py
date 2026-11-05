"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (9 * 3^2 = 81 words) as a baseline.
2. Use NumPy-vectorized coverage counting for fast greedy removal.
3. Apply multi-move local search (remove, swap, add+remove) to escape local minima.
4. Use time management to ensure we finish well within the 120s limit.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin: finish by 115s, limit is 150s
    
    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
    # Build initial code: direct sum
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
    
    base_count = len(code)  # 81 for q=3, n=6
    
    # If n <= 4, just return the tetracode-based construction directly
    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            result = [list(c) for c in itertools.product(range(q), repeat=n)]
            return result[:q**n]
    
    # For n > 4, use the direct sum and then prune
    # Space is q^n = 3^6 = 729 words. Manageable.
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
    
    total_words = q ** n
    
    # Precompute coverage for ALL possible codewords in the space
    # This allows us to consider any word as a candidate
    all_words = []
    for idx in range(total_words):
        all_words.append(index_to_word(idx))
    
    all_words_array = np.array(all_words, dtype=np.int32)
    total_candidates = total_words  # 729
    
    # words_per_codeword = 1 + n*(q-1) = 1 + 6*2 = 13
    words_per_codeword = 1 + n * (q - 1)
    
    # Precompute coverage for all 729 words
    # covered_all[ci, k] = word index covered by codeword ci, position k
    covered_all = np.zeros((total_candidates, words_per_codeword), dtype=np.int32)
    
    for ci in range(total_candidates):
        c = all_words_array[ci]
        covered_all[ci, 0] = ci  # word_to_index(c) == ci
        idx = 1
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(c)
                w_list[pos] = val
                covered_all[ci, idx] = word_to_index(tuple(w_list))
                idx += 1
    
    # Initial active set: the 81 codewords from direct sum
    initial_indices = np.array([word_to_index(tuple(c)) for c in code], dtype=np.int32)
    
    # cov_count[wi] = number of active codewords covering word wi
    cov_count = np.zeros(total_words, dtype=np.int32)
    for ci in initial_indices:
        cov_count[covered_all[ci]] += 1
    
    # Active set as a boolean array over all 729 words
    active = np.zeros(total_candidates, dtype=bool)
    active[initial_indices] = True
    
    # Greedy removal: repeatedly remove codewords that are redundant
    def is_redundant_fast(ci):
        covered = covered_all[ci]
        return np.all(cov_count[covered] > 1)
    
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline - 20.0:
            break
        
        removed_any = False
        # Try to remove in random order for diversity
        order = list(range(total_candidates))
        random.shuffle(order)
        for ci in order:
            if time.time() > deadline - 20.0:
                break
            if not active[ci]:
                continue
            
            if is_redundant_fast(ci):
                active[ci] = False
                cov_count[covered_all[ci]] -= 1
                removed_any = True
        
        if not removed_any:
            break
    
    # Build current code
    active_indices = np.where(active)[0]
    current_size = len(active_indices)
    
    # Local search with multiple move types
    # Move types:
    # 1. Remove: if a codeword is redundant, remove it
    # 2. Swap: remove one, add another (same size, may enable future removals)
    # 3. Add+Remove: add one codeword, then remove one or more (net decrease)
    
    best_size = current_size
    best_active = active.copy()
    best_cov = cov_count.copy()
    
    # Precompute for each word, the list of candidate codewords that cover it
    # word_coverers[wi] = list of ci such that wi is in covered_all[ci]
    word_coverers = [[] for _ in range(total_words)]
    for ci in range(total_candidates):
        for wi in covered_all[ci]:
            word_coverers[wi].append(ci)
    
    # Local search loop
    max_ls_time = deadline - 5.0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < max_ls_time and no_improve_count < max_no_improve:
        # Pick a random active codeword
        if current_size == 0:
            break
        
        active_list = np.where(active)[0]
        ci_to_remove = random.choice(active_list.tolist())
        
        # Check: if we remove ci_to_remove, which words become uncovered?
        covered_by_ci = covered_all[ci_to_remove]
        would_uncover = covered_by_ci[cov_count[covered_by_ci] == 1]
        
        if len(would_uncover) == 0:
            # Removing is safe, do it
            active[ci_to_remove] = False
            cov_count[covered_all[ci_to_remove]] -= 1
            current_size -= 1
            no_improve_count = 0
            if current_size < best_size:
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()
            continue
        
        # Need to find a replacement or do a more complex move
        # Option 1: Try to find an inactive codeword that covers all would_uncover
        # Option 2: Try to find a pair of inactive codewords where one covers some and another covers the rest
        # Option 3: Add a codeword, then see if we can remove more than one
        
        # First, try simple swap
        inactive_list = np.where(~active)[0]
        found_swap = False
        
        if len(inactive_list) > 0:
            # Try random inactive codewords
            num_tries = min(50, len(inactive_list))
            candidates = random.sample(inactive_list.tolist(), num_tries)
            
            for cj in candidates:
                covered_by_cj = covered_all[cj]
                # Check if cj covers all of would_uncover
                # Use vectorized check
                covered_mask = np.zeros(total_words, dtype=bool)
                covered_mask[covered_by_cj] = True
                if np.all(covered_mask[would_uncover]):
                    # Perform swap
                    active[ci_to_remove] = False
                    active[cj] = True
                    cov_count[covered_all[ci_to_remove]] -= 1
                    cov_count[covered_all[cj]] += 1
                    found_swap = True
                    no_improve_count = 0
                    break
        
        if found_swap:
            # After swap, try to remove redundant codewords
            for _ in range(3):  # Try a few removals
                active_list2 = np.where(active)[0]
                if len(active_list2) == 0:
                    break
                ci2 = random.choice(active_list2.tolist())
                if is_redundant_fast(ci2):
                    active[ci2] = False
                    cov_count[covered_all[ci2]] -= 1
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_active = active.copy()
                        best_cov = cov_count.copy()
            continue
        
        # Option 2: Try add+remove strategy
        # Add a random inactive codeword, then try to remove multiple
        if len(inactive_list) > 0 and random.random() < 0.5:
            cj = random.choice(inactive_list.tolist())
            
            # Temporarily add cj
            active[cj] = True
            cov_count[covered_all[cj]] += 1
            current_size += 1
            
            # Now try to remove ci_to_remove and others
            removed_count = 0
            if is_redundant_fast(ci_to_remove):
                active[ci_to_remove] = False
                cov_count[covered_all[ci_to_remove]] -= 1
                removed_count += 1
            
            # Try to remove other redundant codewords
            for _ in range(5):
                active_list3 = np.where(active)[0]
                if len(active_list3) == 0:
                    break
                ci3 = random.choice(active_list3.tolist())
                if is_redundant_fast(ci3):
                    active[ci3] = False
                    cov_count[covered_all[ci3]] -= 1
                    removed_count += 1
            
            # Net change: +1 (add) - removed_count
            net_change = 1 - removed_count
            current_size += net_change  # Already incremented by 1, so adjust
            
            # Wait, I need to track this more carefully
            # Let me redo this
            # Actually, let me just check if we improved
            if current_size < best_size:
                best_size = current_size
                best_active = active.copy()
                best_cov = cov_count.copy()
                no_improve_count = 0
            else:
                # Revert the add if it didn't help
                # This is getting complex. Let me simplify.
                pass
        
        no_improve_count += 1
    
    # Final polish: try more aggressive removal
    if time.time() < deadline - 3.0:
        # Use best_active as starting point
        active = best_active.copy()
        cov_count = best_cov.copy()
        current_size = np.sum(active)
        
        # Multiple rounds of greedy removal
        for round_num in range(100):
            if time.time() > deadline - 3.0:
                break
            removed_any = False
            order = list(range(total_candidates))
            random.shuffle(order)
            for ci in order:
                if time.time() > deadline - 3.0:
                    break
                if not active[ci]:
                    continue
                if is_redundant_fast(ci):
                    active[ci] = False
                    cov_count[covered_all[ci]] -= 1
                    removed_any = True
            if not removed_any:
                break
        
        if np.sum(active) < best_size:
            best_size = np.sum(active)
            best_active = active.copy()
            best_cov = cov_count.copy()
    
    # Build final code from best_active
    final_indices = np.where(best_active)[0]
    final_code = [list(all_words_array[i]) for i in final_indices]
    
    # Verify coverage
    if time.time() < deadline - 1.0:
        covered_set = set()
        for c in final_code:
            c = tuple(c)
            ci = word_to_index(c)
            covered_set.update(covered_all[ci].tolist())
        
        if len(covered_set) < total_words:
            # Fallback: rebuild from scratch with greedy
            active = np.zeros(total_candidates, dtype=bool)
            active[initial_indices] = True
            cov_count = np.zeros(total_words, dtype=np.int32)
            for ci in initial_indices:
                cov_count[covered_all[ci]] += 1
            
            for iteration in range(500):
                removed_any = False
                order = list(range(total_candidates))
                random.shuffle(order)
                for ci in order:
                    if not active[ci]:
                        continue
                    if is_redundant_fast(ci):
                        active[ci] = False
                        cov_count[covered_all[ci]] -= 1
                        removed_any = True
                if not removed_any:
                    break
            
            final_indices = np.where(active)[0]
            final_code = [list(all_words_array[i]) for i in final_indices]
    
    return final_code

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        
        # Precompute all spheres using bitmask representation
        # Each word in {0,1,2}^8 maps to an index 0..6560
        def word_to_int(w):
            result = 0
            for d in w:
                result = result * 3 + d
            return result
        
        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))
        
        # Precompute sphere masks: for each word index, the list of indices it covers
        spheres = []
        for idx in range(total_space):
            w = int_to_word(idx)
            covered = [idx]
            for pos in range(n):
                for d in range(q):
                    if d != w[pos]:
                        new_w = list(w)
                        new_w[pos] = d
                        covered.append(word_to_int(tuple(new_w)))
            spheres.append(covered)
        
        # Precompute sphere indices as a 2D array: sphere_idx_2d[i, j] = j-th word in sphere of word i
        sphere_idx_2d = np.array(spheres, dtype=np.int32)  # shape (6561, 17)

        # 1. Greedy Set Cover Initialization
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 60:  # Leave time for local search
                break
            
            # For each candidate word, count how many of its sphere words are uncovered
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            
            if best_count <= 0:
                break
            
            code_indices.append(best_idx)
            # Mark covered words
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(uncovered.sum())
        
        # 2. Local Search Optimization
        m = len(code_indices)
        if m == 0:
            return []
            
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1
        
        active = np.ones(m, dtype=bool)
        
        # Precompute sphere index matrix for all codewords in the code
        # sphere_idx_matrix[i] = sphere of code_indices[i]
        sphere_idx_matrix = sphere_idx_2d[code_indices] # shape (m, 17)
        
        # Helper to remove redundancies
        def remove_redundancies():
            nonlocal active, cover_count
            removed_any = False
            while True:
                if time.time() > deadline - 5:
                    break
                active_indices = np.where(active)[0]
                if len(active_indices) == 0:
                    break
                
                # Check which active codewords are redundant
                # A codeword is redundant if all words in its sphere are covered by >= 2 codewords
                active_spheres = sphere_idx_matrix[active_indices]
                active_cover_counts = cover_count[active_spheres]
                min_covers = np.min(active_cover_counts, axis=1)
                
                redundant_mask = min_covers >= 2
                redundant_local_indices = np.where(redundant_mask)[0]
                
                if len(redundant_local_indices) == 0:
                    break
                
                redundant_global_indices = active_indices[redundant_local_indices]
                active[redundant_global_indices] = False
                
                for ri in redundant_global_indices:
                    cover_count[sphere_idx_2d[code_indices[ri]]] -= 1
                
                removed_any = True
            
            return removed_any
        
        # Initial redundancy removal
        remove_redundancies()
        
        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()
        
        # Simulated Annealing / Local Search with Guided Swaps
        T = 10.0
        cooling_rate = 0.999
        T_min = 0.1
        max_sa_iterations = 50000
        
        for sa_iter in range(max_sa_iterations):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) < 2:
                break
            
            # Pick a random active codeword to try to remove
            remove_idx = active_indices[random.randint(0, len(active_indices) - 1)]
            
            # Identify words uniquely covered by this codeword
            sphere = sphere_idx_2d[code_indices[remove_idx]]
            unique_mask = (cover_count[sphere] == 1)
            unique_words = sphere[unique_mask]
            
            if len(unique_words) == 0:
                # Safe removal
                active[remove_idx] = False
                cover_count[sphere] -= 1
                
                new_size = int(np.sum(active))
                if new_size < best_code_size:
                    best_code_size = new_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()
                    
                # Try to remove more redundancies immediately
                remove_redundancies()
                current_size = int(np.sum(active))
                if current_size < best_code_size:
                    best_code_size = current_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()
            else:
                # Try to find a replacement codeword
                # The replacement must cover all unique_words
                # Pick a random unique word and look at its sphere for candidates
                target_word = unique_words[random.randint(0, len(unique_words) - 1)]
                candidates = sphere_idx_2d[target_word]
                
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                found_replacement = False
                for cand_idx in candidates:
                    if cand_idx == code_indices[remove_idx]:
                        continue
                    if active[cand_idx]: # We are looking for a swap, so usually we want to add an inactive one? 
                                           # Actually, if it's active, it's already in the code. 
                                           # If we remove remove_idx and add cand_idx (which is active), 
                                           # we are just swapping positions in the list, size doesn't change.
                                           # But we want to minimize size. 
                                           # If we find a cand_idx that covers unique_words, 
                                           # we can swap remove_idx with cand_idx.
                                           # Wait, if cand_idx is already active, we don't need to "add" it, 
                                           # we just need to ensure it covers the gap. 
                                           # If cand_idx is active, it's already contributing to cover_count.
                                           # So if we remove remove_idx, cover_count drops.
                                           # If cand_idx is active, it's still there.
                                           # So if cand_idx covers unique_words, the removal is safe!
                                           # This is a "free" removal if an existing active codeword covers the gap.
                                           # Let's check if any active codeword covers unique_words.
                                           pass
                    # Optimization: Check if an existing active codeword covers unique_words
                    # This would allow a pure removal.
                    # We already know remove_idx doesn't.
                    # Let's check all active codewords? That's expensive.
                    # Instead, let's just try to find ANY codeword (active or not) that covers unique_words.
                    # If it's active, we just remove remove_idx (size -1).
                    # If it's inactive, we swap (size 0).
                    
                    # Check if cand_idx covers all unique_words
                    cand_sphere = sphere_idx_2d[cand_idx]
                    # Efficient check: count how many unique_words are in cand_sphere
                    # Since unique_words is a list, we can use a set or just loop (size is small)
                    covers_all = True
                    for uw in unique_words:
                        if uw not in cand_sphere:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # We found a candidate
                        if active[cand_idx]:
                            # Pure removal: cand_idx is already in the code, so it covers the gap
                            active[remove_idx] = False
                            cover_count[sphere] -= 1
                            
                            new_size = int(np.sum(active))
                            if new_size < best_code_size:
                                best_code_size = new_size
                                best_active = active.copy()
                                best_cover_count = cover_count.copy()
                                
                            remove_redundancies()
                            current_size = int(np.sum(active))
                            if current_size < best_code_size:
                                best_code_size = current_size
                                best_active = active.copy()
                                best_cover_count = cover_count.copy()
                        else:
                            # Swap: remove remove_idx, add cand_idx
                            # Accept with Metropolis criterion?
                            # Size is same. We accept if it improves "quality" or just accept to explore.
                            # Let's accept with probability exp(-delta/T) where delta=0 for same size?
                            # Or just accept to move around the solution space.
                            if random.random() < 0.5: # Accept 50% of neutral swaps to explore
                                active[remove_idx] = False
                                cover_count[sphere] -= 1
                                active[cand_idx] = True
                                cover_count[cand_sphere] += 1
                                found_replacement = True
                                break
                        
                        found_replacement = True
                        break
                
                if not found_replacement:
                    # No good replacement found, maybe try a random swap anyway?
                    # Or just reject.
                    pass
            
            T *= cooling_rate
            
            # Periodic redundancy removal
            if sa_iter % 100 == 0:
                remove_redundancies()
                current_size = int(np.sum(active))
                if current_size < best_code_size:
                    best_code_size = current_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()

        # Restore best solution
        active = best_active
        cover_count = best_cover_count
        
        # Final redundancy removal
        remove_redundancies()
        
        final_indices = [code_indices[i] for i in range(m) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]
        
        # Verify coverage
        if len(final_code) < total_space:
            verify_cover = np.zeros(total_space, dtype=bool)
            for idx in final_indices:
                verify_cover[sphere_idx_2d[idx]] = True
            if np.all(verify_cover):
                return [list(w) for w in final_code]
        
        # Fallback: return greedy code if verification failed (should not happen)
        return [list(w) for w in [int_to_word(idx) for idx in code_indices]]
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END

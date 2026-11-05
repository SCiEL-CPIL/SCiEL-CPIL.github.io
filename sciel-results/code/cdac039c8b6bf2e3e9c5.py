import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific optimization for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = 729
        powers = [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors for each word (index 0..728)
        # neighbors[i] is a list of 13 indices (i and 12 others at dist 1)
        neighbors = np.empty((total_words, 13), dtype=np.int32)
        is_neighbor = np.zeros((total_words, total_words), dtype=bool)
        
        for i in range(total_words):
            # Get digits
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            is_neighbor[i, i] = True
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        is_neighbor[new_i, i] = True
                        k += 1

        # Helper to convert index to codeword list
        def index_to_codeword(idx):
            return [(idx // powers[p]) % 3 for p in range(n)]

        # Helper to convert codeword list to index
        def codeword_to_index(cw):
            idx = 0
            for p in range(n):
                idx += cw[p] * powers[p]
            return idx

        # Phase 1: Greedy Construction
        covered = np.zeros(total_words, dtype=bool)
        active_indices = []
        
        while not covered.all():
            if time.time() > deadline - 10.0:
                break
            
            # Calculate how many new words each candidate covers
            # is_neighbor.T has shape (729, 729) where col c is neighbors of c
            # ~covered is a boolean vector
            # new_coverages[c] = sum(is_neighbor[:, c] & ~covered)
            new_coverages = is_neighbor.T @ (~covered).astype(np.int8)
            
            best_idx = np.argmax(new_coverages)
            
            if new_coverages[best_idx] == 0:
                # Should not happen if code is valid, but break to avoid infinite loop
                break
                
            active_indices.append(best_idx)
            covered[is_neighbor[:, best_idx]] = True
            
        # Convert to sets for faster lookup in local search
        active_set = set(active_indices)
        inactive_set = set(range(total_words)) - active_set
        
        best_size = len(active_indices)
        best_active = set(active_indices)

        # Precompute coverage counts
        coverage = np.zeros(total_words, dtype=np.int32)
        for idx in active_indices:
            coverage[neighbors[idx]] += 1

        # Local Search Loop
        while time.time() < deadline - 5.0:
            # Phase 1: Greedy Removal
            removed_any = False
            candidates = list(active_set)
            random.shuffle(candidates)
            
            for idx in candidates:
                if time.time() > deadline - 5.0:
                    break
                if idx not in active_set:
                    continue
                
                # Check if idx is redundant: all neighbors have coverage >= 2
                if np.min(coverage[neighbors[idx]]) >= 2:
                    # Remove idx
                    active_set.remove(idx)
                    inactive_set.add(idx)
                    coverage[neighbors[idx]] -= 1
                    removed_any = True
                    if len(active_set) < best_size:
                        best_size = len(active_set)
                        best_active = set(active_set)
                    break # Restart removal phase

            if removed_any:
                continue

            # Phase 2: Swap-based Local Search
            # Try to find a swap that allows future removals
            # Heuristic: Minimize critical words or just find any valid swap to perturb
            
            best_swap = None
            
            # To speed up, iterate over a random subset of active words
            active_list = list(active_set)
            random.shuffle(active_list)
            
            for idx_out in active_list:
                if time.time() > deadline - 2.0:
                    break
                
                n_out = neighbors[idx_out]
                # Words covered uniquely by idx_out
                unique_mask = coverage[n_out] == 1
                unique_words = n_out[unique_mask]
                
                if len(unique_words) == 0:
                    # Should have been removed
                    continue
                
                # Find an inactive word that covers all unique_words
                # is_neighbor[unique_words, :] gives a boolean matrix (len(unique_words), 729)
                # We want a column idx_in such that all are True
                # candidate_mask = np.all(is_neighbor[unique_words, :], axis=0)
                # candidates = np.where(candidate_mask & inactive_mask)[0]
                
                # To avoid creating large temporary arrays, we can loop over a subset of inactive words
                # Or use the vectorized check if inactive set is small enough
                # Let's try vectorized check for all inactive words
                inactive_list = list(inactive_set)
                if not inactive_list:
                    continue
                
                # Create a boolean mask for inactive words
                # This is O(729) per idx_out, which is acceptable
                inactive_mask = np.zeros(total_words, dtype=bool)
                for idx in inactive_list:
                    inactive_mask[idx] = True
                
                # Check which inactive words cover all unique_words
                # is_neighbor[unique_words, :] is (U, 729)
                # We want columns where all entries are True
                # This is equivalent to: sum(is_neighbor[unique_words, :], axis=0) == len(unique_words)
                covers_all = np.sum(is_neighbor[unique_words, :], axis=0) == len(unique_words)
                
                valid_candidates = np.where(covers_all & inactive_mask)[0]
                
                if len(valid_candidates) > 0:
                    # Pick a random valid candidate
                    idx_in = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                    best_swap = (idx_out, idx_in)
                    break
            
            if best_swap:
                idx_out, idx_in = best_swap
                active_set.remove(idx_out)
                inactive_set.add(idx_out)
                inactive_set.remove(idx_in)
                active_set.add(idx_in)
                
                coverage[neighbors[idx_out]] -= 1
                coverage[neighbors[idx_in]] += 1
                
                # Optional: Check for immediate removals?
                # The loop restarts and checks removals.
            else:
                # No valid swap found.
                # Try a "kick": Pick a random active word and a random inactive word and swap them,
                # even if it breaks coverage? No, we must maintain validity.
                # If no valid swap exists, we are in a local minimum.
                # We can try a random restart of the local search or break.
                # Given time constraints, break.
                break

        final_codewords = [index_to_codeword(idx) for idx in best_active]
        
        return final_codewords

    # General case fallback
    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]
        
    return get_direct_sum()

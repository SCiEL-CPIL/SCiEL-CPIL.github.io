"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        
        # Precompute all spheres using bitmask representation
        # Each word in {0,1,2}^8 maps to an index 0..6560
        # Precompute for each index, which indices are within distance 1
        def word_to_int(w):
            """Convert a tuple of digits to an integer (base-3 encoding)."""
            result = 0
            for d in w:
                result = result * 3 + d
            return result
        
        def int_to_word(idx):
            """Convert an integer back to a tuple of digits (base-3)."""
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))
        
        # Precompute sphere masks: for each word index, the list of indices it covers
        # This is a list of 6561 lists, each with 17 elements
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
        
        # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []
        
        # For efficiency, we'll maintain an array of how many times each word is covered
        # by the selected codewords, and use a priority-based selection
        
        # Precompute: for each candidate word, its sphere as a numpy array
        sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]
        
        # Greedy selection (vectorized)
        # Precompute sphere indices as a 2D array: sphere_idx_2d[i, j] = j-th word in sphere of word i
        sphere_idx_2d = np.array(spheres, dtype=np.int32)  # shape (6561, 17)
        
        # Use a priority queue approach with lazy deletion for efficiency
        # But first, let's try a faster greedy: process in batches
        # Precompute the initial counts (all words uncovered, so each word covers 17)
        # Actually all start at 17, so we need a different strategy
        
        # Better: use the fact that after selecting a word, we can update counts incrementally
        # But full vectorization each step is O(6561*17) = ~111K ops, which is fast
        
        while uncovered_count > 0:
            if time.time() > deadline - 30:  # Leave time for local search
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
        
        # Now we have a greedy code. Let's try to reduce it via local search + simulated annealing.
        code = [int_to_word(idx) for idx in code_indices]
        
        # Local search: try to remove redundant codewords (vectorized)
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[spheres[idx]] += 1
        
        m = len(code_indices)
        
        # Precompute sphere index matrix for all codewords in the code
        sphere_idx_matrix = np.array([spheres[idx] for idx in code_indices], dtype=np.int32)  # shape (m, 17)
        
        active = np.ones(m, dtype=bool)
        
        # Run many passes of redundant codeword removal
        for pass_num in range(200):
            if time.time() > deadline - 30:
                break
            
            if not np.any(active):
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # Remove all redundant codewords at once
            active_spheres = sphere_idx_matrix[active_indices]  # (num_active, 17)
            active_cover_counts = cover_count[active_spheres]  # (num_active, 17)
            min_covers = np.min(active_cover_counts, axis=1)  # (num_active,)
            
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            
            if len(redundant_local_indices) == 0:
                break
            
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            
            for ri in redundant_global_indices:
                cover_count[spheres[code_indices[ri]]] -= 1
            
            if not np.any(active):
                break
        
        # Simulated annealing phase: swap codewords to find better configurations
        # State: set of active codeword indices
        # Move: replace one active codeword with a random inactive one
        # Cost: number of uncovered words (want to minimize)
        
        def get_uncovered_count():
            """Count uncovered words given current active set."""
            if not np.any(active):
                return 0
            active_idx = np.where(active)[0]
            # Use cover_count to determine coverage
            return int(np.sum(cover_count == 0))
        
        # Simulated annealing parameters
        T = 100.0  # Initial temperature
        cooling_rate = 0.999
        T_min = 0.01
        max_sa_iterations = 50000
        
        best_code_size = m
        best_active = active.copy()
        
        for sa_iter in range(max_sa_iterations):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            active_count = int(np.sum(active))
            if active_count < 2:
                break
            
            # Choose a random active codeword to remove
            active_indices = np.where(active)[0]
            remove_idx = active_indices[random.randint(0, len(active_indices) - 1)]
            
            # Try removing it
            active[remove_idx] = False
            cover_count[spheres[code_indices[remove_idx]]] -= 1
            
            # Check if still covered
            new_uncovered = int(np.sum(cover_count == 0))
            
            if new_uncovered == 0:
                # Accept (improvement or neutral)
                best_code_size = active_count - 1
                best_active = active.copy()
            else:
                # Metropolis acceptance
                delta_cost = new_uncovered  # increase in uncovered words
                if random.random() < np.exp(-delta_cost / T):
                    # Accept worse solution
                    pass
                else:
                    # Reject: restore
                    active[remove_idx] = True
                    cover_count[spheres[code_indices[remove_idx]]] += 1
            
            # Also try adding a new codeword that helps most
            if time.time() < deadline - 10:
                # Find best candidate to add
                # For efficiency, sample a few candidates
                num_candidates = min(20, total_space)
                candidates = random.sample(range(total_space), num_candidates)
                
                best_gain = 0
                best_candidate = None
                for cand in candidates:
                    # Count how many currently uncovered words this candidate covers
                    cand_sphere = sphere_idx_2d[cand]
                    gain = int(np.sum((cover_count[cand_sphere] == 0)))
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate is not None and best_gain > 0:
                    # Add this candidate
                    # First, check if it's already in code_indices
                    if best_candidate not in [code_indices[i] for i in range(m) if active[i]]:
                        # Add it (we'll track it separately for now)
                        # For simplicity, we'll just note this and continue
                        pass
            
            T *= cooling_rate
        
        # After SA, try to remove redundancies again
        for pass_num in range(100):
            if time.time() > deadline - 3:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
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
                cover_count[spheres[code_indices[ri]]] -= 1
        
        final_indices = [code_indices[i] for i in range(m) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]
        
        # Verify coverage
        if len(final_code) < total_space:
            verify_cover = np.zeros(total_space, dtype=bool)
            for idx in final_indices:
                verify_cover[spheres[idx]] = True
            if np.all(verify_cover):
                return [list(w) for w in final_code]
        
        # Fallback: return greedy code
        return [list(w) for w in code]
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END

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
        
        while uncovered_count > 0:
            if time.time() > deadline - 30:  # Leave time for local search
                break
            
            # For each candidate word, count how many of its sphere words are uncovered
            # uncovered[sphere_idx_2d[i, :]] gives the uncovered status for sphere i
            # We want: for each i, sum(uncovered[sphere_idx_2d[i, :]])
            # This is: (uncovered[sphere_idx_2d]).sum(axis=1)
            # uncovered is bool array of size 6561, sphere_idx_2d is (6561, 17)
            # uncovered[sphere_idx_2d] gives a (6561, 17) bool array
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            
            if best_count <= 0:
                break
            
            code_indices.append(best_idx)
            # Mark covered words
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(uncovered.sum())
        
        # Now we have a greedy code. Let's try to reduce it via local search.
        # Convert to list of words
        code = [int_to_word(idx) for idx in code_indices]
        
        # Local search: try to remove redundant codewords (vectorized)
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[spheres[idx]] += 1
        
        # Precompute a 2D array: cover_matrix[i, j] = 1 if codeword i covers word j
        m = len(code_indices)
        cover_matrix = np.zeros((m, total_space), dtype=np.int32)
        for i, idx in enumerate(code_indices):
            cover_matrix[i, spheres[idx]] = 1
        
        active = np.ones(m, dtype=bool)
        
        # Run many passes of redundant codeword removal
        for pass_num in range(500):
            if time.time() > deadline - 2:
                break
            
            if not np.any(active):
                break
            
            # For each active codeword, check if all words it covers are covered by >=2 codewords
            # A codeword is redundant if min(cover_count[sphere[i]]) >= 2
            # Vectorized: for each active codeword, get the min cover count in its sphere
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # For each active codeword, find the minimum cover count among words it covers
            # cover_matrix[i] is a 0/1 vector; we need min(cover_count) over positions where cover_matrix[i]=1
            # This is expensive to do fully vectorized for 6561 elements x m rows
            # Alternative: use the fact that sphere size is only 17
            # For each active codeword i, the sphere is spheres[code_indices[i]] (17 elements)
            
            # Build a matrix: min_cover[i] = min(cover_count[spheres[code_indices[i]]])
            # Since sphere size is fixed at 17, we can precompute the sphere indices as a 2D array
            sphere_idx_matrix = np.array([spheres[idx] for idx in code_indices], dtype=np.int32)  # shape (m, 17)
            
            # For active codewords, get the cover counts in their spheres
            # active_cover_counts shape: (num_active, 17)
            active_spheres = sphere_idx_matrix[active_indices]  # (num_active, 17)
            active_cover_counts = cover_count[active_spheres]  # (num_active, 17)
            min_covers = np.min(active_cover_counts, axis=1)  # (num_active,)
            
            # Find which active codewords are redundant (min_cover >= 2)
            redundant_mask = min_covers >= 2
            redundant_local_indices = np.where(redundant_mask)[0]
            
            if len(redundant_local_indices) == 0:
                break
            
            # Remove them
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            
            # Update cover_count: subtract 1 for each word covered by removed codewords
            for ri in redundant_global_indices:
                cover_count[spheres[code_indices[ri]]] -= 1
            
            # If all removed in this pass, we can stop early or continue
            if not np.any(active):
                break
        
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

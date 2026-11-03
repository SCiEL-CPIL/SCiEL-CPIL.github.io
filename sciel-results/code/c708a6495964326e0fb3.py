"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17
        pow3 = [3 ** i for i in range(n + 1)]

        # Precompute all spheres as a 2D numpy array: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            # Decode idx to digits
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = digits[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        sphere_idx_2d[idx, k] = idx + (d - orig) * base
                        k += 1

        # Phase 1: Greedy set cover with vectorized selection
        # Start with a structured initialization: tetracode product code
        # Tetracode: 9 words of length 4, then extend to length 8
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Build initial code from tetracode product
        uncovered = np.ones(total_space, dtype=bool)
        code_indices = []
        
        # Add tetracode-based words as initial seed
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                # Convert to index
                idx = 0
                for d in word:
                    idx = idx * 3 + d
                if uncovered[idx]:
                    code_indices.append(idx)
                    uncovered[sphere_idx_2d[idx]] = False
        
        uncovered_count = int(np.sum(uncovered))
        
        # Complete with greedy if not fully covered
        greedy_deadline = start_time + 30
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            # Vectorized: count uncovered words covered by each candidate
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            if best_count <= 0:
                break
            code_indices.append(best_idx)
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(np.sum(uncovered))
        
        m = len(code_indices)
        
        # Phase 2: Vectorized redundancy removal
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1
        
        active = np.ones(m, dtype=bool)
        sphere_idx_matrix = sphere_idx_2d[code_indices]  # shape (m, 17)
        
        cleanup_deadline = start_time + 50
        for pass_num in range(500):
            if time.time() > cleanup_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            # For each active codeword, check if all its sphere positions are covered by >= 2 codewords
            active_spheres = sphere_idx_matrix[active_indices]  # (k, 17)
            active_cover_counts = cover_count[active_spheres]    # (k, 17)
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            # Decrement cover_count for removed codewords
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1
        
        best_code_size = int(np.sum(active))
        
        # Phase 3: Local search with swap moves
        # Try to replace one codeword with another to reduce total size or maintain size with better coverage
        sa_deadline = deadline - 5
        T = 50.0
        cooling_rate = 0.9995
        T_min = 1.0
        best_active = active.copy()
        best_cover_count = cover_count.copy()
        best_size = best_code_size
        
        # Precompute for faster access
        while time.time() < sa_deadline and T > T_min:
            active_indices = np.where(active)[0]
            if len(active_indices) < 2:
                break
            
            # Randomly pick a codeword to try removing
            remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]
            remove_idx = code_indices[remove_pos]
            remove_sphere = sphere_idx_matrix[remove_pos]
            
            # Temporarily remove
            active[remove_pos] = False
            cover_count[remove_sphere] -= 1
            uncovered_mask = cover_count == 0
            uncovered_count = int(np.sum(uncovered_mask))
            
            if uncovered_count == 0:
                # Pure improvement: codeword was redundant
                best_code_size = int(np.sum(active))
                best_active = active.copy()
                best_cover_count = cover_count.copy()
                best_size = best_code_size
                T *= cooling_rate
                continue
            else:
                # Try to find a replacement codeword that covers the uncovered words
                # Sample some uncovered words and look for good candidates
                uncovered_indices = np.where(uncovered_mask)[0]
                num_trials = min(30, len(uncovered_indices))
                trial_indices = random.sample(range(len(uncovered_indices)), num_trials)
                
                best_gain = 0
                best_cand = -1
                best_cand_sphere = None
                
                for ti in trial_indices:
                    uw = uncovered_indices[trial_indices[ti]]
                    # Look at the sphere of uw as candidate positions
                    sphere_of_uw = sphere_idx_2d[uw]
                    for sp in sphere_of_uw:
                        sp = int(sp)
                        gain = int(np.sum(cover_count[sphere_idx_2d[sp]] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = sp
                            best_cand_sphere = sphere_idx_2d[sp]
                
                if best_cand >= 0 and best_gain > 0:
                    # Check if best_cand is already in code_indices
                    # For simplicity, just add it if not present
                    if best_cand not in code_indices:
                        code_indices.append(best_cand)
                        m = len(code_indices)
                        sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                        active = np.append(active, True)
                        cover_count[sphere_idx_2d[best_cand]] += 1
                    else:
                        # Find its position
                        for i, ci in enumerate(code_indices):
                            if ci == best_cand:
                                active[i] = True
                                cover_count[sphere_idx_2d[best_cand]] += 1
                                break
                    
                    new_uncovered = int(np.sum(cover_count == 0))
                    if new_uncovered == 0:
                        # Valid swap
                        new_size = int(np.sum(active))
                        if new_size < best_size:
                            best_size = new_size
                            best_active = active.copy()
                            best_cover_count = cover_count.copy()
                        elif new_size == best_size:
                            # Accept with probability based on temperature
                            if random.random() < 0.5:
                                best_active = active.copy()
                                best_cover_count = cover_count.copy()
                    else:
                        # Invalid: revert the addition
                        if best_cand not in code_indices[:-1]:
                            code_indices.pop()
                            m = len(code_indices)
                            sphere_idx_matrix = sphere_idx_matrix[:-1]
                            active = active[:-1]
                        else:
                            for i, ci in enumerate(code_indices):
                                if ci == best_cand:
                                    active[i] = False
                                    cover_count[sphere_idx_2d[best_cand]] -= 1
                                    break
                        # Revert the removal
                        active[remove_pos] = True
                        cover_count[remove_sphere] += 1
                else:
                    # No good replacement found, revert
                    active[remove_pos] = True
                    cover_count[remove_sphere] += 1
            
            T *= cooling_rate
        
        # Use the best solution found
        final_active = best_active
        final_cover_count = best_cover_count
        
        # Extract final code
        final_code_indices = [code_indices[i] for i in range(len(final_active)) if final_active[i]]
        
        # Verify coverage
        if len(final_code_indices) < total_space:
            verify_cover = np.zeros(total_space, dtype=bool)
            for idx in final_code_indices:
                verify_cover[sphere_idx_2d[idx]] = True
            if np.all(verify_cover):
                # Convert to list of words
                final_code = []
                for idx in final_code_indices:
                    digits = []
                    tmp = idx
                    for _ in range(n):
                        digits.append(tmp % 3)
                        tmp //= 3
                    digits.reverse()
                    final_code.append(digits)
                return final_code
        
        # Fallback: return the greedy code
        fallback_code = []
        for idx in code_indices:
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()
            fallback_code.append(digits)
        return fallback_code
    
    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END

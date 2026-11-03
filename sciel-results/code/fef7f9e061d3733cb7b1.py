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

        # Precompute spheres as indices
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
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
        
        # Precompute bitsets for spheres for fast intersection checks
        # Each bitset is a Python int representing which points are covered
        sphere_bits = [0] * total_space
        for i in range(total_space):
            b = 0
            for p in sphere_idx_2d[i]:
                b |= (1 << p)
            sphere_bits[i] = b

        # Initialization: Tetracode product
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        code_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                idx = 0
                for d in word:
                    idx = idx * 3 + d
                code_indices.append(idx)
        
        # Build active set and cover counts
        active = [True] * len(code_indices)
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1
        
        # Helper to convert index to word
        def idx_to_word(idx):
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            return digits[::-1]

        # Phase 1: Aggressive Reduction
        # Remove any codeword that is redundant (all its points covered by >= 2)
        changed = True
        while changed:
            changed = False
            for i in range(len(active)):
                if not active[i]:
                    continue
                idx = code_indices[i]
                sphere = sphere_idx_2d[idx]
                # Check min cover count in sphere
                if np.min(cover_count[sphere]) >= 2:
                    # Remove
                    active[i] = False
                    cover_count[sphere] -= 1
                    changed = True

        # Phase 2: Local Search with Bitset Swap
        # We maintain a list of active indices for fast random selection
        active_indices = [i for i in range(len(active)) if active[i]]
        
        T = 50.0
        cooling_rate = 0.999
        T_min = 1.0
        
        # Track best solution
        best_size = len(active_indices)
        best_active = active.copy()
        best_cover_count = cover_count.copy()

        while time.time() < deadline and T > T_min:
            if not active_indices:
                break
            
            # Randomly pick an active codeword to try removing/swapping
            pos = random.choice(active_indices)
            idx = code_indices[pos]
            sphere = sphere_idx_2d[idx]
            
            # Temporarily remove
            cover_count[sphere] -= 1
            active[pos] = False
            active_indices.remove(pos)
            
            # Find uncovered points
            uncovered_mask = (cover_count == 0)
            uncovered_count = int(np.sum(uncovered_mask))
            
            if uncovered_count == 0:
                # Pure reduction
                best_size = len(active_indices)
                best_active = active.copy()
                best_cover_count = cover_count.copy()
                T *= cooling_rate
                continue
            
            # Need to find a replacement r that covers all uncovered points
            # Construct bitset of uncovered points
            U_bits = 0
            # Optimization: only iterate over uncovered indices
            uncovered_indices = np.where(uncovered_mask)[0]
            for u in uncovered_indices:
                U_bits |= (1 << u)
            
            # Find candidate r
            # We want r such that sphere_bits[r] covers U_bits
            # i.e., (sphere_bits[r] & U_bits) == U_bits
            # We also prefer r that is not currently active (to avoid swapping with self/other active in a way that increases size? 
            # Actually, if r is active, we are just moving coverage? No, if r is active, it's already in the code.
            # If we remove idx and add r (active), we are just swapping idx with r? 
            # But r is already counted in cover_count. 
            # So if r is active, we just "move" the coverage? No, we removed idx, so cover_count decreased.
            # If we "add" r, we increase cover_count.
            # If r is already active, we are effectively replacing idx with r? 
            # But r is still there. We are just removing idx.
            # So if r is active, we don't need to "add" it, we just keep it.
            # But we already removed idx. So the size decreases by 1?
            # No, if r is active, it was already in the code. 
            # The move is: Remove idx.
            # If the resulting code is valid (U_bits == 0), we accept.
            # If U_bits != 0, we need to ADD a new codeword r.
            # So r should ideally NOT be active, or if it is, we are just ensuring it stays?
            # If r is active, it's already in the code. Adding it again does nothing.
            # So we must find r such that adding it covers U_bits.
            # If r is active, it's already there. So if r covers U_bits, it means U_bits was covered by r?
            # But we just checked U_bits are uncovered. Contradiction.
            # So r MUST be inactive (or new).
            
            best_r = -1
            best_gain = -1
            
            # Sample candidates to speed up
            # Try random indices first
            candidates = random.sample(range(total_space), min(500, total_space))
            # Also try neighbors of uncovered points
            for u in uncovered_indices[:5]:
                candidates.extend(sphere_idx_2d[u].tolist())
            
            for r in candidates:
                if active[code_indices.index(r)] if r in code_indices else False:
                    # If r is active, it's already in the code.
                    # But we determined U is uncovered.
                    # So r cannot cover U if r is active? 
                    # Wait. cover_count[u] == 0 means NO active codeword covers u.
                    # So if r is active, it does NOT cover u.
                    # So (sphere_bits[r] & U_bits) will be 0 for any u in U.
                    # So active r's are useless.
                    continue
                
                # Check if r covers all of U
                if (sphere_bits[r] & U_bits) == U_bits:
                    # Valid candidate
                    # Gain: how many points does r cover that are currently covered by >= 2?
                    # This helps future reductions.
                    gain = int(np.sum(cover_count[sphere_idx_2d[r]] >= 2))
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
            
            if best_r != -1:
                # Add best_r
                # Check if it's already in code_indices (inactive)
                if best_r in code_indices:
                    r_pos = code_indices.index(best_r)
                    active[r_pos] = True
                    active_indices.append(r_pos)
                    cover_count[sphere_idx_2d[best_r]] += 1
                else:
                    code_indices.append(best_r)
                    active.append(True)
                    active_indices.append(len(code_indices) - 1)
                    cover_count[sphere_idx_2d[best_r]] += 1
                
                # Check if valid
                if np.all(cover_count > 0):
                    current_size = len(active_indices)
                    if current_size < best_size:
                        best_size = current_size
                        best_active = active.copy()
                        best_cover_count = cover_count.copy()
                    # Accept swap (size same)
                    # To avoid cycling, we can use SA
                    # But for now, accept if size <= best_size
                else:
                    # Should not happen if logic is correct
                    # Revert
                    if best_r in code_indices:
                        r_pos = code_indices.index(best_r)
                        active[r_pos] = False
                        active_indices.remove(r_pos)
                        cover_count[sphere_idx_2d[best_r]] -= 1
                    else:
                        code_indices.pop()
                        active.pop()
                        active_indices.pop()
                        cover_count[sphere_idx_2d[best_r]] -= 1
                    # Revert removal of idx
                    active[pos] = True
                    active_indices.append(pos)
                    cover_count[sphere] += 1
            else:
                # No replacement found, revert
                active[pos] = True
                active_indices.append(pos)
                cover_count[sphere] += 1
            
            T *= cooling_rate

        # Extract best solution
        final_code_indices = [code_indices[i] for i in range(len(best_active)) if best_active[i]]
        
        # Verify
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_code_indices:
            verify_cover[sphere_idx_2d[idx]] = True
        if np.all(verify_cover):
            return [idx_to_word(idx) for idx in final_code_indices]
        
        # Fallback
        return [idx_to_word(idx) for idx in code_indices]

    else:
        # General case
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END

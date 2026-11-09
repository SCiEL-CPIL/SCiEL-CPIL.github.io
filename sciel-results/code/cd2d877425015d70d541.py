import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safe margin for 60s limit
    
    best_score = float('inf')
    best_poly = None
    
    # Helper to count nonzeros in square using NumPy
    def count_nz_square(poly):
        # poly is list of 41 ints
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        # sq length is 41+41-1 = 81
        return np.count_nonzero(sq)

    # Initialize with seed
    seed_poly = [0] * 41
    for i in range(12):
        seed_poly[i] = 1
    best_poly = seed_poly
    best_score = count_nz_square(seed_poly)
    
    # Add some structured candidates
    candidates = [seed_poly]
    
    # Palindromic 12 terms
    p_pal = [0] * 41
    # Try symmetric placement
    # e.g. indices 0, 1, ..., 5, 35, 36, ..., 40 ? No, need 12 terms.
    # Let's just generate a few random structured ones
    for _ in range(5):
        p = [0] * 41
        # Choose 6 pairs of indices to be symmetric?
        # Or just random signs on contiguous block
        signs = [random.choice([-1, 1]) for _ in range(12)]
        for i in range(12):
            p[i] = signs[i]
        candidates.append(p)
        
    # Arithmetic progression with signs
    p_ap = [0] * 41
    for i in range(12):
        p_ap[i*3] = random.choice([-1, 1]) # indices 0, 3, ..., 33
    candidates.append(p_ap)
    
    # Evaluate initial candidates
    for p in candidates:
        # Ensure validity: exactly 12 nonzeros, coeffs in [-64,64], last coeff nonzero
        nz_count = sum(1 for x in p if x != 0)
        if nz_count != 12: continue
        if p[-1] == 0: 
             # Fix last coeff if it's zero? The contract says "nonzero final coefficient".
             # If p[40] is 0, we need to move a term or set it.
             # My random generation might leave p[40] as 0.
             # Let's force the last term to be non-zero if it's 0 by swapping with a non-zero one or setting it.
             # Actually, the contract says "nonzero final coefficient".
             # So we must ensure p[40] != 0.
             # If p[40] == 0, find a non-zero index j, swap p[40] and p[j]? 
             # Or just set p[40]=1 and zero out another term?
             # Simplest: if p[40]==0, pick a random non-zero index k, set p[40]=p[k], p[k]=0.
             nz_indices = [i for i in range(41) if p[i] != 0]
             if nz_indices:
                 k = random.choice(nz_indices)
                 p[40] = p[k]
                 p[k] = 0
             else:
                 p[40] = 1
        if any(abs(x) > 64 for x in p): continue
        
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_poly = p[:]

    # Local Search
    current_poly = best_poly[:]
    current_score = best_score
    
    no_improve_count = 0
    while time.time() < deadline:
        if no_improve_count > 200:
            # Random restart
            current_poly = [0] * 41
            indices = random.sample(range(41), 12)
            for idx in indices:
                current_poly[idx] = random.randint(-64, 64)
                if current_poly[idx] == 0:
                    current_poly[idx] = 1
            # Ensure last coeff non-zero
            if current_poly[40] == 0:
                nz_indices = [i for i in range(41) if current_poly[i] != 0]
                k = random.choice(nz_indices)
                current_poly[40] = current_poly[k]
                current_poly[k] = 0
            current_score = count_nz_square(current_poly)
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_poly = current_poly[:]
            continue

        # Pick a move
        # 1. Change coefficient of a non-zero term
        # 2. Move a term to a new index
        move_type = random.random()
        
        if move_type < 0.5:
            # Change coeff
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            new_val = random.randint(-64, 64)
            if new_val == 0:
                new_val = old_val + random.choice([-1, 1])
                if new_val == 0: new_val = 1
                if abs(new_val) > 64: new_val = 64 if new_val > 0 else -64
            
            current_poly[idx] = new_val
            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                current_poly[idx] = old_val
                no_improve_count += 1
        else:
            # Move index
            nz_indices = [i for i in range(41) if current_poly[i] != 0]
            z_indices = [i for i in range(41) if current_poly[i] == 0]
            if not z_indices:
                no_improve_count += 1
                continue
                
            src_idx = random.choice(nz_indices)
            dst_idx = random.choice(z_indices)
            
            val = current_poly[src_idx]
            current_poly[src_idx] = 0
            current_poly[dst_idx] = val
            
            # Check validity: last coeff must be non-zero
            # If we moved the term at 40 to elsewhere, p[40] becomes 0. Invalid.
            # If we moved a term to 40, p[40] becomes non-zero. Valid (if src wasn't 40).
            # If src=40, dst!=40, then p[40]=0. Invalid.
            # So if src_idx == 40, we must ensure dst_idx != 40 (which is true as dst is from z_indices, and 40 is non-zero so not in z_indices).
            # Wait, if src_idx == 40, then 40 is in nz_indices. dst_idx is in z_indices.
            # So p[40] becomes 0. This violates "nonzero final coefficient".
            # So we must reject moves where src_idx == 40.
            if src_idx == 40:
                # Revert
                current_poly[dst_idx] = 0
                current_poly[src_idx] = val
                no_improve_count += 1
                continue

            new_score = count_nz_square(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            else:
                current_poly[dst_idx] = 0
                current_poly[src_idx] = val
                no_improve_count += 1

    return best_poly

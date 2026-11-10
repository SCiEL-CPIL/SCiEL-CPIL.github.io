import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def init_state(positions, values):
        coeffs = np.zeros(41, dtype=np.int64)
        for p, v in zip(positions, values):
            coeffs[p] = v
        square = np.convolve(coeffs, coeffs, mode='full')
        nonzero_count = np.count_nonzero(square)
        return coeffs, square, nonzero_count

    def local_search(positions, time_limit):
        best_score = float('inf')
        best_coeffs = None
        
        # Pre-compute pairs for fast update
        # pairs[i] = (j, k) where k = positions[i] + positions[j]
        # We need to update square[k] by 2 * delta * coeffs[j]
        # And square[2*positions[i]] by delta^2
        
        # For each i, we iterate over all j.
        # Let's pre-compute the list of (j, k) for each i.
        update_maps = []
        for i in range(12):
            pi = positions[i]
            map_i = []
            for j in range(12):
                pj = positions[j]
                k = pi + pj
                map_i.append((j, k))
            update_maps.append(map_i)
            
        starts = []
        starts.append([1]*12)
        starts.append([1 if i%2==0 else -1 for i in range(12)])
        random.seed(123)
        for _ in range(5):
            starts.append([random.choice([-1, 1]) * random.randint(1, 3) for _ in range(12)])
            
        for init_vals in starts:
            if time.time() > start_time + time_limit:
                break
                
            coeffs, square, nz_count = init_state(positions, init_vals)
            curr_score = nz_count
            
            if curr_score < best_score:
                best_score = curr_score
                best_coeffs = coeffs.copy()
                
            temp = 5.0
            cooling = 0.99
            no_improve = 0
            max_no_improve = 500
            
            while time.time() < start_time + time_limit and no_improve < max_no_improve:
                idx = random.randint(0, 11)
                r = random.random()
                if r < 0.5:
                    new_val = random.choice([-1, 1])
                elif r < 0.8:
                    new_val = random.choice([-2, -1, 1, 2])
                else:
                    new_val = random.randint(-8, 8)
                    if new_val == 0:
                        new_val = 1 if random.random() < 0.5 else -1
                    
                p_idx = positions[idx]
                if coeffs[p_idx] == new_val:
                    continue
                
                old_val = coeffs[p_idx]
                delta = new_val - old_val
                
                # Save state for revert
                # Only save square values that change
                changed_indices = set()
                old_square_vals = {}
                
                # Apply update
                for j, k in update_maps[idx]:
                    if k >= 0 and k < 81:
                        if k not in old_square_vals:
                            old_square_vals[k] = square[k]
                            changed_indices.add(k)
                        square[k] += 2 * delta * coeffs[j]
                        
                k2 = 2 * p_idx
                if k2 >= 0 and k2 < 81:
                    if k2 not in old_square_vals:
                        old_square_vals[k2] = square[k2]
                        changed_indices.add(k2)
                    square[k2] += delta * delta
                
                coeffs[p_idx] = new_val
                
                new_nz = np.count_nonzero(square)
                
                if new_nz < curr_score:
                    curr_score = new_nz
                    nz_count = new_nz
                    no_improve = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_coeffs = coeffs.copy()
                        if best_score <= 20:
                            break
                else:
                    delta_score = new_nz - curr_score
                    if random.random() < np.exp(-delta_score / max(temp, 0.1)):
                        curr_score = new_nz
                        nz_count = new_nz
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_coeffs = coeffs.copy()
                    else:
                        # Revert
                        coeffs[p_idx] = old_val
                        for k in changed_indices:
                            square[k] = old_square_vals[k]
                        no_improve += 1
                        
                temp *= cooling
                
            if best_score <= 20:
                break
                
        return best_score, best_coeffs

    candidates = []
    
    # 1. APs
    for start in range(0, 30):
        for step in range(1, 5):
            pos = tuple(start + i * step for i in range(12))
            if pos[-1] <= 40:
                candidates.append(list(pos))
                
    # 2. Random sets
    random.seed(42)
    for _ in range(500):
        pos = random.sample(range(41), 12)
        pos.sort()
        candidates.append(pos)
        
    def collision_score(pos):
        sums = set()
        for i in range(12):
            for j in range(i, 12):
                sums.add(pos[i] + pos[j])
        return len(sums)
        
    candidates.sort(key=collision_score)
    
    best_score = float('inf')
    best_poly = None
    
    n_candidates = min(100, len(candidates))
    time_per_candidate = 0.5
    
    for idx, pos in enumerate(candidates[:n_candidates]):
        if time.time() > deadline - 20:
            break
            
        score, coeffs = local_search(pos, time_per_candidate)
        
        if score < best_score:
            best_score = score
            best_poly = coeffs
            
    if best_poly is not None and time.time() < deadline - 5:
        active_indices = np.nonzero(best_poly)[0]
        best_positions = list(active_indices)
        
        score, coeffs = local_search(best_positions, deadline - time.time() - 5)
        
        if score < best_score:
            best_score = score
            best_poly = coeffs

    if best_poly is None:
        best_poly = np.zeros(41, dtype=np.int64)
        best_poly[:12] = 1
        
    # Trim trailing zeros
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]
        
    return best_poly.tolist()


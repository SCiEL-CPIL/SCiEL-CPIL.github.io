import numpy as np
import time
import random

def construct():
    N = 51975
    # Find divisors of 51975
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = sorted(list(set(divisors)))
    divisors = [d for d in divisors if d > 1]
    
    num_mods = len(divisors)
    
    # Precompute index arrays for each divisor to speed up coverage checks
    # idx_arrays[i] is the array of indices covered by modulus divisors[i] with residue 0
    # To get indices for residue r, we can use (idx_arrays[i] + r) % N ? No, that's not right for arithmetic progression.
    # Indices for (m, r) are r, r+m, r+2m...
    # We can just use np.arange(r, N, m).
    # However, precomputing the base arrays and shifting might not help directly due to different lengths?
    # All have same length N//m.
    # Let's precompute the "shift" logic?
    # Actually, np.arange is quite fast. Let's stick to on-the-fly but optimized.
    
    # Helper to compute coverage count
    def compute_uncovered(counts):
        return np.sum(counts == 0)
    
    def greedy_construct(seed):
        """
        Greedily construct a covering system by picking the (m, r) that covers most uncovered spots.
        """
        random.seed(seed)
        counts = np.zeros(N, dtype=np.int32)
        used_moduli = set()
        residues = [None] * num_mods
        # Map divisor to index in divisors list
        div_to_idx = {d: i for i, d in enumerate(divisors)}
        
        # To make it faster, we can precompute the potential coverage for each modulus
        # But coverage depends on current 'uncovered' mask.
        
        # List of available moduli
        available_moduli = list(divisors)
        
        while available_moduli:
            best_gain = -1
            best_m = None
            best_r = None
            
            # Sample a subset of moduli if the list is large to save time?
            # There are about 100 divisors. 100 * (N/m) operations per step is feasible.
            # Let's iterate through all available moduli.
            
            # Shuffle to break ties randomly? Or just pick best.
            # To avoid bias, we can shuffle the order of checking moduli.
            random.shuffle(available_moduli)
            
            for m in available_moduli:
                # For this m, find the r that maximizes coverage of uncovered residues.
                # We can iterate r from 0 to m-1? m can be up to 51975.
                # If m is large, iterating all r is slow.
                # Optimization: The number of residues covered by (m, r) is N//m.
                # We want to maximize sum(counts[r + k*m] == 0) for k=0..N//m-1.
                
                # If m is small, we can check all r.
                # If m is large, N//m is small.
                # Actually, we can just check a few random r's for large m?
                # Or better: 
                # Let's try to be smart.
                # For small m, check all r.
                # For large m, the number of elements is small.
                
                # Let's just check all r for m < 1000, and sample for larger?
                # Actually, N=51975 is small.
                # Max m is 51975. N//m = 1.
                # Min m is 3. N//m = 17325.
                
                # Let's check all r for m <= 100.
                # For m > 100, sample 10 random r's?
                # Or just check all r? 
                # Sum of m over all divisors?
                # Let's estimate cost.
                # For each m, cost is roughly m * (N/m) = N operations to check all r?
                # No, for a fixed r, cost is N/m. To check all r, cost is m * (N/m) = N.
                # So checking all r for one m costs O(N).
                # If we do this for all ~100 moduli, cost is 100 * N = 5 million ops.
                # This is fast enough for one greedy step?
                # But we have to do this ~100 times (once per modulus added).
                # Total 500 million ops. Might be too slow in Python/NumPy loop.
                
                # Let's optimize:
                # Instead of checking all r, we can use the fact that we only care about uncovered spots.
                # Let U be the set of uncovered indices.
                # For a given m, we want r that maximizes | { u in U : u % m == r } |.
                # We can compute this by iterating U? No, U is large.
                # We can iterate k in 0..N//m-1, and for each index x = r + k*m, check if x is in U.
                # This is still O(N) per m if we do it naively.
                
                # Let's use NumPy vectorization.
                # For a given m, create an array of indices for r=0: idx0 = np.arange(0, N, m)
                # Then for r=1: idx1 = np.arange(1, N, m)
                # This is hard to vectorize over r easily without large memory.
                
                # Alternative:
                # For small m, just loop r.
                # For large m, the number of items is small.
                
                # Let's try a hybrid:
                # If m < 100: check all r.
                # Else: check 5 random r's.
                
                if m < 100:
                    r_range = range(m)
                else:
                    r_range = random.sample(range(m), min(5, m))
                    
                for r in r_range:
                    # Calculate gain
                    # Indices: r, r+m, ...
                    idxs = np.arange(r, N, m)
                    # Count how many of these are currently uncovered (counts == 0)
                    gain = np.sum(counts[idxs] == 0)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
                        
            if best_m is None:
                break
                
            # Add the best class found
            idx = div_to_idx[best_m]
            residues[idx] = best_r
            available_moduli.remove(best_m)
            
            # Update counts
            idxs = np.arange(best_r, N, best_m)
            counts[idxs] += 1
            
        # Ensure all moduli are assigned a residue (if loop broke early or some were left)
        for i, m in enumerate(divisors):
            if residues[i] is None:
                residues[i] = random.randint(0, m - 1)
                # Add to counts
                idxs = np.arange(residues[i], N, m)
                counts[idxs] += 1
                
        return residues, counts

    def local_refine(residues, counts, time_limit):
        """
        Greedy hill climbing.
        """
        start_time = time.time()
        current_score = compute_uncovered(counts)
        best_score = current_score
        best_residues = residues.copy()
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            best_delta = 0 # We want to minimize uncovered, so we look for negative delta in uncovered count
            # Actually, let's look for the move that reduces uncovered count the most.
            # delta_uncovered = new_uncovered - old_uncovered. We want min delta_uncovered.
            # If delta_uncovered < 0, we improve.
            
            best_idx = -1
            best_new_r = -1
            best_move_delta = 0
            
            for i, m in enumerate(divisors):
                old_r = residues[i]
                
                # Try all residues?
                # If m is small, try all. If m is large, sample?
                # Let's try all for m < 100, sample for others.
                if m < 100:
                    candidates = range(m)
                else:
                    # Sample some randoms plus maybe 0, 1, m//2?
                    # Let's just sample 10 randoms.
                    candidates = random.sample(range(m), min(10, m))
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    
                    # Calculate delta
                    # Remove old, add new
                    idx_old = np.arange(old_r, N, m)
                    idx_new = np.arange(new_r, N, m)
                    
                    # Count how many in idx_old become uncovered (count was 1)
                    # Count how many in idx_new become covered (count was 0)
                    # Note: idx_old and idx_new are disjoint? No, they can overlap if m is small?
                    # Wait, for a fixed m, the sets {r + k*m} are disjoint for different r?
                    # Yes, modulo m they are distinct. So idx_old and idx_new are disjoint.
                    
                    c_old = counts[idx_old]
                    c_new = counts[idx_new]
                    
                    # If we remove old_r:
                    # Elements in idx_old with count 1 will become 0 (uncovered). Increase score by this count.
                    # Elements in idx_old with count > 1 will become count-1 (still covered). No change to score.
                    # So increase = sum(c_old == 1)
                    
                    # If we add new_r:
                    # Elements in idx_new with count 0 will become 1 (covered). Decrease score by this count.
                    # Elements in idx_new with count > 0 will become count+1 (still covered). No change.
                    # So decrease = sum(c_new == 0)
                    
                    inc = np.sum(c_old == 1)
                    dec = np.sum(c_new == 0)
                    delta = inc - dec
                    
                    if delta < best_move_delta:
                        best_move_delta = delta
                        best_idx = i
                        best_new_r = new_r
            
            if best_idx != -1:
                # Apply the best move
                m = divisors[best_idx]
                old_r = residues[best_idx]
                new_r = best_new_r
                
                idx_old = np.arange(old_r, N, m)
                idx_new = np.arange(new_r, N, m)
                
                counts[idx_old] -= 1
                counts[idx_new] += 1
                residues[best_idx] = new_r
                
                current_score += best_move_delta
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                improved = True
        
        return best_residues, best_score

    # Main loop
    start_time = time.time()
    timeout = 115.0
    best_global_score = N
    best_global_residues = None
    
    # Run multiple restarts
    # Each restart: Greedy Construct + Local Refine
    # Allocate time budget per restart.
    # Say, 5 restarts of 20s each?
    # Or just loop until time is up.
    
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        # Time budget for this restart
        # If we have lots of time left, use more for refinement.
        # Let's say greedy is fast (~0.1s), so most time goes to refinement.
        # Reserve 10s for refinement per restart?
        refine_time = 10.0
        if time.time() - start_time + refine_time > timeout:
            refine_time = max(0.1, timeout - (time.time() - start_time) - 1.0)
            
        if refine_time < 1.0:
            break
            
        seed = restart_count
        residues, counts = greedy_construct(seed)
        
        # Refine
        refined_residues, score = local_refine(residues, counts, refine_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_residues = refined_residues.copy()
            
        # If we hit 0, we can stop?
        if best_global_score == 0:
            break

    # If no best found (should not happen), fallback to random
    if best_global_residues is None:
        best_global_residues = np.array([random.randint(0, d - 1) for d in divisors])

    classes = [[divisors[i], int(best_global_residues[i])] for i in range(num_mods)]
    return classes

import time
import random
import numpy as np

def construct():
    N = 51975
    # Generate divisors of 51975 = 3^3 * 5^2 * 7 * 11
    # We need distinct moduli m > 1 dividing N
    divs = []
    for d in range(2, N + 1):
        if N % d == 0:
            divs.append(d)
    
    # Precompute masks for each divisor
    # mask[m] is a boolean numpy array of length N where mask[m][x] is True if x % m == r
    # But r varies. So we need a way to quickly get the mask for (m, r).
    # Since N is small (51975), we can store masks for all (m, r) pairs?
    # Number of divs is ~ 48. Total pairs = sum(m for m in divs) ~ 200k.
    # Storing 200k boolean arrays of size 51975 is ~ 10GB. Too much.
    
    # Alternative:
    # Represent coverage as a numpy array `counts` of int16 or similar.
    # To update counts for a class (m, r):
    # indices = np.arange(r, N, m)
    # counts[indices] += 1
    # This is fast in numpy.
    
    # To evaluate gain for a class (m, r):
    # indices = np.arange(r, N, m)
    # gain = np.sum(counts[indices] == 0)
    # This is also fast.
    
    # Strategy: Global Greedy Set Cover
    # 1. Start with empty set of classes.
    # 2. Maintain `counts` array (initially all 0).
    # 3. Maintain `uncovered` set or just use counts.
    # 4. Iteratively pick the class (m, r) that maximizes the number of newly covered elements.
    # 5. Add it to the set, update counts.
    # 6. Stop when no class adds any new coverage, or time limit reached.
    # 7. Local search: Try to swap existing classes or adjust residues to improve coverage.
    
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety
    
    # We can use multiple restarts with different random initializations or orderings if time permits.
    # But global greedy is deterministic if we break ties consistently.
    # Let's implement the greedy first.
    
    def run_greedy(divs):
        # `counts` tracks how many times each residue is covered
        counts = np.zeros(N, dtype=np.int16)
        # `used_moduli` tracks which moduli have been used
        used_moduli = set()
        classes = [] # list of [m, r]
        
        # Precompute index arrays for each m to speed up np.arange calls?
        # np.arange is fast, but let's see.
        # Actually, we can precompute the list of indices for each (m, r) if we want, but memory is high.
        # Let's just use np.arange on the fly.
        
        while time.time() < deadline - 5:  # Stop 5s early to allow for local search
            best_gain = 0
            best_m = None
            best_r = None
            
            # Iterate over all available moduli
            for m in divs:
                if m in used_moduli:
                    continue
                
                # For this m, find the r that maximizes gain
                # We can vectorize this?
                # For a fixed m, we want to find r in [0, m) that maximizes sum(counts[r::m] == 0)
                
                # Optimization:
                # If m is large, there are few residues per class, but many classes.
                # If m is small, there are many residues per class, but few classes.
                
                # Let's just loop r for now.
                # To speed up, we can check if it's even possible to get gain > best_gain.
                # Max possible gain for m is N // m (roughly).
                # If N // m <= best_gain, skip m?
                # Not exactly, because some might already be covered. But it's an upper bound.
                
                # Upper bound check
                if N // m <= best_gain:
                    continue
                
                max_gain_for_m = 0
                best_r_for_m = 0
                
                # We can try to optimize this inner loop.
                # For small m, loop is short. For large m, loop is long but gain is small.
                
                for r in range(m):
                    indices = np.arange(r, N, m)
                    # Count how many are currently uncovered
                    gain = np.sum(counts[indices] == 0)
                    
                    if gain > max_gain_for_m:
                        max_gain_for_m = gain
                        best_r_for_m = r
                    # Early termination if we can't beat global best?
                    # No, because we need the best r for this m to compare with other m's.
                    # But if max_gain_for_m + (remaining potential) <= best_gain, we could break?
                    # Hard to estimate remaining potential quickly.
                    
                if max_gain_for_m > best_gain:
                    best_gain = max_gain_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_gain == 0:
                break
            
            # Add the best class
            used_moduli.add(best_m)
            classes.append([best_m, best_r])
            
            # Update counts
            indices = np.arange(best_r, N, best_m)
            counts[indices] += 1
            
            # Optional: Check if we covered everything
            if np.all(counts > 0):
                break
        
        # Calculate final score
        score = np.sum(counts == 0)
        return score, classes

    def local_search(classes, counts):
        # Coordinate descent: for each class, try all possible residues and swap if better.
        # Also try swapping moduli? No, moduli are distinct and fixed set?
        # The problem says "distinct moduli m > 1 dividing 51975".
        # It doesn't say we MUST use all divisors. We can choose a subset.
        # However, the seed program used all divisors.
        # The greedy above adds moduli one by one. It might stop before using all.
        # Local search should probably allow adding/removing moduli as well?
        # Or just optimize residues for the current set?
        # Let's stick to optimizing residues for the current set first.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, (m, r) in enumerate(classes):
                # Remove current class
                indices = np.arange(r, N, m)
                counts[indices] -= 1
                current_score = np.sum(counts == 0)
                
                # Try all other residues
                best_new_r = r
                best_new_score = current_score
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_indices = np.arange(new_r, N, m)
                    # Tentatively add
                    # We need to know how many new ones this covers.
                    # gain = np.sum(counts[new_indices] == 0)
                    # new_score = current_score - gain
                    # This is O(N/m) per r.
                    
                    # Optimization:
                    # We can precompute the "zero mask" for the current state?
                    # zero_mask = (counts == 0)
                    # gain = np.sum(zero_mask[new_indices])
                    
                    # Let's just compute it.
                    gain = np.sum(counts[new_indices] == 0)
                    new_score = current_score - gain
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_r = new_r
                
                # Apply best
                if best_new_r != r:
                    indices = np.arange(best_new_r, N, m)
                    counts[indices] += 1
                    classes[i][1] = best_new_r
                    improved = True
                else:
                    # Revert
                    indices = np.arange(r, N, m)
                    counts[indices] += 1
        
        score = np.sum(counts == 0)
        return score, classes

    # Run global greedy
    score, classes = run_greedy(divs)
    
    # Run local search
    if time.time() < deadline - 2:
        score, classes = local_search(classes, np.zeros(N, dtype=np.int16)) # Wait, counts was local to run_greedy
        # I need to return counts from run_greedy or recompute.
        # Let's refactor run_greedy to return counts.

    # Refactored approach:
    
    def run_greedy_v2(divs):
        counts = np.zeros(N, dtype=np.int16)
        used_moduli = set()
        classes = []
        
        while time.time() < deadline - 10:
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divs:
                if m in used_moduli:
                    continue
                if N // m <= best_gain:
                    continue
                
                max_gain_for_m = 0
                best_r_for_m = 0
                
                for r in range(m):
                    indices = np.arange(r, N, m)
                    gain = np.sum(counts[indices] == 0)
                    if gain > max_gain_for_m:
                        max_gain_for_m = gain
                        best_r_for_m = r
                
                if max_gain_for_m > best_gain:
                    best_gain = max_gain_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_gain == 0:
                break
            
            used_moduli.add(best_m)
            classes.append([best_m, best_r])
            indices = np.arange(best_r, N, best_m)
            counts[indices] += 1
            
            if np.all(counts > 0):
                break
        
        return classes, counts

    def local_search_v2(classes, counts):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(classes)):
                m, r = classes[i]
                indices = np.arange(r, N, m)
                counts[indices] -= 1
                current_score = np.sum(counts == 0)
                
                best_new_r = r
                best_new_score = current_score
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_indices = np.arange(new_r, N, m)
                    gain = np.sum(counts[new_indices] == 0)
                    new_score = current_score - gain
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_r = new_r
                
                if best_new_r != r:
                    indices = np.arange(best_new_r, N, m)
                    counts[indices] += 1
                    classes[i][1] = best_new_r
                    improved = True
                else:
                    indices = np.arange(r, N, m)
                    counts[indices] += 1
        
        return classes, counts

    # Execution
    classes, counts = run_greedy_v2(divs)
    
    if time.time() < deadline - 2:
        classes, counts = local_search_v2(classes, counts)
        
    # One more pass of local search if time permits
    if time.time() < deadline - 2:
        classes, counts = local_search_v2(classes, counts)

    return classes


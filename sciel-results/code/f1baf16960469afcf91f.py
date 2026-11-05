"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    base = 51975
    
    # Get all odd divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending
    divisors.sort(reverse=True)
    
    def compute_coverage(classes_list):
        """Compute coverage array from a list of classes."""
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes_list:
            cov[r::m] = 1
        return cov
    
    def uncovered_count(cov):
        """Count uncovered residues."""
        return int((cov == 0).sum())
    
    # Phase 1: Randomized greedy with adaptive look-ahead
    # We use a probability-weighted selection based on marginal gain
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    remaining_moduli = list(divisors)
    
    while remaining_moduli:
        if time.time() > deadline - 10:
            break
        
        # For each remaining modulus, compute the best marginal gain
        # We'll use a softmax-like selection: higher marginal gain -> higher probability
        gains = []
        best_residues = []
        
        for m in remaining_moduli:
            best_r = 0
            best_count = -1
            # For large moduli, sample residues; for small, try all
            if m <= 200:
                candidates = range(m)
            else:
                # Strategic candidates
                candidates = list(range(min(30, m)))
                for k in range(1, min(m, 15)):
                    candidates.append(k * m // 15)
                candidates = list(set(c % m for c in candidates))
                # Add random samples
                rng = np.random.RandomState(len(classes) * 12345 + m)
                for _ in range(20):
                    candidates.append(int(rng.randint(m)))
                candidates = list(set(candidates))
            
            for r in candidates:
                positions = np.arange(r, base, m)
                count = int((covered[positions] == 0).sum())
                if count > best_count:
                    best_count = count
                    best_r = r
            
            gains.append(best_count)
            best_residues.append(best_r)
        
        # Convert gains to probabilities using softmax-like weighting
        # Add small epsilon to avoid zero probabilities
        gains_arr = np.array(gains, dtype=np.float64) + 1e-6
        # Normalize
        total_gain = gains_arr.sum()
        if total_gain > 0:
            probs = gains_arr / total_gain
        else:
            probs = np.ones(len(remaining_moduli)) / len(remaining_moduli)
        
        # Select a modulus based on probabilities
        idx = np.random.choice(len(remaining_moduli), p=probs)
        m = remaining_moduli[idx]
        r = best_residues[idx]
        
        classes.append([m, r])
        remaining_moduli.pop(idx)
        covered[r::m] = 1
    
    current_score = uncovered_count(covered)
    
    # Phase 2: Local search with efficient delta computation
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Shuffle order for diversity
            indices = list(range(len(classes)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 15:
                    break
                m, r = classes[i]
                
                # Compute base coverage without this class
                # Efficient: remove this class's contribution
                old_vals = covered[r::m].copy()
                covered[r::m] = 0
                base_uncovered = int((covered == 0).sum())
                
                best_r = r
                best_uncovered = base_uncovered
                
                # Try all residues (or sampled for large m)
                if m <= 500:
                    candidates = range(m)
                else:
                    candidates = list(range(min(100, m)))
                    for k in range(1, min(m, 20)):
                        candidates.append(k * m // 20)
                    candidates = list(set(c % m for c in candidates))
                    rng = np.random.RandomState(i * 9999 + int(time.time() * 1000) % 100000)
                    for _ in range(50):
                        candidates.append(int(rng.randint(m)))
                    candidates = list(set(candidates))
                
                for new_r in candidates:
                    if new_r == r:
                        continue
                    # Count how many currently uncovered positions would be covered
                    new_cover_positions = np.arange(new_r, base, m)
                    would_cover = int((covered[new_cover_positions] == 0).sum())
                    new_uncovered = base_uncovered - would_cover
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = new_r
                
                # Restore
                covered[r::m] = old_vals
                
                if best_r != r:
                    classes[i][1] = best_r
                    covered[r::m] = 0
                    covered[best_r::m] = 1
                    current_score = best_uncovered
                    improved = True
    
    # Phase 3: Try removing classes that don't help
    if time.time() < deadline - 10:
        i = 0
        while i < len(classes) and time.time() < deadline - 10:
            m, r = classes[i]
            saved = classes.pop(i)
            # Recompute coverage
            covered_new = compute_coverage(classes)
            score_without = uncovered_count(covered_new)
            if score_without <= current_score:
                current_score = score_without
                covered = covered_new
            else:
                classes.insert(i, saved)
                i += 1
    
    # Phase 4: Final polish - more iterations of local search
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(classes)):
                if time.time() > deadline - 3:
                    break
                m, r = classes[i]
                
                old_vals = covered[r::m].copy()
                covered[r::m] = 0
                base_uncovered = int((covered == 0).sum())
                
                best_r = r
                best_uncovered = base_uncovered
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_cover_positions = np.arange(new_r, base, m)
                    would_cover = int((covered[new_cover_positions] == 0).sum())
                    new_uncovered = base_uncovered - would_cover
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = new_r
                
                covered[r::m] = old_vals
                
                if best_r != r:
                    classes[i][1] = best_r
                    covered[r::m] = 0
                    covered[best_r::m] = 1
                    current_score = best_uncovered
                    improved = True
    
    return classes
# EVOLVE-BLOCK-END

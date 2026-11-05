import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    L_max = 100000  # Cap as specified
    
    covered = np.zeros(L_max + 1, dtype=bool)
    indices = np.arange(1, L_max + 1)
    
    current_sum = Fraction(0)
    selected_classes = []
    used_primes = set()
    
    # Precompute residues for all primes
    residue_cache = {}
    for p in primes:
        residue_cache[p] = indices % p
    
    # Track the current frontier: largest L such that 1..L all covered
    frontier = 0
    
    while time.time() < deadline:
        # Check if we've already covered 1..999
        if frontier >= 999:
            break
            
        best_score = -1
        best_p = None
        best_r = None
        best_cost = None
        best_gain = 0
        
        # Only consider primes that fit in budget
        available_primes = [p for p in primes if p not in used_primes and current_sum + Fraction(1, p) <= 1]
        
        if not available_primes:
            break
        
        for p in available_primes:
            cost = Fraction(1, p)
            residues = residue_cache[p]
            
            # Find uncovered indices
            unc_mask = ~covered[1:L_max+1]
            if not np.any(unc_mask):
                break
            
            unc_residues = residues[unc_mask]
            counts = np.bincount(unc_residues, minlength=p)
            
            max_gain = int(np.max(counts))
            if max_gain == 0:
                continue
            
            # Score: prioritize covering numbers near the frontier
            # Get the residue with max count
            best_r_candidate = int(np.argmax(counts))
            
            # Compute how many numbers in 1..(frontier+1000) are newly covered
            # This focuses on extending the frontier
            window_end = min(frontier + 2000, L_max)
            window_start = 1
            window_mask = (indices >= window_start) & (indices <= window_end) & (~covered[1:L_max+1])
            if np.any(window_mask):
                window_residues = residues[window_mask]
                window_counts = np.bincount(window_residues, minlength=p)
                window_gain = int(window_counts[best_r_candidate])
            else:
                window_gain = 0
            
            # Combined score: emphasize frontier extension
            score = max_gain * 0.3 + window_gain * 7.0
            
            if score > best_score:
                best_score = score
                best_p = p
                best_r = best_r_candidate
                best_cost = cost
                best_gain = max_gain
        
        if best_p is None:
            break
        
        used_primes.add(best_p)
        current_sum += best_cost
        selected_classes.append([best_p, best_r])
        
        # Update covered array
        mask = (indices % best_p) == best_r
        covered[1:L_max+1][mask] = True
        
        # Update frontier
        # Find the first uncovered position
        false_indices = np.where(~covered[1:L_max+1])[0]
        if len(false_indices) == 0:
            frontier = L_max
        else:
            frontier = int(false_indices[0])
    
    # Calculate final L
    false_indices = np.where(~covered[1:L_max+1])[0]
    if len(false_indices) == 0:
        L = L_max
    else:
        L = int(false_indices[0])
    
    return selected_classes

import time
import numpy as np
from math import gcd
import random

BASE = 51975

def _get_divisors(n):
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted([d for d in divs if d > 1])

def _greedy_build(moduli, base):
    covered = np.zeros(base, dtype=np.int32)
    classes = []
    available = list(moduli)
    
    # We sort by descending size to prioritize large moduli? 
    # Or we just iterate through the provided order.
    for m in available:
        best_r = 0
        best_gain = -1
        
        # Optimization: if m is large, r::m is small. If m is small, r::m is large.
        # We can vectorize the gain calculation.
        # gain[r] = sum(covered[r::m] == 0)
        
        # Iterate residues
        # For speed, we can compute gains for all r at once?
        # Let's try a loop first, it's only m iterations. Max m=51975.
        # 51975 iterations * 47 moduli = ~2.4 million iterations.
        # Each iteration does a numpy sum over base/m elements.
        # Total ops: sum(m * (base/m)) = base * num_moduli = 51975 * 47 ~ 2.4M ops.
        # This is fast.
        
        for r in range(m):
            gain = np.sum(covered[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain <= 0:
            # No more useful moduli? 
            # We could break, but maybe a later small modulus helps?
            # If gain is 0, it means this modulus covers no *new* points.
            # It might still be useful in combination with swaps later?
            # For now, we add it if it's the best available, but if gain is 0, 
            # adding it doesn't change the score immediately.
            # However, we must use distinct moduli. 
            # The task says "construct() -> classes". It doesn't say we MUST use all.
            # "With every divisor of 51975 available once... minimize".
            # So we can choose a subset.
            # If gain is 0, it's useless for now. Skip it?
            # Yes, skip if gain is 0 to keep list small?
            # But maybe it becomes useful after others are optimized?
            # Let's keep it in the list for now, but if gain is 0, 
            # maybe we should just stop? 
            # No, stop adding if gain is 0.
            break
            
        classes.append([m, best_r])
        covered[best_r::m] += 1
        
    return classes, covered

def _local_search_residues(classes, covered, base, time_limit):
    start_time = time.time()
    current_score = np.sum(covered == 0)
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle classes to break symmetry?
        random.shuffle(classes)
        
        for i in range(len(classes)):
            if (time.time() - start_time) > time_limit:
                break
                
            m, r = classes[i]
            old_idx = np.arange(r, base, m)
            
            # Try all other residues
            # Optimization: If m is large, loop is long.
            # But m is a divisor of base.
            for r_new in range(m):
                if r_new == r:
                    continue
                
                new_idx = np.arange(r_new, base, m)
                
                # Calculate delta
                # Zeros lost: positions in old_idx where covered==1
                # Zeros gained: positions in new_idx where covered==0
                
                # We need to be careful if old_idx and new_idx overlap?
                # They are disjoint because r != r_new and 0 <= r, r_new < m.
                
                old_counts = covered[old_idx]
                new_counts = covered[new_idx]
                
                # If we move r -> r_new:
                # covered[old_idx] decreases by 1
                # covered[new_idx] increases by 1
                
                # Uncovered count changes by:
                # +1 for each x in old_idx where covered[x] becomes 0 (was 1)
                # -1 for each x in new_idx where covered[x] becomes 1 (was 0)
                
                delta = np.sum(old_counts == 1) - np.sum(new_counts == 0)
                
                if delta < 0:
                    # Improvement
                    covered[old_idx] -= 1
                    covered[new_idx] += 1
                    classes[i][1] = r_new
                    current_score += delta
                    improved = True
                    break # Restart sweep
    
    return classes, covered, current_score

def construct():
    start_time = time.time()
    deadline = 140 # Safety margin
    
    divisors = _get_divisors(BASE)
    
    # Phase 1: Greedy with Descending Moduli
    classes, covered = _greedy_build(sorted(divisors, reverse=True), BASE)
    score = np.sum(covered == 0)
    
    # Phase 2: Local Search
    if time.time() < deadline - 20:
        classes, covered, score = _local_search_residues(classes, covered, BASE, time_limit=deadline - time.time() - 20)
        
    # Phase 3: Try another greedy start with Ascending Moduli if time permits
    # And keep the best
    if time.time() < deadline - 40:
        classes2, covered2 = _greedy_build(sorted(divisors), BASE)
        score2 = np.sum(covered2 == 0)
        
        if score2 < score:
            classes, covered, score = classes2, covered2, score2
            
        if time.time() < deadline - 10:
            classes, covered, score = _local_search_residues(classes, covered, BASE, time_limit=deadline - time.time() - 10)
            
    # Phase 4: Random Restarts of Greedy + Local Search
    best_classes = classes
    best_score = score
    
    while time.time() < deadline - 10:
        random.shuffle(divisors)
        classes_r, covered_r = _greedy_build(divisors, BASE)
        score_r = np.sum(covered_r == 0)
        
        # Quick local search
        if score_r < best_score or (score_r == best_score and time.time() < deadline - 20):
            if time.time() < deadline - 20:
                classes_r, covered_r, score_r = _local_search_residues(classes_r, covered_r, BASE, time_limit=min(20, deadline - time.time() - 10))
            
            if score_r < best_score:
                best_score = score_r
                best_classes = classes_r
                
    return best_classes

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _compute_covered(classes, base):
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def _score_solution(classes, base):
    covered = _compute_covered(classes, base)
    return int(np.sum(1 - covered))

def _greedy_construct(divisors, base, seed=0):
    rng = random.Random(seed)
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    divs = sorted(divisors)
    for m in divs:
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = int(np.sum(covered[r::m] == 0))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, divisors, base, max_time=10.0):
    start_time = time.time()
    covered = _compute_covered(classes, base)
    current_score = int(np.sum(covered == 0))
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try single-residue moves: for each class, try all residues, find best
        for i, (m, r) in enumerate(classes):
            # Remove this class temporarily
            covered[r::m] = 0
            base_uncovered = int(np.sum(covered == 0))
            
            best_new_r = r
            best_new_score = base_uncovered  # if we don't change, we get base_uncovered + coverage from this class
            
            # Actually, if we remove and don't re-add, score is base_uncovered
            # If we re-add with new_r, score is base_uncovered - (newly covered that weren't covered before)
            for new_r in range(m):
                # Count how many residues in new_r::m are currently uncovered
                newly_covered = int(np.sum(covered[new_r::m] == 0))
                new_score = base_uncovered - newly_covered
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_r = new_r
            
            # Re-add with best residue
            covered[best_new_r::m] = 1
            if best_new_r != r:
                classes[i] = [m, best_new_r]
                current_score = best_new_score
                improved = True
                break
        
        if not improved:
            break
    
    return classes, current_score

def construct():
    base = 51975
    divisors = _get_divisors(base)
    
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with greedy
    classes = _greedy_construct(divisors, base, seed=0)
    best_classes = classes[:]
    best_score = _score_solution(classes, base)
    
    # Local search
    classes, score = _local_search(classes, divisors, base, max_time=30.0)
    if score < best_score:
        best_score = score
        best_classes = classes[:]
    
    # Try a few more restarts if time permits
    for seed in range(1, 20):
        if time.time() > deadline - 20.0:
            break
        classes = _greedy_construct(divisors, base, seed=seed)
        score = _score_solution(classes, base)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
        
        classes, score = _local_search(classes, divisors, base, max_time=10.0)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    return best_classes
# EVOLVE-BLOCK-END

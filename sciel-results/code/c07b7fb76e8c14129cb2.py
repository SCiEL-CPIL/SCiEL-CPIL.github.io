import numpy as np
import time
import random
from math import gcd
from itertools import product

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def _uncovered_count(classes, base=BASE):
    """Compute number of uncovered residues."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _compute_covered(classes):
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    divs = [d for d in divs if d > 1]
    
    # Greedy initialization: pick residue that covers most uncovered residues
    def greedy_init(moduli_set, order='desc'):
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        if order == 'desc':
            sorted_mods = sorted(moduli_set, reverse=True)
        elif order == 'asc':
            sorted_mods = sorted(moduli_set)
        else:
            sorted_mods = list(moduli_set)
            random.shuffle(sorted_mods)
        for m in sorted_mods:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = np.sum(covered[r::m] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes
    
    # Local search: for each modulus, try to find a better residue
    # Uses refcount-based incremental evaluation for speed
    def local_search(classes, time_budget):
        covered = _compute_covered(classes)
        current_uncovered = int(np.sum(covered == 0))
        
        # Build refcount array: how many classes cover each residue
        refcount = np.zeros(BASE, dtype=np.int16)
        for m, r in classes:
            refcount[r::m] += 1
        
        improved = True
        max_rounds = 100
        round_num = 0
        
        while improved and time.time() < time_budget and round_num < max_rounds:
            improved = False
            round_num += 1
            current_uncovered = int(np.sum(covered == 0))
            
            # Shuffle order for diversification
            indices = list(range(len(classes)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() >= time_budget:
                    break
                m, r = classes[idx]
                
                best_r = r
                best_uncovered = current_uncovered
                
                # Try all residues for small moduli, sample for large
                if m <= 500:
                    candidate_rs = list(range(m))
                    random.shuffle(candidate_rs)
                else:
                    n_samples = min(m, 300)
                    candidate_rs = random.sample(range(m), n_samples)
                    candidate_rs.append(r)
                
                for r_new in candidate_rs:
                    if r_new == r:
                        continue
                    if time.time() >= time_budget:
                        break
                    
                    # Incremental update using refcounts
                    # Remove old coverage contribution
                    old_positions = np.arange(r, BASE, m)
                    new_positions = np.arange(r_new, BASE, m)
                    
                    # For positions that lose their last cover
                    old_lose = old_positions[refcount[old_positions] == 1]
                    # For positions that gain a new cover
                    new_gain = new_positions[refcount[new_positions] == 0]
                    
                    net_change = len(new_gain) - len(old_lose)
                    if net_change >= 0:
                        continue  # No improvement possible
                    
                    # Apply change
                    refcount[old_positions] -= 1
                    refcount[new_positions] += 1
                    classes[idx][1] = r_new
                    
                    new_uncovered = current_uncovered + net_change
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new
                    
                    # Keep the change (we're doing steepest descent within this round)
                    current_uncovered = new_uncovered
                
                if best_r != r:
                    covered = _compute_covered(classes)
                    current_uncovered = int(np.sum(covered == 0))
                    improved = True
        
        return classes, current_uncovered
    
    # Try multiple initializations and local searches
    best_classes = None
    best_score = BASE + 1
    
    orders = ['desc', 'asc', 'rand', 'rand', 'rand', 'rand']
    
    for order in orders:
        if time.time() >= deadline:
            break
        classes = greedy_init(divs, order=order)
        classes, score = local_search(classes, deadline)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    if best_classes is None:
        best_classes = greedy_init(divs, order='desc')
        best_score = _uncovered_count(best_classes)
    
    # Final polish: try pair swaps and single moves with full recomputation
    current_classes = best_classes[:]
    current_score = best_score
    
    # Single move polish
    for _ in range(3):
        if time.time() >= deadline:
            break
        improved = False
        indices = list(range(len(current_classes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            m, r = current_classes[idx]
            
            # Try all residues for this modulus
            best_r = r
            best_score_local = current_score
            
            if m <= 500:
                candidate_rs = list(range(m))
                random.shuffle(candidate_rs)
            else:
                candidate_rs = random.sample(range(m), min(m, 400))
                candidate_rs.append(r)
            
            for r_new in candidate_rs:
                if r_new == r:
                    continue
                if time.time() >= deadline:
                    break
                
                old_r = current_classes[idx][1]
                current_classes[idx][1] = r_new
                new_score = _uncovered_count(current_classes)
                current_classes[idx][1] = old_r
                
                if new_score < best_score_local:
                    best_score_local = new_score
                    best_r = r_new
            
            if best_r != r:
                current_classes[idx][1] = best_r
                current_score = best_score_local
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_classes = current_classes[:]
        
        if not improved:
            break
    
    return best_classes

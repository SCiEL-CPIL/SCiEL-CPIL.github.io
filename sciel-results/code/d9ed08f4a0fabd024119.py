import numpy as np
import time
import random
from math import gcd

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

def _compute_refcounts(classes):
    """Compute reference count for each residue: how many classes cover it."""
    refcount = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        refcount[r::m] += 1
    return refcount

def construct():
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    divs = [d for d in divs if d > 1]
    n_classes = len(divs)
    
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
    
    # Try multiple initializations
    best_classes = None
    best_score = BASE + 1
    
    for order in ['desc', 'asc', 'rand', 'rand', 'rand', 'rand']:
        if time.time() >= deadline:
            break
        classes = greedy_init(divs, order=order)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    if best_classes is None:
        best_classes = greedy_init(divs, order='desc')
        best_score = _uncovered_count(best_classes)
    
    # Local search with reference-count-based incremental updates
    current_classes = best_classes[:]
    current_refcount = _compute_refcounts(current_classes)
    current_uncovered = int(np.sum(current_refcount == 0))
    
    improved = True
    max_rounds = 100
    round_num = 0
    while improved and time.time() < deadline and round_num < max_rounds:
        improved = False
        round_num += 1
        current_uncovered = int(np.sum(current_refcount == 0))
        
        # Shuffle order for diversification
        indices = list(range(n_classes))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            m, r = current_classes[idx]
            
            # Find best residue for this modulus using refcount
            best_r = r
            best_uncovered = current_uncovered
            
            # Determine candidate residues
            if m <= 500:
                candidate_rs = list(range(m))
            else:
                # Use more samples for large moduli
                n_samples = min(m, 500)
                candidate_rs = random.sample(range(m), n_samples)
                if r not in candidate_rs:
                    candidate_rs.append(r)
            
            for r_new in candidate_rs:
                if r_new == r:
                    continue
                
                # Quick filter using refcount
                # Residues that become uncovered: those with refcount==1 in old slice
                # Residues that become covered: those with refcount==0 in new slice
                old_slice = current_refcount[r::m]
                new_slice = current_refcount[r_new::m]
                
                # How many residues go from covered to uncovered (refcount 1 -> 0)
                lost = np.sum(old_slice == 1)
                # How many residues go from uncovered to covered (refcount 0 -> 1)  
                gained = np.sum(new_slice == 0)
                
                # Net change in uncovered count
                delta_uncovered = lost - gained
                if delta_uncovered >= 0:
                    continue
                
                # This is a promising move; verify with full refcount update
                # Apply the change temporarily
                current_refcount[r::m] -= 1
                current_refcount[r_new::m] += 1
                new_uncovered = int(np.sum(current_refcount == 0))
                # Revert
                current_refcount[r::m] += 1
                current_refcount[r_new::m] -= 1
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = r_new
            
            if best_r != r:
                # Apply the best move
                current_refcount[r::m] -= 1
                current_refcount[best_r::m] += 1
                current_classes[idx][1] = best_r
                current_uncovered = best_uncovered
                improved = True
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]
    
    return best_classes

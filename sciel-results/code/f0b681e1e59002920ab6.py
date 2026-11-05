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
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _compute_refcounts(classes):
    refcount = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        refcount[r::m] += 1
    return refcount

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
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
    
    for order in ['desc', 'asc', 'rand', 'rand', 'rand']:
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
    
    # Phase 1: Fast residue-only local search
    current_classes = best_classes[:]
    current_refcount = _compute_refcounts(current_classes)
    current_uncovered = int(np.sum(current_refcount == 0))
    
    improved = True
    max_rounds = 50
    round_num = 0
    while improved and time.time() < deadline - 30 and round_num < max_rounds:
        improved = False
        round_num += 1
        current_uncovered = int(np.sum(current_refcount == 0))
        
        indices = list(range(n_classes))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline - 30:
                break
            m, r = current_classes[idx]
            
            best_r = r
            best_uncovered = current_uncovered
            
            if m <= 300:
                candidate_rs = list(range(m))
            else:
                n_samples = min(m, 400)
                candidate_rs = random.sample(range(m), n_samples)
                if r not in candidate_rs:
                    candidate_rs.append(r)
            
            for r_new in candidate_rs:
                if r_new == r:
                    continue
                
                old_slice = current_refcount[r::m]
                new_slice = current_refcount[r_new::m]
                
                lost = np.sum(old_slice == 1)
                gained = np.sum(new_slice == 0)
                
                delta_uncovered = lost - gained
                if delta_uncovered >= 0:
                    continue
                
                current_refcount[r::m] -= 1
                current_refcount[r_new::m] += 1
                new_uncovered = int(np.sum(current_refcount == 0))
                current_refcount[r::m] += 1
                current_refcount[r_new::m] -= 1
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = r_new
            
            if best_r != r:
                current_refcount[r::m] -= 1
                current_refcount[best_r::m] += 1
                current_classes[idx][1] = best_r
                current_uncovered = best_uncovered
                improved = True
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]
    
    # Phase 2: Modulus swap local search (use remaining time)
    if time.time() < deadline - 5:
        current_classes = best_classes[:]
        current_refcount = _compute_refcounts(current_classes)
        current_uncovered = int(np.sum(current_refcount == 0))
        
        # Build modulus-to-index map
        mod_to_idx = {}
        for i, (m, r) in enumerate(current_classes):
            mod_to_idx[m] = i
        
        swap_improved = True
        swap_rounds = 0
        max_swap_rounds = 20
        while swap_improved and time.time() < deadline - 5 and swap_rounds < max_swap_rounds:
            swap_improved = False
            swap_rounds += 1
            
            indices = list(range(n_classes))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() >= deadline - 5:
                    break
                m1, r1 = current_classes[idx]
                
                # Try swapping this modulus with another
                for jdx in indices:
                    if jdx == idx or time.time() >= deadline - 5:
                        continue
                    m2, r2 = current_classes[jdx]
                    
                    # Quick filter: only try swaps where moduli are "close" in size
                    if abs(m1 - m2) > max(m1, m2) * 0.5:
                        continue
                    
                    # Try all residue combinations for the swap
                    # Sample residues for efficiency
                    if m1 <= 100:
                        rs1 = list(range(m1))
                    else:
                        rs1 = random.sample(range(m1), min(m1, 100))
                        if r1 not in rs1:
                            rs1.append(r1)
                    
                    if m2 <= 100:
                        rs2 = list(range(m2))
                    else:
                        rs2 = random.sample(range(m2), min(m2, 100))
                        if r2 not in rs2:
                            rs2.append(r2)
                    
                    best_pair = None
                    best_new_uncovered = current_uncovered
                    
                    # Only check a subset of residue pairs for speed
                    n_pairs = min(len(rs1) * len(rs2), 200)
                    if len(rs1) * len(rs2) <= 200:
                        pairs = [(r1, r2) for r1 in rs1 for r2 in rs2]
                    else:
                        pairs = random.sample([(r1, r2) for r1 in rs1 for r2 in rs2], 200)
                    
                    for nr1, nr2 in pairs:
                        # Temporarily apply swap
                        current_refcount[r1::m1] -= 1
                        current_refcount[r2::m2] -= 1
                        current_refcount[nr1::m1] += 1
                        current_refcount[nr2::m2] += 1
                        
                        new_uncovered = int(np.sum(current_refcount == 0))
                        
                        # Revert
                        current_refcount[nr1::m1] -= 1
                        current_refcount[nr2::m2] -= 1
                        current_refcount[r1::m1] += 1
                        current_refcount[r2::m2] += 1
                        
                        if new_uncovered < best_new_uncovered:
                            best_new_uncovered = new_uncovered
                            best_pair = (nr1, nr2)
                    
                    if best_pair is not None:
                        nr1, nr2 = best_pair
                        # Apply the swap
                        current_refcount[r1::m1] -= 1
                        current_refcount[r2::m2] -= 1
                        current_refcount[nr1::m1] += 1
                        current_refcount[nr2::m2] += 1
                        current_classes[idx][1] = nr1
                        current_classes[jdx][1] = nr2
                        current_uncovered = best_new_uncovered
                        swap_improved = True
                        if current_uncovered < best_score:
                            best_score = current_uncovered
                            best_classes = current_classes[:]
    
    # Final: one more quick residue pass with whatever time remains
    if time.time() < deadline - 2:
        current_classes = best_classes[:]
        current_refcount = _compute_refcounts(current_classes)
        current_uncovered = int(np.sum(current_refcount == 0))
        
        indices = list(range(n_classes))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline - 2:
                break
            m, r = current_classes[idx]
            
            best_r = r
            best_uncovered = current_uncovered
            
            if m <= 200:
                candidate_rs = list(range(m))
            else:
                n_samples = min(m, 300)
                candidate_rs = random.sample(range(m), n_samples)
                if r not in candidate_rs:
                    candidate_rs.append(r)
            
            for r_new in candidate_rs:
                if r_new == r:
                    continue
                
                old_slice = current_refcount[r::m]
                new_slice = current_refcount[r_new::m]
                
                lost = np.sum(old_slice == 1)
                gained = np.sum(new_slice == 0)
                
                delta_uncovered = lost - gained
                if delta_uncovered >= 0:
                    continue
                
                current_refcount[r::m] -= 1
                current_refcount[r_new::m] += 1
                new_uncovered = int(np.sum(current_refcount == 0))
                current_refcount[r::m] += 1
                current_refcount[r_new::m] -= 1
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = r_new
            
            if best_r != r:
                current_refcount[r::m] -= 1
                current_refcount[best_r::m] += 1
                current_classes[idx][1] = best_r
                current_uncovered = best_uncovered
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]
    
    return best_classes

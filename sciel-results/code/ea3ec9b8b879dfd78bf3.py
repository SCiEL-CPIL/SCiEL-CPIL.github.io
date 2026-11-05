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

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    divs = [d for d in divs if d > 1]
    num_moduli = len(divs)
    
    def greedy_init(moduli_list, order='asc'):
        """Greedy initialization using coverage count array."""
        counts = np.zeros(BASE, dtype=np.int8)
        residues = [0] * len(moduli_list)
        
        if order == 'asc':
            sorted_mods = sorted(moduli_list)
        elif order == 'desc':
            sorted_mods = sorted(moduli_list, reverse=True)
        else:
            sorted_mods = list(moduli_list)
            random.shuffle(sorted_mods)
        
        for i, m in enumerate(sorted_mods):
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[moduli_list.index(m)] = best_r
            counts[best_r::m] += 1
        
        return moduli_list, residues, counts
    
    def compute_uncovered(counts):
        return int(np.count_nonzero(counts == 0))
    
    def local_search(moduli_list, residues, counts, max_rounds=30, deadline_check=None):
        """Local search with incremental updates using coverage counts."""
        current_uncovered = compute_uncovered(counts)
        improved = True
        round_num = 0
        
        while improved and round_num < max_rounds:
            if deadline_check and deadline_check():
                break
            improved = False
            round_num += 1
            
            # Shuffle order for diversification
            indices = list(range(len(moduli_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if deadline_check and deadline_check():
                    break
                
                m = moduli_list[idx]
                old_r = residues[idx]
                best_r = old_r
                best_uncovered = current_uncovered
                
                # Try all residues for small moduli, sample for large
                if m <= 500:
                    candidate_rs = list(range(m))
                else:
                    n_samples = min(m, 300)
                    candidate_rs = list(random.sample(range(m), n_samples))
                    candidate_rs.append(old_r)
                
                for r_new in candidate_rs:
                    if r_new == old_r:
                        continue
                    
                    # Incremental update: compute delta in uncovered count
                    # Remove old coverage
                    old_positions = np.arange(old_r, BASE, m)
                    new_positions = np.arange(r_new, BASE, m)
                    
                    # Count how many positions in old_positions have count == 1 (will become uncovered)
                    # Count how many positions in new_positions have count == 0 (will become covered)
                    old_will_uncover = np.count_nonzero(counts[old_positions] == 1)
                    new_will_cover = np.count_nonzero(counts[new_positions] == 0)
                    
                    delta = old_will_uncover - new_will_cover
                    if delta >= 0:
                        continue
                    
                    # Apply the change
                    counts[old_positions] -= 1
                    counts[new_positions] += 1
                    
                    new_uncovered = current_uncovered + delta
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new
                    else:
                        # Revert
                        counts[old_positions] += 1
                        counts[new_positions] -= 1
                
                if best_r != old_r:
                    # Re-apply the best change
                    old_positions = np.arange(old_r, BASE, m)
                    new_positions = np.arange(best_r, BASE, m)
                    counts[old_positions] -= 1
                    counts[new_positions] += 1
                    residues[idx] = best_r
                    current_uncovered = best_uncovered
                    improved = True
        
        return residues, counts, current_uncovered
    
    def local_search_with_swaps(moduli_list, residues, counts, max_rounds=20, deadline_check=None):
        """Local search that also tries swapping residues between moduli."""
        current_uncovered = compute_uncovered(counts)
        improved = True
        round_num = 0
        
        while improved and round_num < max_rounds:
            if deadline_check and deadline_check():
                break
            improved = False
            round_num += 1
            
            indices = list(range(len(moduli_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if deadline_check and deadline_check():
                    break
                
                m = moduli_list[idx]
                old_r = residues[idx]
                best_r = old_r
                best_uncovered = current_uncovered
                
                if m <= 500:
                    candidate_rs = list(range(m))
                else:
                    n_samples = min(m, 300)
                    candidate_rs = list(random.sample(range(m), n_samples))
                    candidate_rs.append(old_r)
                
                for r_new in candidate_rs:
                    if r_new == old_r:
                        continue
                    
                    old_positions = np.arange(old_r, BASE, m)
                    new_positions = np.arange(r_new, BASE, m)
                    
                    old_will_uncover = np.count_nonzero(counts[old_positions] == 1)
                    new_will_cover = np.count_nonzero(counts[new_positions] == 0)
                    
                    delta = old_will_uncover - new_will_cover
                    if delta >= 0:
                        continue
                    
                    counts[old_positions] -= 1
                    counts[new_positions] += 1
                    
                    new_uncovered = current_uncovered + delta
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new
                    else:
                        counts[old_positions] += 1
                        counts[new_positions] -= 1
                
                if best_r != old_r:
                    old_positions = np.arange(old_r, BASE, m)
                    new_positions = np.arange(best_r, BASE, m)
                    counts[old_positions] -= 1
                    counts[new_positions] += 1
                    residues[idx] = best_r
                    current_uncovered = best_uncovered
                    improved = True
        
        return residues, counts, current_uncovered
    
    best_score = BASE + 1
    best_classes = None
    
    # Try multiple initializations
    for order in ['asc', 'desc', 'rand', 'rand', 'rand']:
        if time.time() >= deadline:
            break
        
        moduli_list, residues, counts = greedy_init(divs, order=order)
        current_uncovered = compute_uncovered(counts)
        
        # Local search
        residues, counts, current_uncovered = local_search(
            divs, residues, counts, 
            max_rounds=20,
            deadline_check=lambda: time.time() >= deadline
        )
        
        if current_uncovered < best_score:
            best_score = current_uncovered
            best_classes = [[divs[i], residues[i]] for i in range(num_moduli)]
    
    if best_classes is None:
        moduli_list, residues, counts = greedy_init(divs, order='asc')
        best_classes = [[divs[i], residues[i]] for i in range(num_moduli)]
        best_score = compute_uncovered(counts)
    
    # Additional local search from the best solution found
    if time.time() < deadline:
        residues = [best_classes[i][1] for i in range(num_moduli)]
        counts = np.zeros(BASE, dtype=np.int8)
        for i in range(num_moduli):
            m = divs[i]
            r = residues[i]
            counts[r::m] += 1
        
        residues, counts, current_uncovered = local_search(
            divs, residues, counts,
            max_rounds=30,
            deadline_check=lambda: time.time() >= deadline
        )
        
        if current_uncovered < best_score:
            best_score = current_uncovered
            best_classes = [[divs[i], residues[i]] for i in range(num_moduli)]
    
    return best_classes

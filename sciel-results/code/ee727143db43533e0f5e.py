import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    
    base = 51975
    
    # Get all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # Precompute residue class indices as numpy arrays for speed
    class_indices = {}
    for m in divisors:
        class_indices[m] = [np.array(range(r, base, m), dtype=np.int32) for r in range(m)]
    
    def count_uncovered(covered):
        return int(np.count_nonzero(covered == 0))
    
    def greedy_initial(order):
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() >= deadline:
                break
            best_r = 0
            best_count = -1
            for r in range(m):
                idxs = class_indices[m][r]
                cnt = int(np.count_nonzero(covered[idxs] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[class_indices[m][best_r]] = 1
        return classes, covered, count_uncovered(covered)
    
    # Try multiple greedy orderings
    best_classes = None
    best_uncovered = base
    best_covered = None
    
    # Standard increasing order
    classes, covered, unc = greedy_initial(divisors)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = [c[:] for c in classes]
        best_covered = covered.copy()
    
    # Try decreasing order
    if time.time() < deadline:
        classes, covered, unc = greedy_initial(sorted(divisors, reverse=True))
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
    
    # Try random orderings
    random.seed(42)
    for attempt in range(30):
        if time.time() >= deadline - 80:
            break
        random_order = divisors.copy()
        random.shuffle(random_order)
        classes, covered, unc = greedy_initial(random_order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c[:] for c in classes]
            best_covered = covered.copy()
    
    if best_classes is None:
        best_classes, best_covered, best_uncovered = greedy_initial(divisors)
    
    # Local search with efficient numpy-based move evaluation
    classes = [c[:] for c in best_classes]
    covered = best_covered.copy()
    uncovered = best_uncovered
    
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in divisors if m not in used_moduli]
    
    def compute_cover_count(classes):
        """Compute how many classes cover each residue"""
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in classes:
            cover_count[class_indices[m][r]] += 1
        return cover_count
    
    cover_count = compute_cover_count(classes)
    
    # Precompute masks for fast residue change evaluation
    # For each modulus m, precompute boolean masks for each residue
    residue_masks = {}
    for m in divisors:
        residue_masks[m] = [np.zeros(base, dtype=bool) for _ in range(m)]
        for r in range(m):
            residue_masks[m][r][class_indices[m][r]] = True
    
    def eval_residue_change(idx, old_r, new_r):
        """Evaluate changing residue of class idx from old_r to new_r using numpy masks"""
        m = classes[idx][0]
        old_mask = residue_masks[m][old_r]
        new_mask = residue_masks[m][new_r]
        
        # Losing: in old but not new
        losing = old_mask & ~new_mask
        # Gaining: in new but not old
        gaining = new_mask & ~old_mask
        
        # Residues in losing that have cover_count == 1 become uncovered (+1 each)
        # Residues in gaining that have cover_count == 0 become covered (-1 each)
        delta = int(np.sum(cover_count[losing] == 1)) - int(np.sum(cover_count[gaining] == 0))
        return delta
    
    def apply_residue_change(idx, old_r, new_r):
        """Apply residue change and update cover_count"""
        m = classes[idx][0]
        old_idxs = class_indices[m][old_r]
        new_idxs = class_indices[m][new_r]
        cover_count[old_idxs] -= 1
        cover_count[new_idxs] += 1
        classes[idx][1] = new_r
    
    improved = True
    max_rounds = 100
    round_num = 0
    
    while improved and time.time() < deadline - 5 and round_num < max_rounds:
        improved = False
        round_num += 1
        
        # Try changing residue for each existing class
        for idx in range(len(classes)):
            if time.time() >= deadline - 5:
                break
            m, r = classes[idx]
            best_delta = 0
            best_new_r = None
            
            for new_r in range(m):
                if new_r == r:
                    continue
                delta = eval_residue_change(idx, r, new_r)
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = new_r
            
            if best_new_r is not None:
                apply_residue_change(idx, r, best_new_r)
                uncovered += best_delta
                improved = True
        
        # Try adding a new modulus
        if time.time() < deadline - 5 and available_moduli:
            best_add_delta = 0
            best_add = None
            
            for m in available_moduli:
                if time.time() >= deadline - 5:
                    break
                best_r_for_m = None
                best_delta_for_m = 0
                
                for r in range(m):
                    idxs = class_indices[m][r]
                    gain = int(np.count_nonzero(cover_count[idxs] == 0))
                    if gain > best_delta_for_m:
                        best_delta_for_m = gain
                        best_r_for_m = r
                
                if best_delta_for_m > best_add_delta:
                    best_add_delta = best_delta_for_m
                    best_add = (m, best_r_for_m)
            
            if best_add is not None:
                m, r = best_add
                classes.append([m, r])
                idxs = class_indices[m][r]
                cover_count[idxs] += 1
                uncovered -= best_add_delta
                used_moduli.add(m)
                available_moduli.remove(m)
                improved = True
        
        # Try removing a modulus if it doesn't increase uncovered
        if time.time() < deadline - 5:
            for idx in range(len(classes)):
                if time.time() >= deadline - 5:
                    break
                m, r = classes[idx]
                idxs = class_indices[m][r]
                loss = int(np.count_nonzero(cover_count[idxs] == 1))
                if loss == 0:
                    cover_count[idxs] -= 1
                    classes.pop(idx)
                    used_moduli.discard(m)
                    if m not in available_moduli:
                        available_moduli.append(m)
                    improved = True
                    break
    
    # Final coverage recomputation
    final_covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        final_covered[class_indices[m][r]] = 1
    final_uncovered = count_uncovered(final_covered)
    
    return classes

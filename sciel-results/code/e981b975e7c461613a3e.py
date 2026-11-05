"""Combined program: greedy construction with randomized local search polish."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975
    # All divisors > 1 of base (odd divisors since base is odd)
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size ascending for greedy
    divisors_sorted = sorted(divisors)
    
    def score_of(classes):
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(base - covered.sum())
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick best residue for each."""
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() > deadline - 20:
                break
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(covered[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, covered
    
    def local_search(classes, covered, max_iterations=500):
        """Iteratively try to improve by changing residues or adding/removing classes."""
        if not classes:
            return classes, covered
        sc = int(base - covered.sum())
        used_moduli = set(m for m, _ in classes)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Try changing residue of existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    return classes, covered
                m, r = classes[i]
                best_r = r
                best_sc = sc
                # Try all residues if small, else sample
                if m <= 30:
                    candidates = list(range(m))
                else:
                    candidates = [r] + [random.randint(0, m - 1) for _ in range(40)]
                for tr in candidates:
                    if tr == r:
                        continue
                    # Compute the new coverage efficiently
                    # Remove class i's contribution, add tr's contribution
                    # Count how many positions change from covered to uncovered and vice versa
                    # A position x is covered by class i if (x - r) % m == 0
                    # It is covered by the new class if (x - tr) % m == 0
                    # Positions covered by old but not new: (x-r)%m==0 and (x-tr)%m!=0
                    # Positions covered by new but not old: (x-r)%m!=0 and (x-tr)%m==0
                    # But we also need to consider other classes
                    # Simplest: compute delta = new_sc - sc
                    # new_sc = base - sum(covered_after_change)
                    # Let's compute: for each position, is it covered by other classes?
                    # This is expensive. Instead, let's just rebuild for small number of classes.
                    classes[i][1] = tr
                    new_covered_arr = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        if j == i:
                            continue
                        new_covered_arr[rj::mj] = 1
                    # Now add the new class
                    newly_covered_by_i = (new_covered_arr[tr::m] == 0).sum()
                    # Positions that were covered only by class i (now uncovered)
                    old_only = (covered[r::m] == 1) & (new_covered_arr[r::m] == 0)
                    old_only_count = int(old_only.sum())
                    # New score
                    new_sc = sc - int(newly_covered_by_i) + old_only_count
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_r = tr
                        # Update covered
                        covered[tr::m] = 1
                        # Also need to un-cover positions that were only covered by old r
                        # But we can't easily do that with uint8. Let's rebuild.
                        covered = new_covered_arr.copy()
                        covered[tr::m] = 1
                    else:
                        classes[i][1] = r
                if best_r != r:
                    sc = best_sc
                    improved = True
            
            # Try removing a class if it doesn't reduce coverage
            if time.time() < deadline - 15:
                for i in range(len(classes)):
                    if time.time() > deadline - 10:
                        break
                    m, r = classes[i]
                    # Check if removing this class changes coverage
                    other_covered = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        if j == i:
                            continue
                        other_covered[rj::mj] = 1
                    # Positions covered only by class i
                    only_by_i = (covered[r::m] == 1) & (other_covered[r::m] == 0)
                    if only_by_i.sum() == 0:
                        # Safe to remove
                        classes.pop(i)
                        used_moduli.discard(m)
                        covered = other_covered
                        sc = int(base - covered.sum())
                        improved = True
                        break  # Restart loop since indices changed
            
            # Try adding an unused modulus
            if time.time() < deadline - 15:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    # Pick a few candidates
                    random.shuffle(unused)
                    for m in unused[:5]:
                        if time.time() > deadline - 10:
                            break
                        best_r = 0
                        best_gain = -1
                        # Try more candidates for larger moduli
                        n_trials = min(50, m)
                        for _ in range(n_trials):
                            r = random.randint(0, m - 1)
                            gain = int(np.sum(covered[r::m] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_r = r
                        # Also check r=0 and r=1 explicitly
                        for r in [0, 1]:
                            if r < m:
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        if best_gain > 0:
                            classes.append([m, best_r])
                            covered[best_r::m] = 1
                            used_moduli.add(m)
                            sc = int(base - covered.sum())
                            improved = True
        
        return classes, covered

    best_classes = []
    best_score = base + 1
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_greedy = 15
    for attempt in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        if attempt == 0:
            order = divisors_sorted
        elif attempt == 1:
            order = sorted(divisors, reverse=True)
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)
        
        classes, covered = greedy_construct(order)
        sc = int(base - covered.sum())
        if sc < best_score:
            best_score = sc
            best_classes = classes[:]
        
        # Phase 2: Local search on this solution
        if time.time() < deadline - 20:
            classes_ls, covered_ls = local_search(classes, covered, max_iterations=100)
            sc_ls = int(base - covered_ls.sum())
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]
    
    # Phase 3: More aggressive local search on the best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered_best[r::m] = 1
        best_classes, covered_best = local_search(best_classes, covered_best, max_iterations=500)
        final_sc = int(base - covered_best.sum())
        if final_sc < best_score:
            best_score = final_sc
    
    # Phase 4: Final high-precision polish - try all residues for small moduli
    if best_classes and time.time() < deadline - 5:
        covered_final = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered_final[r::m] = 1
        sc_final = int(base - covered_final.sum())
        
        # Try changing each class to its best residue (full search for small m)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(best_classes)):
                if time.time() > deadline - 3:
                    break
                m, r = best_classes[i]
                if m > 100:
                    continue  # Skip large moduli for full search
                best_r = r
                best_sc = sc_final
                for tr in range(m):
                    if tr == r:
                        continue
                    # Compute new score
                    classes_copy = best_classes[:]
                    classes_copy[i][1] = tr
                    new_covered = np.zeros(base, dtype=np.uint8)
                    for mj, rj in classes_copy:
                        new_covered[rj::mj] = 1
                    new_sc = int(base - new_covered.sum())
                    if new_sc < best_sc:
                        best_sc = new_sc
                        best_r = tr
                        covered_final = new_covered
                if best_r != r:
                    best_classes[i][1] = best_r
                    sc_final = best_sc
                    improved = True
        
        if sc_final < best_score:
            best_score = sc_final
    
    if not best_classes:
        best_classes = [[3, 0]]
    
    return best_classes

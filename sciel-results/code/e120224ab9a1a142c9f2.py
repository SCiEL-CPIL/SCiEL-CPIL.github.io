"""Rewritten program with stronger search for odd covering systems."""
import time
import random
import numpy as np

BASE = 51975
DEADLINE = 145.0  # safety margin before 150s kill

def _get_divisors(n):
    """Get all divisors of n greater than 1, sorted."""
    divs = []
    i = 3
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 2
    divs.append(n)
    divs.sort()
    return divs

def _score(covered):
    """Count uncovered residues."""
    return int(np.count_nonzero(covered == 0))

def _apply_class(covered, m, r):
    """Apply a class to the covered array (marks residues)."""
    idx = np.arange(r, BASE, m)
    covered[idx] = 1

def _remove_class(covered, m, r, classes, idx_in_list):
    """Remove a class and rebuild coverage from remaining classes."""
    # Rebuild from scratch excluding this class
    new_covered = np.zeros(BASE, dtype=np.uint8)
    for i, (mm, rr) in enumerate(classes):
        if i != idx_in_list:
            idx = np.arange(rr, BASE, mm)
            new_covered[idx] = 1
    covered[:] = new_covered

def _greedy_build(moduli_order, covered=None):
    """Greedy build given a specific order of moduli."""
    if covered is None:
        covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = np.arange(r, BASE, m)
            count = np.count_nonzero(covered[idx] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        idx = np.arange(best_r, BASE, m)
        covered[idx] = 1
    return classes, covered

def _local_search(classes, covered, time_limit):
    """Local search: try swapping residues or removing/adding moduli."""
    start_time = time.time()
    best_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_covered = covered.copy()
    
    # Try all pairs of residue swaps
    n = len(classes)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            m_i, r_i = classes[i]
            for j in range(n):
                if i == j:
                    continue
                if (time.time() - start_time) > time_limit:
                    break
                m_j, r_j = classes[j]
                # Try changing r_i to a different value
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    # Temporarily change
                    old_r_i = r_i
                    classes[i][1] = r_new
                    
                    # Recompute coverage
                    temp_covered = np.zeros(BASE, dtype=np.uint8)
                    for k, (mm, rr) in enumerate(classes):
                        idx = np.arange(rr, BASE, mm)
                        temp_covered[idx] = 1
                    
                    s = _score(temp_covered)
                    if s < best_score:
                        best_score = s
                        best_classes = [c[:] for c in classes]
                        best_covered = temp_covered.copy()
                        improved = True
                    else:
                        classes[i][1] = old_r_i
    
    return best_classes, best_covered, best_score

def _simulated_annealing(classes, covered, time_limit, rng):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    current_classes = [c[:] for c in classes]
    current_covered = covered.copy()
    current_score = _score(current_covered)
    
    best_classes = [c[:] for c in current_classes]
    best_covered = current_covered.copy()
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    while T > T_min and (time.time() - start_time) < time_limit:
        n = len(current_classes)
        move_type = rng.random()
        
        if move_type < 0.7:
            # Change residue of a random class
            i = rng.randint(0, n - 1)
            m_i = current_classes[i][0]
            old_r = current_classes[i][1]
            new_r = rng.randint(0, m_i - 1)
            current_classes[i][1] = new_r
        elif move_type < 0.85:
            # Swap residues of two classes
            i, j = rng.sample(range(n), 2)
            current_classes[i][1], current_classes[j][1] = current_classes[j][1], current_classes[i][1]
        else:
            # Remove a class
            i = rng.randint(0, n - 1)
            removed = current_classes.pop(i)
        
        # Compute new score
        temp_covered = np.zeros(BASE, dtype=np.uint8)
        for mm, rr in current_classes:
            idx = np.arange(rr, BASE, mm)
            temp_covered[idx] = 1
        new_score = _score(temp_covered)
        
        delta = new_score - current_score
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            current_covered = temp_covered
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in current_classes]
                best_covered = current_covered.copy()
        else:
            # Reject: undo
            if move_type < 0.7:
                current_classes[i][1] = old_r
            elif move_type < 0.85:
                current_classes[i][1], current_classes[j][1] = current_classes[j][1], current_classes[i][1]
            else:
                current_classes.insert(i, removed)
        
        T *= cooling
    
    return best_classes, best_covered, best_score

def construct():
    start_time = time.time()
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    divisors = _get_divisors(BASE)
    n_divs = len(divisors)
    
    # Multiple greedy initializations
    best_classes = None
    best_covered = None
    best_score = BASE  # worst case
    
    # Ascending greedy
    classes, covered = _greedy_build(divisors)
    s = _score(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
        best_covered = covered.copy()
    
    # Descending greedy
    classes, covered = _greedy_build(divisors[::-1])
    s = _score(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
        best_covered = covered.copy()
    
    # Random orderings
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        n_random = min(20, int(remaining / 2))
        for _ in range(n_random):
            if time.time() - start_time > DEADLINE - 10:
                break
            order = divisors[:]
            rng.shuffle(order)
            classes, covered = _greedy_build(order)
            s = _score(covered)
            if s < best_score:
                best_score = s
                best_classes = [c[:] for c in classes]
                best_covered = covered.copy()
    
    # Local search on best solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_classes is not None:
        ls_time = min(remaining * 0.4, 40.0)
        best_classes, best_covered, best_score = _local_search(
            best_classes, best_covered, ls_time
        )
    
    # Simulated annealing
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_classes is not None:
        sa_time = min(remaining * 0.5, 45.0)
        sa_classes, sa_covered, sa_score = _simulated_annealing(
            best_classes, best_covered, sa_time, np_rng
        )
        if sa_score < best_score:
            best_score = sa_score
            best_classes = sa_classes
            best_covered = sa_covered
    
    # Final local search polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_classes is not None:
        ls_time = min(remaining * 0.8, 20.0)
        best_classes, best_covered, best_score = _local_search(
            best_classes, best_covered, ls_time
        )
    
    # Safety check
    if time.time() - start_time > DEADLINE:
        return best_classes if best_classes is not None else [[3, 0]]
    
    return best_classes if best_classes is not None else [[3, 0]]

# EVOLVE-BLOCK-END

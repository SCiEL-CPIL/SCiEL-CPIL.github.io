"""Optimized program: local search over residue assignments for odd covering systems."""
import time
import random
import math
from collections import defaultdict

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# All odd divisors > 1
_DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
_DIVISORS_SORTED_DESC = sorted(_DIVISORS, reverse=True)  # larger moduli first
_DIVISORS_SORTED_ASC = sorted(_DIVISORS)  # smaller moduli first


def _uncovered_count(classes):
    """Count residues mod BASE covered by no class. classes = list of (m, r)."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _build_initial_greedy_desc():
    """Greedy construction: for each modulus (largest first), pick residue that maximizes new coverage."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_DESC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    return classes


def _build_initial_greedy_asc():
    """Greedy construction: for each modulus (smallest first), pick residue that maximizes new coverage."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_ASC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    return classes


def _build_initial_greedy_random(order):
    """Greedy construction with random ordering."""
    covered = bytearray(BASE)
    classes = []
    mods = list(order)
    random.shuffle(mods)
    for m in mods:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    return classes


def _fast_uncovered(classes):
    """Fast uncovered count using bytearray."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _coverage_mask(classes):
    """Return a bytearray of length BASE where 1 means covered."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return covered


def _score_from_mask(covered):
    """Compute uncovered count from a coverage mask."""
    return BASE - sum(covered)


def _local_search(classes, deadline, max_iterations=200):
    """Local search: try changing residues, adding, removing classes.
    Uses coverage mask for efficient incremental updates."""
    curr_classes = [list(c) for c in classes]
    curr_mask = _coverage_mask(curr_classes)
    curr_score = _score_from_mask(curr_mask)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iterations and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # 1. Try changing residue of existing classes
        for i in range(len(curr_classes)):
            m, r = curr_classes[i]
            best_new_r = r
            best_new_score = curr_score
            
            # Try all residues for small moduli, sample for large
            if m <= 50:
                candidates = list(range(m))
            else:
                candidates = set()
                candidates.add((r - 1) % m)
                candidates.add((r + 1) % m)
                candidates.add(r)
                for _ in range(min(20, m)):
                    candidates.add(random.randint(0, m - 1))
                candidates = list(candidates)
            
            for new_r in candidates:
                if new_r == r:
                    continue
                # Update mask: remove old coverage, add new coverage
                old_mask = curr_mask[:]
                # Remove old residues
                for x in range(r, BASE, m):
                    # Only set to 0 if no other class covers x
                    # This is complex, so we'll rebuild for correctness
                    pass
                
                # Simpler: just rebuild mask for this change
                test_classes = [list(c) for c in curr_classes]
                test_classes[i][1] = new_r
                test_mask = _coverage_mask(test_classes)
                test_score = _score_from_mask(test_mask)
                
                if test_score < best_new_score:
                    best_new_score = test_score
                    best_new_r = new_r
            
            if best_new_r != r and best_new_score < curr_score:
                curr_classes[i][1] = best_new_r
                curr_mask = _coverage_mask(curr_classes)
                curr_score = best_new_score
                improved = True
        
        # 2. Try adding a new class
        used_mods = set(m for m, _ in curr_classes)
        available = [m for m in _DIVISORS if m not in used_mods]
        
        if available:
            # Try all available moduli, but limit to a subset if too many
            if len(available) > 10:
                samples = random.sample(available, 10)
            else:
                samples = available
                
            for m in samples:
                best_r = -1
                best_gain = 0
                for r in range(m):
                    # Count how many new residues would be covered
                    gain = 0
                    for x in range(r, BASE, m):
                        if not curr_mask[x]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_r >= 0 and best_gain > 0:
                    test_classes = curr_classes + [[m, best_r]]
                    test_mask = _coverage_mask(test_classes)
                    test_score = _score_from_mask(test_mask)
                    if test_score < curr_score:
                        curr_classes.append([m, best_r])
                        curr_mask = test_mask
                        curr_score = test_score
                        improved = True
        
        # 3. Try removing a class
        for i in range(len(curr_classes)):
            test_classes = curr_classes[:i] + curr_classes[i+1:]
            test_mask = _coverage_mask(test_classes)
            test_score = _score_from_mask(test_mask)
            if test_score < curr_score:
                del curr_classes[i]
                curr_mask = test_mask
                curr_score = test_score
                improved = True
                break
    
    return curr_classes, curr_score


def _simulated_annealing(classes, deadline, max_iterations=500):
    """Simulated annealing to escape local minima."""
    curr_classes = [list(c) for c in classes]
    curr_mask = _coverage_mask(curr_classes)
    curr_score = _score_from_mask(curr_mask)
    best_classes = [list(c) for c in curr_classes]
    best_score = curr_score
    
    temp = 10.0
    cooling = 0.95
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        # Choose a random move
        move_type = random.choice(['change', 'add', 'remove'])
        
        if move_type == 'change' and curr_classes:
            i = random.randint(0, len(curr_classes) - 1)
            m, r = curr_classes[i]
            new_r = random.randint(0, m - 1)
            if new_r == r:
                continue
            test_classes = [list(c) for c in curr_classes]
            test_classes[i][1] = new_r
            test_mask = _coverage_mask(test_classes)
            test_score = _score_from_mask(test_mask)
            delta = test_score - curr_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                curr_classes = test_classes
                curr_mask = test_mask
                curr_score = test_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [list(c) for c in curr_classes]
        
        elif move_type == 'add':
            used_mods = set(m for m, _ in curr_classes)
            available = [m for m in _DIVISORS if m not in used_mods]
            if available:
                m = random.choice(available)
                r = random.randint(0, m - 1)
                test_classes = curr_classes + [[m, r]]
                test_mask = _coverage_mask(test_classes)
                test_score = _score_from_mask(test_mask)
                delta = test_score - curr_score
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    curr_classes = test_classes
                    curr_mask = test_mask
                    curr_score = test_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [list(c) for c in curr_classes]
        
        elif move_type == 'remove' and len(curr_classes) > 1:
            i = random.randint(0, len(curr_classes) - 1)
            test_classes = curr_classes[:i] + curr_classes[i+1:]
            test_mask = _coverage_mask(test_classes)
            test_score = _score_from_mask(test_mask)
            delta = test_score - curr_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                curr_classes = test_classes
                curr_mask = test_mask
                curr_score = test_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [list(c) for c in curr_classes]
        
        temp *= cooling
        if best_score == 0:
            break
    
    return best_classes, best_score


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 150s limit
    
    # Build multiple initial solutions
    inits = []
    inits.append(('desc', _build_initial_greedy_desc()))
    inits.append(('asc', _build_initial_greedy_asc()))
    
    # Add a few random-order greedy inits
    for _ in range(3):
        inits.append(('rand', _build_initial_greedy_random(_DIVISORS)))
    
    # Evaluate all inits
    best_classes = None
    best_score = float('inf')
    for name, classes in inits:
        score = _fast_uncovered(classes)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Run local search on the best initial solution
    if time.time() < DEADLINE - 20.0:
        ls_classes, ls_score = _local_search(best_classes, DEADLINE - 20.0, max_iterations=100)
        if ls_score < best_score:
            best_classes = ls_classes
            best_score = ls_score
    
    # Run simulated annealing
    if time.time() < DEADLINE - 10.0:
        sa_classes, sa_score = _simulated_annealing(best_classes, DEADLINE - 10.0, max_iterations=300)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
    
    # Try a few more restarts if time permits
    restart_count = 0
    while time.time() < DEADLINE - 5.0 and restart_count < 3:
        restart_count += 1
        # Randomize the initial solution
        random_classes = []
        random_mods = random.sample(_DIVISORS, min(len(_DIVISORS), 25))
        for m in random_mods:
            r = random.randint(0, m - 1)
            random_classes.append((m, r))
        
        rand_score = _fast_uncovered(random_classes)
        if rand_score < best_score:
            best_classes = random_classes
            best_score = rand_score
        
        if time.time() < DEADLINE - 5.0:
            improved_classes, improved_score = _local_search(best_classes, DEADLINE - 5.0, max_iterations=50)
            if improved_score < best_score:
                best_classes = improved_classes
                best_score = improved_score
        
        if best_score == 0:
            break
    
    # Final polish: one more local search pass
    if time.time() < DEADLINE - 2.0 and best_score > 0:
        final_classes, final_score = _local_search(best_classes, DEADLINE - 2.0, max_iterations=30)
        if final_score < best_score:
            best_classes = final_classes
            best_score = final_score
    
    # Convert to list of lists for return
    return [[m, r] for m, r in best_classes]

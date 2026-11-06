"""Rewritten solver for odd covering systems with moduli dividing 51975."""
import time
import random
import math

BASE = 51975  # 3^3 * 5^2 * 7 * 11

# Precompute all divisors of BASE that are > 1
def _get_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

ALL_DIVS = [d for d in _get_divisors(BASE) if d > 1]

# For fast scoring, precompute for each divisor m the list of residues it covers
# We'll use a bytearray of length BASE for coverage tracking

def _count_uncovered(covered):
    return sum(1 for x in covered if x == 0)

def _score(classes):
    covered = bytearray(BASE)
    for m, r in classes:
        for i in range(r, BASE, m):
            covered[i] = 1
    return _count_uncovered(covered)

def _greedy_build(order):
    """Greedy: for moduli in given order, pick residue covering most uncovered."""
    covered = bytearray(BASE)
    classes = []
    for m in order:
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered residues in this class
            cnt = 0
            for i in range(r, BASE, m):
                if covered[i] == 0:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        for i in range(best_r, BASE, m):
            covered[i] = 1
    return classes, _count_uncovered(covered)

def _local_search(classes, time_limit, rng):
    """Local search: try swapping residues or swapping moduli."""
    classes = [c[:] for c in classes]
    current_score = _score(classes)
    best_classes = [c[:] for c in classes]
    best_score = current_score
    start = time.time()
    
    # Build a mapping: for each modulus in use, its index in classes
    used_moduli = [c[0] for c in classes]
    
    # Precompute coverage for current state
    covered = bytearray(BASE)
    for m, r in classes:
        for i in range(r, BASE, m):
            covered[i] = 1
    
    while time.time() - start < time_limit:
        improved = False
        # Try changing residue for each class
        for idx in range(len(classes)):
            m, r = classes[idx]
            # Try all residues for this modulus
            # First, remove current class from coverage
            for i in range(r, BASE, m):
                covered[i] = 0
            
            best_r = r
            best_new_score = current_score
            for nr in range(m):
                # Add new residue
                gain = 0
                for i in range(nr, BASE, m):
                    if covered[i] == 0:
                        gain += 1
                new_score = current_score - gain
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_r = nr
            
            # Restore with best residue
            for i in range(best_r, BASE, m):
                covered[i] = 1
            
            if best_r != r:
                classes[idx][1] = best_r
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                improved = True
        
        if not improved:
            # Try swapping two moduli: replace one modulus with an unused one
            unused_moduli = [m for m in ALL_DIVS if m not in used_moduli]
            if unused_moduli:
                # Try replacing a random class with an unused modulus
                idx = rng.randint(0, len(classes) - 1)
                old_m, old_r = classes[idx]
                new_m = rng.choice(unused_moduli)
                
                # Remove old class
                for i in range(old_r, BASE, old_m):
                    covered[i] = 0
                # Find best residue for new modulus
                best_r = 0
                best_gain = -1
                for nr in range(new_m):
                    gain = 0
                    for i in range(nr, BASE, new_m):
                        if covered[i] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = nr
                # Add new class
                for i in range(best_r, BASE, new_m):
                    covered[i] = 1
                new_score = current_score - best_gain
                
                if new_score < best_score:
                    classes[idx] = [new_m, best_r]
                    used_moduli[idx] = new_m
                    current_score = new_score
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    improved = True
                else:
                    # Revert
                    for i in range(best_r, BASE, new_m):
                        covered[i] = 0
                    for i in range(old_r, BASE, old_m):
                        covered[i] = 1
                    classes[idx] = [old_m, old_r]
                    used_moduli[idx] = old_m
            
            if not improved:
                # Stuck in local optimum, try a random perturbation
                idx = rng.randint(0, len(classes) - 1)
                m = classes[idx][0]
                new_r = rng.randint(0, m - 1)
                # Remove old
                for i in range(classes[idx][1], BASE, m):
                    covered[i] = 0
                # Add new
                for i in range(new_r, BASE, m):
                    covered[i] = 1
                classes[idx][1] = new_r
                current_score = _score(classes)
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    rng = random.Random(42)
    
    # Try multiple greedy orders
    best_classes = None
    best_score = BASE  # Worst case: nothing covered
    
    # Order 1: ascending
    asc_order = sorted(ALL_DIVS)
    classes, score = _greedy_build(asc_order)
    if score < best_score:
        best_score = score
        best_classes = classes
    
    # Order 2: descending
    desc_order = sorted(ALL_DIVS, reverse=True)
    classes, score = _greedy_build(desc_order)
    if score < best_score:
        best_score = score
        best_classes = classes
    
    # Order 3: by size (number of residues covered) descending
    # Smaller moduli cover more residues
    size_order = sorted(ALL_DIVS, key=lambda m: BASE // m, reverse=True)
    classes, score = _greedy_build(size_order)
    if score < best_score:
        best_score = score
        best_classes = classes
    
    # Order 4: random
    for _ in range(3):
        rand_order = ALL_DIVS[:]
        rng.shuffle(rand_order)
        classes, score = _greedy_build(rand_order)
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Local search from best
    remaining = deadline - time.time()
    if remaining > 5 and best_classes is not None:
        best_classes, best_score = _local_search(best_classes, remaining - 5, rng)
    
    return best_classes

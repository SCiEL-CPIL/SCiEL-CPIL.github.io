import time
import numpy as np

_BASE = 51975
_DEADLINE = time.time() + 130  # safety margin before 150s kill

def _divisors_of(n):
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(d for d in divs if d > 1)

def _covered_mask(classes, base):
    """Return a boolean array of length base indicating covered residues."""
    mask = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        mask[r::m] = 1
    return mask

def _uncovered_count(classes, base):
    mask = _covered_mask(classes, base)
    return base - np.sum(mask)

def _greedy_init(moduli, base, order=None):
    """Greedy: for each modulus in given order, pick residue covering most uncovered."""
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    if order is None:
        order = list(range(len(moduli)))  # default: increasing modulus
    for idx in order:
        m = moduli[idx]
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered residues in this class
            cnt = np.sum(covered[r::m] == 0)
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, moduli, base, max_passes=50):
    """Coordinate descent: for each modulus, try all residues, keep best."""
    classes = [list(c) for c in classes]
    covered = _covered_mask(classes, base)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        passes += 1
        for i, (m, r) in enumerate(classes):
            # Remove current class from covered
            # We need to recount without this class
            # Efficient: for each candidate r', compute new uncovered count
            # But that's expensive. Instead, let's do:
            # For each r', the new covered set = (covered without class i) ∪ {r' + km}
            # uncovered_new = uncovered_without_i - (newly covered by r')
            # This is complex. Simpler: just try each r' and compute the full mask.
            # But that's O(m * base) per modulus, which is too slow.
            
            # Better approach: for each r', compute how many NEW residues it covers
            # given the other classes.
            # First, compute covered_without_i
            covered_wo_i = _covered_mask(classes[:i] + classes[i+1:], base)
            # Now for each r', new covered = covered_wo_i with r'::m set to 1
            # New uncovered = base - sum(new_covered)
            # = base - sum(covered_wo_i) - (number of residues in r'::m that were 0 in covered_wo_i)
            base_uncov_wo_i = base - np.sum(covered_wo_i)
            best_r = r
            best_uncov = base_uncov_wo_i
            for r_prime in range(m):
                newly_covered = np.sum(covered_wo_i[r_prime::m] == 0)
                uncov = base_uncov_wo_i - newly_covered
                if uncov < best_uncov:
                    best_uncov = uncov
                    best_r = r_prime
            if best_r != r:
                classes[i][1] = best_r
                improved = True
        # Update covered
        covered = _covered_mask(classes, base)
    return classes

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130
    
    moduli = _divisors_of(_BASE)
    base = _BASE
    n_mod = len(moduli)
    
    # Greedy initialization
    best_classes = _greedy_init(moduli, base)
    best_score = _uncovered_count(best_classes, base)
    
    # Local search
    if time.time() < _DEADLINE - 10:
        ls_classes = _local_search(best_classes, moduli, base)
        ls_score = _uncovered_count(ls_classes, base)
        if ls_score < best_score:
            best_classes = ls_classes
            best_score = ls_score
    
    # Multi-start with different orders
    import random
    rng = random.Random(42)
    attempt = 0
    while time.time() < _DEADLINE - 10 and attempt < 20:
        attempt += 1
        # Random permutation of moduli for greedy
        order = list(range(n_mod))
        rng.shuffle(order)
        # Or: reverse order (largest first)
        if attempt % 2 == 0:
            order = list(range(n_mod - 1, -1, -1))
        
        classes = _greedy_init(moduli, base, order=order)
        score = _uncovered_count(classes, base)
        
        if score < best_score:
            best_classes = classes
            best_score = score
        
        # Local search on this
        if time.time() < _DEADLINE - 10:
            ls_classes = _local_search(classes, moduli, base, max_passes=20)
            ls_score = _uncovered_count(ls_classes, base)
            if ls_score < best_score:
                best_classes = ls_classes
                best_score = ls_score
    
    return best_classes

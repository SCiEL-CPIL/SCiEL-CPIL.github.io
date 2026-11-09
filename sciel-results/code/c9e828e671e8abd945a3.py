"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _score_classes(classes, base=51975):
    """Count uncovered residues."""
    covered = bytearray(base)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, base, m))
    return base - sum(covered)

def _greedy_construct(base=51975):
    """Greedy: moduli in increasing order, each takes the class covering most uncovered residues."""
    covered = bytearray(base)
    classes = []
    for m in [d for d in range(3, base + 1, 2) if base % d == 0]:
        best = max(range(m), key=lambda r: covered[r::m].count(0))
        classes.append([m, best])
        covered[best::m] = b'\x01' * len(range(best, base, m))
    return classes

def _local_search(classes, base=51975, time_limit=10):
    """Local search: try changing residues or swapping moduli."""
    start_time = time.time()
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    div_set = set(divisors)
    best_score = _score_classes(classes, base)
    best_classes = [c[:] for c in classes]
    
    # Precompute coverage for each class
    def get_coverage(m, r):
        return np.array([1 if (x - r) % m == 0 else 0 for x in range(base)])
    
    # Try changing each residue
    for i, (m, r) in enumerate(best_classes):
        if time.time() - start_time > time_limit:
            break
        # Try all residues for this modulus
        for new_r in range(m):
            if new_r == r:
                continue
            new_classes = [c[:] for c in best_classes]
            new_classes[i] = [m, new_r]
            s = _score_classes(new_classes, base)
            if s < best_score:
                best_score = s
                best_classes = new_classes
                r = new_r  # update current r for subsequent checks
    
    # Try adding a modulus not yet used
    used_moduli = set(m for m, r in best_classes)
    for m in divisors:
        if time.time() - start_time > time_limit:
            break
        if m in used_moduli:
            continue
        best_r = max(range(m), key=lambda r: _score_classes(best_classes + [[m, r]], base))
        s = _score_classes(best_classes + [[m, best_r]], base)
        if s < best_score:
            best_score = s
            best_classes.append([m, best_r])
            used_moduli.add(m)
    
    return best_classes

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with greedy
    classes = _greedy_construct(base)
    best_score = _score_classes(classes, base)
    
    # Try local search with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        classes = _local_search(classes, base, time_limit=min(remaining - 5, 30))
        new_score = _score_classes(classes, base)
        if new_score < best_score:
            best_score = new_score
    
    # If still time, try more local search passes
    remaining = deadline - time.time()
    if remaining > 5:
        classes = _local_search(classes, base, time_limit=min(remaining - 5, 30))
        new_score = _score_classes(classes, base)
        if new_score < best_score:
            best_score = new_score
    
    return classes
# EVOLVE-BLOCK-END

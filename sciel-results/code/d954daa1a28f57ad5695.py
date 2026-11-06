import time
import random
import numpy as np

_BASE = 51975
_DIVISORS = [d for d in range(3, _BASE + 1, 2) if _BASE % d == 0]
_DIVISORS_SET = frozenset(_DIVISORS)


def _count_uncovered(covered, m, r):
    """Count uncovered residues in class (m, r)."""
    return int(np.sum(1 - covered[r::_BASE // m * m + r - r]))  # placeholder, will fix


def _best_residue(covered, m, base):
    """Find residue r for modulus m that covers most uncovered residues.
    Returns (best_r, best_count)."""
    # For modulus m, residues r=0..m-1
    # covered[r::m] gives positions r, r+m, r+2m, ... < base
    # We want to count zeros in each such slice
    # Vectorize: for all r, count uncovered
    # covered is a numpy array of 0s and 1s
    # For a given m, we can compute all residue counts at once
    # uncovered = 1 - covered
    # For each r in 0..m-1: sum of uncovered[r::m]
    # This is equivalent to: for each position i, it contributes to residue i % m
    # So: counts[r] = sum of uncovered[i] for i where i % m == r
    # = sum of uncovered[0:base:m] at positions where index % m == r
    
    # Efficient approach: 
    # uncovered reshaped? No, strides differ.
    # Just loop over r but use numpy slice sum (fast for each)
    uncovered = 1 - covered
    best_r = 0
    best_cnt = -1
    for r in range(m):
        cnt = int(np.sum(uncovered[r::m]))
        if cnt > best_cnt:
            best_cnt = cnt
            best_r = r
    return best_r, best_cnt


def _build_covered(classes, base):
    """Build covered array from list of [m, r] classes."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered


def _score(covered, base):
    return base - int(np.sum(covered))


def _greedy_add_best(covered, used, classes, base, deadline, best, order=None):
    """Repeatedly add the best (m, r) pair that covers most uncovered."""
    if order is None:
        order = list(_DIVISORS)
    
    while time.time() < deadline:
        remaining = [m for m in order if m not in used]
        if not remaining:
            break
        
        best_m, best_r, best_cnt = None, None, -1
        for m in remaining:
            r, cnt = _best_residue(covered, m, base)
            if cnt > best_cnt:
                best_cnt = cnt
                best_m = m
                best_r = r
        
        if best_m is None or best_cnt <= 0:
            break
        
        classes.append([best_m, best_r])
        used.add(best_m)
        covered[best_r::best_m] = 1
        
        s = int(np.sum(covered))
        if s > best[0]:
            best[0] = s
            best[1] = list(classes)
        
        if time.time() > deadline:
            break


def _local_search(covered, used, classes, base, deadline, best):
    """Try to improve by swapping or re-optimizing residues."""
    # Try re-optimizing each existing class's residue
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m_old, r_old = classes[i]
            # Try all residues for this modulus
            best_r_new, best_cnt_new = _best_residue(covered, m_old, base)
            if best_r_new != r_old and best_cnt_new > 0:
                # Check if changing improves overall
                # Remove old, add new, compare
                old_classes = list(classes)
                classes[i] = [m_old, best_r_new]
                new_covered = _build_covered(classes, base)
                old_score = _score(covered, base)
                new_score = _score(new_covered, base)
                if new_score < old_score:
                    covered = new_covered
                    improved = True
                    s = int(np.sum(covered))
                    if s > best[0]:
                        best[0] = s
                        best[1] = list(classes)
                else:
                    classes[i] = [m_old, r_old]
            # Also try removing this class if it doesn't help
            if len(classes) > 1:
                trial_classes = [c for j, c in enumerate(classes) if j != i]
                trial_covered = _build_covered(trial_classes, base)
                trial_score = _score(trial_covered, base)
                cur_score = _score(covered, base)
                if trial_score <= cur_score:
                    classes = trial_classes
                    covered = trial_covered
                    used.discard(m_old)
                    improved = True
                    s = int(np.sum(covered))
                    if s > best[0]:
                        best[0] = s
                        best[1] = list(classes)


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s (evaluator kills at 150s)
    rng = random.Random(1234567)
    base = _BASE
    best = [0, []]  # [covered_count, classes]

    # Initial greedy: process divisors in various orders
    orders = [
        sorted(_DIVISORS),
        sorted(_DIVISORS, reverse=True),
        sorted(_DIVISORS, key=lambda m: base // m),
        sorted(_DIVISORS, key=lambda m: -base // m),
        list(_DIVISORS),
    ]
    for o in orders:
        rng.shuffle(o)

    iters = 0
    while time.time() < deadline - 10.0:
        iters += 1
        order = orders[iters % len(orders)]
        
        if iters % 3 == 0:
            # Fresh start with random order
            fresh_order = list(_DIVISORS)
            rng.shuffle(fresh_order)
            covered = np.zeros(base, dtype=np.uint8)
            used = set()
            classes = []
            _greedy_add_best(covered, used, classes, base, min(deadline - 5.0, start + 115.0), best, fresh_order)
        else:
            # Perturb best: drop some classes and re-greedy
            covered = _build_covered(best[1], base)
            used = set(m for m, r in best[1])
            classes = list(best[1])
            
            # Randomly drop 1-3 classes
            if classes and rng.random() < 0.6:
                drop = rng.randrange(1, max(2, len(classes) // 3 + 1))
                for _ in range(drop):
                    if not classes:
                        break
                    idx = rng.randrange(len(classes))
                    m = classes[idx][0]
                    classes.pop(idx)
                    used.discard(m)
                covered = _build_covered(classes, base)
            
            _greedy_add_best(covered, used, classes, base, min(deadline - 5.0, start + 115.0), best, order)
        
        # Local search on best
        if time.time() < deadline - 5.0 and best[1]:
            covered_ls = _build_covered(best[1], base)
            used_ls = set(m for m, r in best[1])
            classes_ls = list(best[1])
            _local_search(covered_ls, used_ls, classes_ls, base, min(deadline - 3.0, start + 115.0), best)

    return best[1] if best[1] else [[3, 0]]


if __name__ == "__main__":
    res = construct()
    print(len(res))

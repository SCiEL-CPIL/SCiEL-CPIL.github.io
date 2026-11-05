"""Optimized odd covering system for moduli dividing 51975."""
import time
import random
from collections import defaultdict

# 51975 = 3^3 * 5^2 * 7 * 11
BASE = 51975

# Precompute all odd divisors of 51975 greater than 1
def _get_divisors():
    n = BASE
    divs = set()
    # factor: 3^3 * 5^2 * 7 * 11
    for a in range(4):      # 3^a, a=0..3
        for b in range(3):  # 5^b, b=0..2
            for c in range(2):  # 7^c, c=0..1
                for d in range(2):  # 11^d, d=0..1
                    m = (3**a) * (5**b) * (7**c) * (11**d)
                    if m > 1:
                        divs.add(m)
    return sorted(divs)

ALL_MODULI = _get_divisors()

def _count_uncovered(covered, m, r):
    """Count how many uncovered residues are in class (m, r)."""
    # Residues are r, r+m, r+2m, ... < BASE
    count = 0
    for x in range(r, BASE, m):
        if not covered[x]:
            count += 1
    return count

def _apply_class(covered, m, r):
    """Mark residues in class (m, r) as covered."""
    for x in range(r, BASE, m):
        covered[x] = 1

def _score(covered):
    """Number of uncovered residues."""
    return sum(1 for x in range(BASE) if not covered[x])

def _greedy_build(moduli_order, random_residues=False, rng=None):
    """Greedy: for each modulus in order, pick residue covering most uncovered."""
    covered = bytearray(BASE)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_count = -1
        if random_residues and rng:
            candidates = list(range(m))
            rng.shuffle(candidates)
        else:
            candidates = range(m)
        for r in candidates:
            c = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    c += 1
            if c > best_count:
                best_count = c
                best_r = r
        classes.append([m, best_r])
        _apply_class(covered, m, best_r)
    return classes, covered

def _local_search(classes, covered, max_iters=50, rng=None):
    """Try to improve by swapping residues or adding/removing classes."""
    if rng is None:
        rng = random.Random(42)
    
    current_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_covered = covered[:]
    
    for _ in range(max_iters):
        improved = False
        # Try changing residue of an existing class
        for i in range(len(best_classes)):
            m, r = best_classes[i]
            # Try a few random residues
            for _trial in range(5):
                new_r = rng.randrange(m)
                if new_r == r:
                    continue
                # Compute new score if we change (m, r) to (m, new_r)
                # First unmark old, mark new, count
                old_count = 0
                for x in range(r, BASE, m):
                    if best_covered[x]:
                        old_count += 1
                # Count how many of the new class are already covered by others
                new_uncovered = 0
                for x in range(new_r, BASE, m):
                    if not best_covered[x]:
                        new_uncovered += 1
                # The change in score:
                # We lose coverage of (m, r) residues that aren't covered by others
                # We gain coverage of (m, new_r) residues that aren't covered by others
                # This is complex; let's just simulate
                temp = best_covered[:]
                # Unmark old
                for x in range(r, BASE, m):
                    # Check if covered by other classes
                    still_covered = False
                    for j in range(len(best_classes)):
                        if j == i:
                            continue
                        mj, rj = best_classes[j]
                        if x % mj == rj:
                            still_covered = True
                            break
                    if not still_covered:
                        temp[x] = 0
                # Mark new
                for x in range(new_r, BASE, m):
                    temp[x] = 1
                new_score = _score(temp)
                if new_score < current_score:
                    best_classes[i][1] = new_r
                    best_covered = temp
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_classes, best_covered, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_classes = None
    best_score = BASE  # worst case
    
    # Strategy 1: Greedy with moduli in increasing order
    classes, covered = _greedy_build(ALL_MODULI)
    s = _score(covered)
    if s < best_score:
        best_score = s
        best_classes = [c[:] for c in classes]
    
    # Strategy 2: Greedy with moduli in decreasing order
    if time.time() < deadline:
        classes, covered = _greedy_build(sorted(ALL_MODULI, reverse=True))
        s = _score(covered)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
    
    # Strategy 3: Greedy with random orderings
    rng = random.Random(12345)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = ALL_MODULI[:]
        rng.shuffle(order)
        classes, covered = _greedy_build(order)
        s = _score(covered)
        if s < best_score:
            best_score = s
            best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    if time.time() < deadline and best_classes:
        best_classes, _, best_score = _local_search(best_classes, bytearray(BASE), max_iters=30, rng=rng)
        # Rebuild covered from best_classes for consistency
        covered = bytearray(BASE)
        for m, r in best_classes:
            _apply_class(covered, m, r)
    
    # Strategy 4: Try to remove redundant classes
    if time.time() < deadline and best_classes:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_classes) - 1, -1, -1):
                if time.time() > deadline:
                    break
                m, r = best_classes[i]
                # Check if removing this class increases uncovered count
                covered_without = bytearray(BASE)
                for j, (mj, rj) in enumerate(best_classes):
                    if j == i:
                        continue
                    _apply_class(covered_without, mj, rj)
                new_score = _score(covered_without)
                if new_score <= best_score:
                    # Removing is safe or better
                    if new_score < best_score:
                        best_score = new_score
                    best_classes.pop(i)
                    improved = True
                    break
    
    # Final: ensure all moduli have distinct values (they do by construction)
    # Verify
    if best_classes is None:
        best_classes = [[m, 0] for m in ALL_MODULI]
    
    return best_classes

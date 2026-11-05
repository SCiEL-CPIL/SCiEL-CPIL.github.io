"""Optimized odd covering system for moduli dividing 51975."""
import time
import random
import math
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

def _apply_class(covered, m, r):
    """Mark residues in class (m, r) as covered."""
    for x in range(r, BASE, m):
        covered[x] = 1

def _score(covered):
    """Number of uncovered residues."""
    return sum(1 for x in range(BASE) if not covered[x])

def _marginal_gain(covered, m, r):
    """Count how many uncovered residues are in class (m, r)."""
    count = 0
    for x in range(r, BASE, m):
        if not covered[x]:
            count += 1
    return count

def _adaptive_greedy_build(remaining_moduli, covered, rng, deadline, alpha=0.3, beta=0.7):
    """
    Adaptive look-ahead greedy: probabilistically select next modulus based on
    marginal gain relative to remaining search budget.
    
    For each candidate modulus, compute the best possible marginal gain
    (max over residues). Then use a softmax-like selection where probability
    is proportional to gain^alpha * (1/m)^beta, favoring high-gain but also
    smaller moduli (which cover more residues per class).
    """
    classes = []
    remaining = list(remaining_moduli)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Compute marginal gain for each remaining modulus
        gains = {}
        best_residues = {}
        for m in remaining:
            best_g = -1
            best_r = 0
            # For efficiency, sample residues if m is large
            if m <= 200:
                candidates = range(m)
            else:
                # Sample a subset of residues for large moduli
                candidates = [rng.randrange(m) for _ in range(min(m, 50))]
                # Always include 0
                candidates.append(0)
            
            for r in candidates:
                g = 0
                for x in range(r, BASE, m):
                    if not covered[x]:
                        g += 1
                        if g > best_g:
                            best_g = g
                            best_r = r
                            if g == BASE // m:  # Can't do better than full coverage
                                break
                # Continue checking other residues to find the true best
                if g <= best_g:
                    continue
                # Check remaining residues more carefully if needed
                for r2 in range(r+1, m):
                    g2 = 0
                    for x in range(r2, BASE, m):
                        if not covered[x]:
                            g2 += 1
                    if g2 > best_g:
                        best_g = g2
                        best_r = r2
                    if g2 == BASE // m:
                        break
            
            if best_g >= 0:
                gains[m] = best_g
                best_residues[m] = best_r
        
        if not gains:
            break
        
        # Adaptive selection: probability proportional to gain^alpha * (1/m)^beta
        # This balances coverage gain against modulus size
        total_weight = 0.0
        weights = {}
        for m, g in gains.items():
            if g <= 0:
                weights[m] = 0.0
            else:
                w = (float(g) ** alpha) * ((1.0 / m) ** beta)
                weights[m] = w
                total_weight += w
        
        if total_weight <= 0:
            # All gains are zero, pick smallest modulus
            m = min(remaining, key=lambda x: x)
        else:
            # Sample from weighted distribution
            cum = 0.0
            u = rng.random() * total_weight
            m = remaining[0]
            for mm in remaining:
                cum += weights.get(mm, 0.0)
                if u <= cum:
                    m = mm
                    break
        
        # Apply the best residue for this modulus
        r = best_residues[m]
        classes.append([m, r])
        _apply_class(covered, m, r)
        remaining.remove(m)
        
        # Early termination if fully covered
        if _score(covered) == 0:
            break
    
    return classes, covered

def _local_search(classes, covered, max_iters=30, rng=None, deadline=None):
    """Try to improve by swapping residues of existing classes."""
    if rng is None:
        rng = random.Random(42)
    if deadline is None:
        deadline = time.time() + 10
    
    current_score = _score(covered)
    best_classes = [c[:] for c in classes]
    best_covered = bytearray(covered)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try changing residue of a random existing class
        idx = rng.randrange(len(best_classes))
        m, r = best_classes[idx]
        
        for _trial in range(8):
            new_r = rng.randrange(m)
            if new_r == r:
                continue
            
            # Simulate: unmark old class, check if still covered by others, mark new
            temp = bytearray(best_covered)
            for x in range(r, BASE, m):
                # Check if covered by other classes
                still_covered = False
                for j in range(len(best_classes)):
                    if j == idx:
                        continue
                    mj, rj = best_classes[j]
                    if x % mj == rj:
                        still_covered = True
                        break
                if not still_covered:
                    temp[x] = 0
            
            for x in range(new_r, BASE, m):
                temp[x] = 1
            
            new_score = _score(temp)
            if new_score < current_score:
                best_classes[idx][1] = new_r
                best_covered = temp
                current_score = new_score
                improved = True
                break
        
        if not improved:
            # Try a few more random indices before giving up
            for _extra in range(3):
                idx2 = rng.randrange(len(best_classes))
                m2, r2 = best_classes[idx2]
                for _trial in range(5):
                    new_r2 = rng.randrange(m2)
                    if new_r2 == r2:
                        continue
                    temp = bytearray(best_covered)
                    for x in range(r2, BASE, m2):
                        still_covered = False
                        for j in range(len(best_classes)):
                            if j == idx2:
                                continue
                            mj, rj = best_classes[j]
                            if x % mj == rj:
                                still_covered = True
                                break
                        if not still_covered:
                            temp[x] = 0
                    for x in range(new_r2, BASE, m2):
                        temp[x] = 1
                    new_score = _score(temp)
                    if new_score < current_score:
                        best_classes[idx2][1] = new_r2
                        best_covered = temp
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_classes, best_covered, current_score

def _remove_redundant(classes, deadline):
    """Remove classes that don't contribute to coverage."""
    if not classes:
        return classes
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(classes) - 1, -1, -1):
            if time.time() > deadline:
                break
            m, r = classes[i]
            # Check if removing this class increases uncovered count
            covered_without = bytearray(BASE)
            for j, (mj, rj) in enumerate(classes):
                if j == i:
                    continue
                _apply_class(covered_without, mj, rj)
            new_score = _score(covered_without)
            # Compare with current score
            covered_current = bytearray(BASE)
            for mj, rj in classes:
                _apply_class(covered_current, mj, rj)
            current_score = _score(covered_current)
            if new_score <= current_score:
                classes.pop(i)
                improved = True
                break
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    best_classes = None
    best_score = BASE  # worst case
    
    rng = random.Random(12345)
    
    # Strategy 1: Adaptive greedy with multiple random seeds and parameter variations
    for trial in range(15):
        if time.time() > deadline:
            break
        
        if trial == 0:
            # Deterministic: increasing order
            order = ALL_MODULI[:]
            covered = bytearray(BASE)
            classes = []
            for m in order:
                best_r = 0
                best_g = -1
                for r in range(m):
                    g = 0
                    for x in range(r, BASE, m):
                        if not covered[x]:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_r = r
                classes.append([m, best_r])
                _apply_class(covered, m, best_r)
                if _score(covered) == 0:
                    break
            s = _score(covered)
            if s < best_score:
                best_score = s
                best_classes = [c[:] for c in classes]
        
        elif trial == 1:
            # Deterministic: decreasing order
            order = sorted(ALL_MODULI, reverse=True)
            covered = bytearray(BASE)
            classes = []
            for m in order:
                best_r = 0
                best_g = -1
                for r in range(m):
                    g = 0
                    for x in range(r, BASE, m):
                        if not covered[x]:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_r = r
                classes.append([m, best_r])
                _apply_class(covered, m, best_r)
                if _score(covered) == 0:
                    break
            s = _score(covered)
            if s < best_score:
                best_score = s
                best_classes = [c[:] for c in classes]
        
        else:
            # Adaptive greedy with varying parameters
            alpha = 0.2 + 0.6 * rng.random()  # 0.2 to 0.8
            beta = 0.3 + 0.8 * rng.random()   # 0.3 to 1.1
            covered = bytearray(BASE)
            classes, covered = _adaptive_greedy_build(ALL_MODULI, covered, rng, deadline, alpha, beta)
            s = _score(covered)
            if s < best_score:
                best_score = s
                best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    if time.time() < deadline and best_classes:
        covered = bytearray(BASE)
        for m, r in best_classes:
            _apply_class(covered, m, r)
        best_classes, _, best_score = _local_search(best_classes, covered, max_iters=20, rng=rng, deadline=deadline)
    
    # Remove redundant classes
    if time.time() < deadline and best_classes:
        best_classes = _remove_redundant(best_classes, deadline)
        # Recompute score after removal
        covered = bytearray(BASE)
        for m, r in best_classes:
            _apply_class(covered, m, r)
        s = _score(covered)
        best_score = s
    
    # Final verification
    if best_classes is None:
        best_classes = [[m, 0] for m in ALL_MODULI]
    
    return best_classes

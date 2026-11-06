"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

_BASE = 51975
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _divisors():
    ds = []
    for d in range(3, _BASE + 1, 2):
        if _BASE % d == 0:
            ds.append(d)
    return ds

def _score(covered):
    # number of uncovered residues
    return int(np.count_nonzero(covered == 0))

def construct():
    divisors = _divisors()
    # Use a numpy bit-array over residues 0..51974
    covered = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    remaining = set(divisors)

    def add_class(m, r):
        covered[r::m] = 1
        classes.append([m, r])
        remaining.discard(m)

    # Build a full greedy: repeatedly pick (m,r) maximizing new coverage.
    # Use numpy for speed.
    gain_cache = {}
    for m in divisors:
        best_r, best_gain = 0, -1
        for r in range(m):
            g = int(np.count_nonzero(covered[r::m] == 0))
            if g > best_gain:
                best_gain = g
                best_r = r
        gain_cache[m] = (best_gain, best_r)

    # Greedily add in order of descending gain while it helps.
    while remaining:
        # pick m with max gain
        best_m, best = None, None
        for m in list(remaining):
            g, r = gain_cache[m]
            if g <= 0:
                continue
            if best is None or g > best[0] or (g == best[0] and m < best[1]):
                best = (g, m, r)
        if best is None:
            break
        g, m, r = best
        add_class(m, r)
        if time.time() > _DEADLINE - 5:
            break
        # Recompute gains for remaining divisors
        for mm in list(remaining):
            best_r, best_gain = 0, -1
            for r in range(mm):
                gg = int(np.count_nonzero(covered[r::mm] == 0))
                if gg > best_gain:
                    best_gain = gg
                    best_r = r
            gain_cache[mm] = (best_gain, best_r)

    # Local search: try to swap classes to improve coverage
    if time.time() < _DEADLINE - 10:
        # Create a mapping from modulus to class
        m_to_class = {m: r for m, r in classes}
        
        # Try to find improvements by swapping residues or removing/adding classes
        improved = True
        while improved and time.time() < _DEADLINE - 10:
            improved = False
            current_score = _score(covered)
            
            # Try swapping residues of existing classes
            for i, (m1, r1) in enumerate(classes):
                if time.time() > _DEADLINE - 10:
                    break
                for j, (m2, r2) in enumerate(classes):
                    if i == j or time.time() > _DEADLINE - 10:
                        continue
                    # Try swapping residues r1 and r2
                    # First, remove both classes
                    covered[r1::m1] = 0
                    covered[r2::m2] = 0
                    # Add swapped classes
                    covered[r2::m1] = 1
                    covered[r1::m2] = 1
                    new_score = _score(covered)
                    if new_score < current_score:
                        # Accept the swap
                        m_to_class[m1] = r2
                        m_to_class[m2] = r1
                        classes[i][1] = r2
                        classes[j][1] = r1
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        covered[r2::m1] = 0
                        covered[r1::m2] = 0
                        covered[r1::m1] = 1
                        covered[r2::m2] = 1
            
            # Try removing a class and adding a new one
            if not improved:
                for i, (m1, r1) in enumerate(classes):
                    if time.time() > _DEADLINE - 10:
                        break
                    # Remove class i
                    covered[r1::m1] = 0
                    current_score = _score(covered)
                    
                    # Try adding a new class from remaining divisors
                    for m2 in list(remaining):
                        if time.time() > _DEADLINE - 10:
                            break
                        best_r, best_gain = 0, -1
                        for r in range(m2):
                            g = int(np.count_nonzero(covered[r::m2] == 0))
                            if g > best_gain:
                                best_gain = g
                                best_r = r
                        if best_gain > 0:
                            # Check if adding this class improves the score
                            covered[best_r::m2] = 1
                            new_score = _score(covered)
                            if new_score < current_score:
                                # Accept
                                classes.append([m2, best_r])
                                remaining.discard(m2)
                                current_score = new_score
                                improved = True
                            else:
                                # Revert
                                covered[best_r::m2] = 0
                    # Restore class i
                    covered[r1::m1] = 1

    return classes
# EVOLVE-BLOCK-END

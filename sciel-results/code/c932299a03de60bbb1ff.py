"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the lower bound score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):  # symmetry: a+b = b+a
            sumset.add(U[i] + U[j])
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    
    num_sum = len(sumset)
    num_diff = len(diffset)
    
    if num_sum == 0:
        return 0.0
    
    ratio = num_diff / num_sum
    if ratio <= 0:
        return 0.0
    
    exponent = 1.0 + np.log(ratio) / np.log(2 * max_u + 1)
    return exponent

def _construct_small():
    """Return a small known good set."""
    # [0, 1, 3] gives sumset {0,1,2,3,4,6}, diffset {-3,-2,-1,0,1,2,3}
    # |sum|=6, |diff|=7, max=3, score = 1 + log(7/6)/log(7)
    return [0, 1, 3]

def _try_known_constructions():
    """Try some known constructions and return the best one."""
    candidates = []
    
    # Small known set
    candidates.append([0, 1, 3])
    
    # Try some structured sets
    # Base-3 digits: {0,1,3,9,27,...}
    for k in range(1, 8):
        s = [0]
        power = 1
        for i in range(k):
            power *= 3
            s.append(power)
        candidates.append(sorted(s))
    
    # Try {0, 1, 3, 9, 27, 81} (5 elements)
    candidates.append([0, 1, 3, 9, 27, 81])
    
    # Try {0, 1, 2, 4, 8, 16, 32} (powers of 2)
    candidates.append([0, 1, 2, 4, 8, 16, 32])
    
    # Try {0, 1, 3, 6, 10, 15} (triangular numbers)
    candidates.append([0, 1, 3, 6, 10, 15])
    
    # Try {0, 1, 2, 5, 10, 20, 40}
    candidates.append([0, 1, 2, 5, 10, 20, 40])
    
    # Try {0, 1, 3, 7, 15, 31} (2^k - 1)
    candidates.append([0, 1, 3, 7, 15, 31])
    
    # Try {0, 1, 4, 16, 64} (powers of 4)
    candidates.append([0, 1, 4, 16, 64])
    
    # Try {0, 1, 3, 9, 27, 81, 243}
    candidates.append([0, 1, 3, 9, 27, 81, 243])
    
    # Try {0, 1, 2, 4, 8, 16, 32, 64}
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64])
    
    # Try {0, 1, 3, 7, 15, 31, 63}
    candidates.append([0, 1, 3, 7, 15, 31, 63])
    
    best = None
    best_score = 0.0
    for cand in candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best = cand
    
    return best if best is not None else [0, 1, 3]

def _local_search(start_set, max_steps=5000, time_limit=30):
    """Local search: try adding/removing/swapping elements."""
    start_set = sorted(set(start_set))
    best_set = start_set[:]
    best_score = _score_set(start_set)
    
    deadline = time.time() + time_limit
    rng = np.random.default_rng(42)
    
    step = 0
    while step < max_steps and time.time() < deadline:
        step += 1
        current = best_set[:]
        n = len(current)
        max_val = max(current) if current else 0
        
        # Try adding a random element
        if rng.random() < 0.4:
            # Add a random element not in the set
            candidates_to_add = []
            for _ in range(20):
                if rng.random() < 0.5:
                    # Small addition
                    cand = int(rng.integers(1, max(2, max_val + 5)))
                else:
                    # Larger addition
                    cand = int(rng.integers(max_val + 1, max_val * 2 + 10))
                if cand not in current and cand >= 0:
                    candidates_to_add.append(cand)
            
            for cand in candidates_to_add[:3]:
                new_set = sorted(current + [cand])
                s = _score_set(new_set)
                if s > best_score + 1e-10:
                    best_set = new_set
                    best_score = s
                    break
        
        # Try removing an element (except 0)
        elif rng.random() < 0.4:
            removable = [x for x in current if x != 0]
            if removable:
                for _ in range(min(3, len(removable))):
                    idx = int(rng.integers(len(removable)))
                    to_remove = removable[idx]
                    new_set = [x for x in current if x != to_remove]
                    if 0 not in new_set:
                        new_set.append(0)
                        new_set = sorted(new_set)
                    s = _score_set(new_set)
                    if s > best_score + 1e-10:
                        best_set = new_set
                        best_score = s
                        break
        
        # Try swapping an element
        else:
            swappable = [x for x in current if x != 0]
            if swappable:
                for _ in range(min(3, len(swappable))):
                    idx = int(rng.integers(len(swappable)))
                    to_swap = swappable[idx]
                    if rng.random() < 0.5:
                        new_val = int(rng.integers(1, max(2, max_val + 5)))
                    else:
                        new_val = int(rng.integers(max_val, max_val * 2 + 10))
                    if new_val not in current and new_val != to_swap and new_val >= 0:
                        new_set = sorted([x if x != to_swap else new_val for x in current])
                        s = _score_set(new_set)
                        if s > best_score + 1e-10:
                            best_set = new_set
                            best_score = s
                            break
    
    return best_set, best_score

def _polynomial_search(time_limit=45):
    """Search using polynomial structure ideas."""
    deadline = time.time() + time_limit
    best_set = None
    best_score = 0.0
    
    rng = np.random.default_rng(123)
    
    # Try sets with structured gaps
    for base in [2, 3, 4, 5]:
        for k in range(2, 8):
            if time.time() > deadline:
                break
            s = [0]
            val = 1
            for i in range(k):
                val *= base
                s.append(val)
            sc = _score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s[:]
    
    # Try mixed bases
    for _ in range(50):
        if time.time() > deadline:
            break
        k = int(rng.integers(3, 8))
        s = [0]
        val = 1
        for i in range(k):
            base = int(rng.integers(2, 6))
            val *= base
            s.append(val)
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Start with known good construction
    best_set = _try_known_constructions()
    best_score = _score_set(best_set)
    
    # Check time
    if time.time() > deadline - 60:
        return best_set
    
    # Polynomial-based search
    poly_set, poly_score = _polynomial_search(time_limit=30)
    if poly_score > best_score:
        best_set = poly_set
        best_score = poly_score
    
    if time.time() > deadline - 40:
        return best_set
    
    # Local search refinement
    ls_set, ls_score = _local_search(best_set, max_steps=3000, time_limit=35)
    if ls_score > best_score:
        best_set = ls_set
        best_score = ls_score
    
    if time.time() > deadline - 15:
        return best_set
    
    # Final local search pass
    ls_set2, ls_score2 = _local_search(best_set, max_steps=2000, time_limit=12)
    if ls_score2 > best_score:
        best_set = ls_set2
        best_score = ls_score2
    
    return best_set
# EVOLVE-BLOCK-END

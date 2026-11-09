"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_sidon_with_diagonals(subset):
    """Check if all pair sums (including diagonals) are unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find the largest subset of arr that is Sidon-with-diagonals."""
    n = len(arr)
    best = 0
    # Check all subsets from largest to smallest
    for size in range(n, 2, -1):
        for combo in combinations(arr, size):
            if _is_sidon_with_diagonals(combo):
                return size
    return 3  # Any 3-element set might work, but at minimum 3

def _max_sidon_size_fast(arr):
    """Faster check: only check up to size 4, since we suspect min is 3."""
    n = len(arr)
    # Check if any 4-subset is Sidon
    for combo in combinations(arr, 4):
        if _is_sidon_with_diagonals(combo):
            return 4
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Try structured sets: blocks of consecutive numbers
    # Pattern: k blocks of m consecutive numbers
    random.seed(42)
    
    # Try various block structures
    candidates = []
    
    # 3 blocks of 4
    for b1 in range(0, 10):
        for gap1 in range(5, 30):
            b2 = b1 + 4 + gap1
            if b2 + 4 > 129:
                continue
            for gap2 in range(5, 30):
                b3 = b2 + 4 + gap2
                if b3 + 4 > 129:
                    continue
                s = list(range(b1, b1+4)) + list(range(b2, b2+4)) + list(range(b3, b3+4))
                candidates.append(s)
    
    # 4 blocks of 3
    for b1 in range(0, 10):
        for gap1 in range(4, 20):
            b2 = b1 + 3 + gap1
            if b2 + 3 > 129:
                continue
            for gap2 in range(4, 20):
                b3 = b2 + 3 + gap2
                if b3 + 3 > 129:
                    continue
                for gap3 in range(4, 20):
                    b4 = b3 + 3 + gap3
                    if b4 + 3 > 129:
                        continue
                    s = list(range(b1, b1+3)) + list(range(b2, b2+3)) + list(range(b3, b3+3)) + list(range(b4, b4+3))
                    candidates.append(s)
    
    # 6 blocks of 2
    for b1 in range(0, 10):
        for gap1 in range(3, 15):
            b2 = b1 + 2 + gap1
            if b2 + 2 > 129:
                continue
            for gap2 in range(3, 15):
                b3 = b2 + 2 + gap2
                if b3 + 2 > 129:
                    continue
                for gap3 in range(3, 15):
                    b4 = b3 + 2 + gap3
                    if b4 + 2 > 129:
                        continue
                    for gap4 in range(3, 15):
                        b5 = b4 + 2 + gap4
                        if b5 + 2 > 129:
                            continue
                        for gap5 in range(3, 15):
                            b6 = b5 + 2 + gap5
                            if b6 + 2 > 129:
                                continue
                            s = list(range(b1, b1+2)) + list(range(b2, b2+2)) + list(range(b3, b3+2)) + list(range(b4, b4+2)) + list(range(b5, b5+2)) + list(range(b6, b6+2))
                            candidates.append(s)
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = cand
            if best_score == 3:
                break
    
    # Random search with local improvement
    if best_score > 3 and time.time() < deadline:
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Random set with structure bias
            base = random.sample(range(129), 12)
            score = _max_sidon_size_fast(base)
            if score < best_score:
                best_score = score
                best_set = base
                if best_score == 3:
                    break
            elif score == best_score:
                # Local search: swap one element
                for i in range(12):
                    if time.time() > deadline:
                        break
                    for new_val in random.sample([x for x in range(129) if x not in base], min(5, 128)):
                        trial = base[:]
                        trial[i] = new_val
                        trial.sort()
                        t_score = _max_sidon_size_fast(trial)
                        if t_score < best_score:
                            best_score = t_score
                            best_set = trial
                            base = trial
                            if best_score == 3:
                                break
                    if best_score == 3:
                        break
    
    return best_set

# EVOLVE-BLOCK-END

import time
import random
from math import log

def score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Compute sumset and difference set
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline):
    """Perform local search to improve the set U."""
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try moving individual elements
        for i in range(1, n):
            old_val = current[i]
            # Try a range of nearby values
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Ensure uniqueness
                if new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(new_set)
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if time.time() < deadline and n < 20:
            max_val = max(current)
            candidates_to_try = []
            # Values near max
            for new_val in range(max_val + 1, max_val + 30):
                candidates_to_try.append(new_val)
            # Gaps between existing elements
            for i in range(1, n):
                gap = current[i] - current[i-1]
                if gap > 3:
                    candidates_to_try.append(current[i-1] + 1)
                    candidates_to_try.append(current[i] - 1)
                    if gap > 5:
                        mid = (current[i-1] + current[i]) // 2
                        candidates_to_try.append(mid)
            random.shuffle(candidates_to_try)
            for new_val in candidates_to_try:
                if new_val <= 0 or new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing an element
        if time.time() < deadline and n > 4:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
    
    return current, current_score

def geometric_set(base, k):
    """Generate a geometric sequence: 0, 1, base, base^2, ..., base^(k-1)."""
    U = [0]
    val = 1
    for _ in range(k):
        U.append(val)
        val *= base
    return U

def base_expansion_set(base, digits, max_level):
    """Generate a set using base-expansion hierarchy.
    
    Elements are u_k = sum_{j=0}^k b_j * base^j where b_j in {0, 1}.
    We include all combinations up to a certain level.
    """
    U = set([0])
    # Generate all subsets of digits up to max_level
    for level in range(1, max_level + 1):
        for mask in range(1, 1 << level):
            val = 0
            for j in range(level):
                if mask & (1 << j):
                    val += digits[j] * (base ** j)
            U.add(val)
    return sorted(U)

def generate_candidates(deadline):
    """Generate diverse candidate sets."""
    candidates = []
    
    # 1. Geometric sequences with various bases
    for base in [2, 3, 4, 5, 7, 10, 13]:
        for k in range(3, 12):
            if time.time() > deadline - 20:
                return candidates
            U = geometric_set(base, k)
            candidates.append(U)
    
    # 2. Base-expansion sets
    for base in [2, 3, 4, 5]:
        for max_level in range(3, 8):
            if time.time() > deadline - 20:
                return candidates
            digits = [1] * max_level  # All digits are 1
            U = base_expansion_set(base, digits, max_level)
            if len(U) > 2:
                candidates.append(U)
    
    # 3. Powers with offsets
    for base in [2, 3, 4]:
        for offset in [0, 1, 2]:
            U = [0]
            val = 1 + offset
            for _ in range(10):
                U.append(val)
                val = val * base + offset
            candidates.append(U)
    
    # 4. Randomized geometric
    random.seed(12345)
    for seed in range(20):
        if time.time() > deadline - 20:
            return candidates
        random.seed(seed)
        base = random.choice([2, 3, 4, 5, 6, 7])
        k = random.randint(4, 10)
        U = [0]
        val = random.randint(1, 3)
        for _ in range(k):
            U.append(val)
            val = val * base + random.randint(-1, 1)
        if len(U) > 2:
            candidates.append(sorted(set(U)))
    
    # 5. Sparse random sets
    random.seed(999)
    for _ in range(15):
        if time.time() > deadline - 20:
            return candidates
        n_elem = random.randint(5, 14)
        max_val = random.randint(100, 2000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate and evaluate candidates
    candidates = generate_candidates(deadline)
    
    for U in candidates:
        if time.time() > deadline - 15:
            break
        try:
            refined, score = local_search(U, deadline)
            if score > best_score:
                best_score = score
                best_set = refined
        except Exception:
            continue
    
    return best_set

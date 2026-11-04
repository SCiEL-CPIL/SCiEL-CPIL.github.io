import time
import random
import math
import numpy as np

def fast_score(U):
    """Vectorized scoring using numpy."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    n = len(U)
    arr = np.array(U, dtype=np.int64)
    # Compute all sums and differences
    # sums: a[i] + a[j] for all i,j
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def score_set(U):
    """Original scoring interface."""
    return fast_score(U)

def geometric_set(base, k):
    """Generate a geometric progression set."""
    U = [0]
    val = 1
    for i in range(k - 1):
        U.append(val)
        val *= base
    return sorted(U)

def evaluate(U):
    """Evaluate a set, ensuring 0 is present."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    return fast_score(U)

def local_search(U, deadline, max_iter=3000):
    """Local search with add/remove/swap moves."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    best = U[:]
    best_score = fast_score(best)
    max_val = max(best) if best else 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a new element
            new_val = None
            for _ in range(15):
                if random.random() < 0.6:
                    base = random.choice(best)
                    new_val = base + random.randint(1, max(1, max_val // 4))
                else:
                    new_val = random.randint(1, max_val + max_val // 2)
                if new_val not in best and new_val > 0:
                    break
            if new_val is not None:
                U_new = best + [new_val]
                s = fast_score(U_new)
                if s > best_score:
                    best = U_new
                    best_score = s
                    max_val = max(best)
        elif move_type < 0.7:
            # Remove an element (not 0)
            removable = [x for x in best if x != 0]
            if removable:
                rem = random.choice(removable)
                U_new = [x for x in best if x != rem]
                s = fast_score(U_new)
                if s > best_score:
                    best = U_new
                    best_score = s
                    max_val = max(best) if best else 0
        else:
            # Swap an element
            removable = [x for x in best if x != 0]
            if removable:
                rem = random.choice(removable)
                U_base = [x for x in best if x != rem]
                for _ in range(8):
                    if random.random() < 0.5:
                        base = random.choice(U_base) if U_base else 0
                        new_val = base + random.randint(1, max(1, max_val // 4))
                    else:
                        new_val = random.randint(1, max_val + max_val // 2)
                    if new_val not in U_base and new_val > 0:
                        U_new = U_base + [new_val]
                        s = fast_score(U_new)
                        if s > best_score:
                            best = U_new
                            best_score = s
                            max_val = max(best) if best else 0
                            break
    
    return best, best_score

def generate_candidates(deadline, max_candidates=500):
    """Generate diverse candidate sets."""
    candidates = []
    
    # Geometric progressions
    for base in [2, 3, 4, 5, 6, 7, 8, 10]:
        for k in range(2, min(12, int(math.log(100000, base)) + 2)):
            s = geometric_set(base, k)
            if max(s) <= 100000:
                candidates.append(s)
    
    # Mixed structured sets
    for _ in range(100):
        U = set([0])
        # Add some powers
        for _ in range(random.randint(1, 3)):
            base = random.choice([2, 3, 5, 7])
            exp = random.randint(1, 5)
            U.add(base ** exp)
        # Add some random elements
        max_u = max(U)
        for _ in range(random.randint(1, 5)):
            U.add(random.randint(1, max_u + max_u // 2))
        candidates.append(sorted(U))
    
    # Arithmetic progressions
    for _ in range(50):
        a = random.randint(1, 10)
        d = random.randint(1, 50)
        k = random.randint(3, 10)
        U = [0] + [a + i * d for i in range(k)]
        candidates.append(sorted(U))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    random.seed(42)
    
    best_set = [0, 1, 3]
    best_score = fast_score(best_set)
    
    # Generate and evaluate candidates
    candidates = generate_candidates(deadline)
    
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        s = fast_score(cand)
        if s > best_score:
            best_score = s
            best_set = cand[:]
    
    # Local search from best
    if time.time() < deadline - 20:
        improved, imp_score = local_search(best_set, deadline - 10, max_iter=3000)
        if imp_score > best_score:
            best_score = imp_score
            best_set = improved
    
    # Multiple restarts
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        
        # Generate new seed
        if restart_count % 4 == 1:
            base = random.choice([2, 3, 4, 5, 6])
            k = random.randint(3, 9)
            seed = geometric_set(base, k)
        elif restart_count % 4 == 2:
            # Mixed
            U = set([0])
            for _ in range(random.randint(2, 4)):
                base = random.choice([2, 3, 5])
                exp = random.randint(1, 4)
                U.add(base ** exp)
            max_u = max(U) if U else 1
            for _ in range(random.randint(1, 3)):
                U.add(random.randint(1, max_u + max_u // 2))
            seed = sorted(U)
        else:
            # Random
            max_val = random.randint(10, 200)
            size = random.randint(4, 12)
            seed = [0] + sorted(random.sample(range(1, max_val + 1), min(size - 1, max_val)))
        
        s = fast_score(seed)
        if s > best_score:
            best_score = s
            best_set = seed[:]
        
        # Local search from this seed
        if time.time() < deadline - 5:
            improved, imp_score = local_search(seed, deadline - 5, max_iter=1500)
            if imp_score > best_score:
                best_score = imp_score
                best_set = improved
    
    # Final polish
    if time.time() < deadline - 2:
        improved, imp_score = local_search(best_set, deadline - 1, max_iter=5000)
        if imp_score > best_score:
            best_score = imp_score
            best_set = improved
    
    return sorted(best_set)

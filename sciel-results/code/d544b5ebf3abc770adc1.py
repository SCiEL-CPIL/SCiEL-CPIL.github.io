import time
import random
import math
from itertools import combinations

def score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0  # or compute anyway
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def evaluate(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def local_search(U, deadline, max_iter=5000):
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    best = U[:]
    best_score = evaluate(best)
    n = len(best)
    max_val = max(best) if best else 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try adding a random element
        if random.random() < 0.4:
            # Add new element
            candidates = []
            # Try values near existing ones or in gaps
            for _ in range(20):
                if random.random() < 0.5:
                    base = random.choice(best)
                    new_val = base + random.randint(1, max(1, max_val // 3))
                else:
                    new_val = random.randint(1, max_val + max_val // 2)
                if new_val not in best and new_val >= 0:
                    candidates.append(new_val)
            if candidates:
                new_val = random.choice(candidates)
                U_new = best + [new_val]
                s = evaluate(U_new)
                if s > best_score:
                    best = U_new
                    best_score = s
                    max_val = max(best)
        elif random.random() < 0.3:
            # Remove an element (not 0)
            removable = [x for x in best if x != 0]
            if removable:
                rem = random.choice(removable)
                U_new = [x for x in best if x != rem]
                s = evaluate(U_new)
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
                # Try replacing with new value
                for _ in range(10):
                    if random.random() < 0.5:
                        base = random.choice(U_base) if U_base else 0
                        new_val = base + random.randint(1, max(1, max_val // 3))
                    else:
                        new_val = random.randint(1, max_val + max_val // 2)
                    if new_val not in U_base and new_val >= 0:
                        U_new = U_base + [new_val]
                        s = evaluate(U_new)
                        if s > best_score:
                            best = U_new
                            best_score = s
                            max_val = max(best) if best else 0
                            break
    
    return best, best_score

def geometric_seed(base, k):
    U = [0]
    val = 1
    for i in range(k - 1):
        U.append(val)
        val *= base
    return U

def mixed_seed():
    # Mix of geometric progressions
    U = set([0])
    # Add powers of 2 up to some point
    v = 1
    for i in range(random.randint(2, 5)):
        U.add(v)
        v *= 2
    # Add powers of 3
    v = 3
    for i in range(random.randint(1, 3)):
        U.add(v)
        v *= 3
    # Add some random elements
    max_u = max(U)
    for _ in range(random.randint(1, 4)):
        U.add(random.randint(1, max_u + max_u // 2))
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    random.seed(42)
    
    best_set = [0, 1, 3]
    best_score = evaluate(best_set)
    
    # Try various seeds
    seeds = []
    
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        for k in range(2, 10):
            seeds.append(geometric_seed(base, k))
    
    # Mixed seeds
    for _ in range(20):
        seeds.append(mixed_seed())
    
    # Some structured sets
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 3, 9])
    seeds.append([0, 1, 3, 9, 27])
    seeds.append([0, 1, 4, 16])
    seeds.append([0, 1, 2, 4, 8])
    seeds.append([0, 1, 3, 7])
    seeds.append([0, 1, 3, 7, 15])
    
    # Evaluate all seeds
    for seed in seeds:
        if time.time() > deadline:
            break
        seed = sorted(set(seed))
        if 0 not in seed:
            seed.append(0)
            seed = sorted(seed)
        s = evaluate(seed)
        if s > best_score:
            best_score = s
            best_set = seed[:]
    
    # Local search from best seed and some random restarts
    # First, local search from the best
    if time.time() < deadline - 10:
        improved, imp_score = local_search(best_set, deadline - 5)
        if imp_score > best_score:
            best_score = imp_score
            best_set = improved
    
    # Multiple restarts
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        # Generate a new seed
        if restart_count % 3 == 1:
            seed = mixed_seed()
        elif restart_count % 3 == 2:
            base = random.choice([2, 3, 4, 5, 6])
            k = random.randint(2, 8)
            seed = geometric_seed(base, k)
        else:
            # Random set
            max_val = random.randint(5, 100)
            size = random.randint(3, 10)
            seed = [0] + sorted(random.sample(range(1, max_val + 1), min(size - 1, max_val)))
        
        seed = sorted(set(seed))
        if 0 not in seed:
            seed.append(0)
            seed = sorted(seed)
        
        s = evaluate(seed)
        if s > best_score:
            best_score = s
            best_set = seed[:]
        
        # Local search from this seed
        if time.time() < deadline - 5:
            improved, imp_score = local_search(seed, deadline - 5, max_iter=2000)
            if imp_score > best_score:
                best_score = imp_score
                best_set = improved
    
    # Final polish: try to improve best_set
    if time.time() < deadline - 2:
        improved, imp_score = local_search(best_set, deadline - 1, max_iter=5000)
        if imp_score > best_score:
            best_score = imp_score
            best_set = improved
    
    return sorted(best_set)

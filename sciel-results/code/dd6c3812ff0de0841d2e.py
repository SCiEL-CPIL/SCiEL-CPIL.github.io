import time
from math import log, gcd, exp
from itertools import combinations
import random

def _score_raw(U):
    """Score an already-normalized sorted set (contains 0, len>=2)."""
    n = len(U)
    if n < 2:
        return -1.0
    sums = set()
    diffs = set()
    # Use list for speed in inner loop if possible, but set is fine for n<100
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1:
        return -1.0
    D = 2 * U[-1] + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def normalize(U):
    """Return sorted distinct nonnegative ints containing 0, divided by gcd."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    g = 0
    for x in U:
        g = gcd(g, x)
    if g > 1:
        U = sorted(set(x // g for x in U))
    return U

def score_set(U):
    U = normalize(list(U))
    if len(U) < 2:
        return -1.0
    return _score_raw(U)

def local_search(U, deadline, max_val=300, max_n=30, T0=0.05):
    """Simulated-annealing local search in normalized space."""
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 500
    start_time = time.time()
    
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        n = len(U)
        new_U = None
        
        # Move generation
        if r < 0.35 and n < max_n:
            # Add a random element
            v = random.randint(1, max_val)
            if v not in U:
                new_U = normalize(U + [v])
        elif r < 0.50 and n > 2:
            # Remove a random element
            idx = random.randint(1, n - 1)
            new_U = normalize([x for j, x in enumerate(U) if j != idx])
        elif r < 0.70:
            # Replace a random element
            idx = random.randint(1, n - 1)
            v = random.randint(1, max_val)
            if v not in U:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        elif r < 0.85:
            # Swap two elements
            if n > 3:
                i = random.randint(1, n - 1)
                j = random.randint(1, n - 1)
                if i != j:
                    cand = list(U)
                    cand[i], cand[j] = cand[j], cand[i]
                    new_U = normalize(cand)
        else:
            # Perturb a random element
            idx = random.randint(1, n - 1)
            delta = random.randint(-15, 15)
            v = U[idx] + delta
            if v >= 1 and v not in U:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        
        if new_U is None or len(new_U) < 2:
            no_improve += 1
            continue
            
        new_score = _score_raw(new_U)
        
        if new_score > cur_score:
            U = new_U
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(U)
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                U = new_U
                cur_score = new_score
            no_improve += 1
            
        T = max(T * 0.995, 1e-6)
        
    return best_score, best_set

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin

    # Initial seeds
    seeds = [
        [0, 1, 3, 4],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 20, 35],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 5, 13, 25, 41],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 4, 10, 18, 28],
        [0, 1, 2, 5, 13, 34],
        [0, 1, 2, 4, 7, 9],
        [0, 1, 3, 6, 7, 11],
        [0, 1, 2, 3, 5, 7, 10, 12, 15],
        [0, 1, 2, 4, 6, 9, 11, 14],
        [0, 1, 3, 5, 8, 10, 13, 15],
    ]
    
    best_score = -1.0
    best_set = [0, 1]
    
    for s in seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))
            
    # Add random seeds
    for _ in range(20):
        n = random.randint(3, 10)
        vals = random.sample(range(1, 100), n)
        s = normalize([0] + vals)
        seeds.append(s)
        
    # Local search restarts
    restart_seeds = [list(best_set)] + seeds[:15]
    attempts = 0
    max_attempts = 1000 # Will be limited by time
    
    while time.time() < deadline - 10:
        base = restart_seeds[attempts % len(restart_seeds)]
        # Use a fraction of remaining time for each search, or just run until deadline - 10
        # To allow multiple restarts, we limit each search duration slightly
        search_deadline = min(deadline - 10, time.time() + 10)
        sc, s = local_search(base, search_deadline, max_val=200, max_n=25, T0=random.uniform(0.02, 0.08))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            # Add to restart seeds
            if len(restart_seeds) < 20:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1
        
    # Final polish on the best set found
    if time.time() < deadline - 2:
        cur = list(best_set)
        cur_score = _score_raw(cur)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n = len(cur)
            # Try adding/removing/replacing with small deltas
            for i in range(1, n):
                if time.time() > deadline - 2: break
                old = cur[i]
                for delta in range(-5, 6):
                    if delta == 0: continue
                    v = old + delta
                    if v < 1 or v in cur: continue
                    cand = list(cur)
                    cand[i] = v
                    newU = normalize(cand)
                    if len(newU) != n: continue
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        break
                if improved: break
            if improved: continue
            
            # Try adding
            if n < 25:
                for v in range(1, cur[-1] + 20):
                    if v in cur: continue
                    newU = normalize(cur + [v])
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        break
            if improved: continue
            
            # Try removing
            for i in range(1, n):
                newU = normalize([x for j, x in enumerate(cur) if j != i])
                if len(newU) < 2: continue
                sc = _score_raw(newU)
                if sc > cur_score:
                    cur, cur_score, improved = newU, sc, True
                    break
                    
        if cur_score > best_score:
            best_score = cur_score
            best_set = cur

    return normalize(list(best_set))

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

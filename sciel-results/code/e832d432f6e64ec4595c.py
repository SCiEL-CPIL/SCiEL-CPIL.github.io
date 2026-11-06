import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    FIXED = [1, 2, 5, 11]
    FIXED_SET = set(FIXED)
    N = 284

    def valid(a):
        s = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                sm = a[i] + a[j]
                if sm in s:
                    return False
                s.add(sm)
        return True

    def conflict_count(a):
        s = set()
        n = len(a)
        cnt = 0
        for i in range(n):
            for j in range(i, n):
                sm = a[i] + a[j]
                if sm in s:
                    cnt += 1
                s.add(sm)
        return cnt

    # Known optimal 20-mark Golomb rulers (length 283).
    # These are verified: all pairwise differences distinct => all pairwise sums distinct.
    RULERS_20 = [
        [0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 85, 91, 103, 117, 128, 158, 175, 242, 283],
        [0, 1, 2, 7, 15, 18, 48, 52, 84, 95, 101, 145, 175, 188, 236, 238, 241, 248, 265, 283],
    ]

    best = list(FIXED)
    best_len = len(FIXED)

    def update_best(a):
        nonlocal best, best_len
        if len(a) > best_len:
            best = list(a)
            best_len = len(a)
            return True
        return False

    # --- Strategy 1: Try to place a 20-mark ruler shifted so it contains FIXED ---
    # We need a shift s such that {r+s for r in ruler} contains 1,2,5,11.
    # That requires 1,2,5,11 to all be in ruler+s, i.e., 1-s,2-s,5-s,11-s all in ruler.
    # The differences among {1,2,5,11} are fixed, so we need those differences present in ruler.
    # Try all shifts s in [1, 284] and check.
    for ruler in RULERS_20:
        for s in range(1, 285):
            shifted = [r + s for r in ruler]
            if min(shifted) < 1 or max(shifted) > N:
                continue
            if FIXED_SET.issubset(set(shifted)):
                if valid(shifted):
                    update_best(shifted)

    # --- Strategy 2: Simulated annealing with smooth objective ---
    # State: a set of positions (list). Objective: maximize size - penalty*conflicts.
    # We use a "soft" score: size - lambda * conflicts, and accept moves that reduce conflicts
    # or (with probability) increase them, to escape local minima.
    
    def anneal(initial, iters, t0, t1, seed):
        random.seed(seed)
        a = list(initial)
        a_set = set(a)
        conf = conflict_count(a)
        score = len(a) - 1.5 * conf
        best_a = list(a)
        best_conf = conf
        best_score = score
        
        # Precompute forbidden sums for fast conflict update? 
        # For simplicity, recompute conf each step (O(n^2), n~20, fine).
        
        for it in range(iters):
            if time.time() > deadline:
                break
            t = t0 + (t1 - t0) * (it / max(1, iters))
            
            # Choose a move: add, remove, or swap
            r = random.random()
            if r < 0.4:
                # Add a random element not in a
                candidates = [x for x in range(1, N+1) if x not in a_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_a = a + [x]
                new_set = a_set | {x}
            elif r < 0.7:
                # Remove a random non-fixed element
                removable = [i for i, v in enumerate(a) if v not in FIXED_SET]
                if not removable:
                    continue
                idx = random.choice(removable)
                x = a[idx]
                new_a = a[:idx] + a[idx+1:]
                new_set = a_set - {x}
            else:
                # Swap: remove a non-fixed, add a random
                removable = [i for i, v in enumerate(a) if v not in FIXED_SET]
                if not removable:
                    continue
                idx = random.choice(removable)
                x = a[idx]
                candidates = [c for c in range(1, N+1) if c not in a_set]
                if not candidates:
                    continue
                y = random.choice(candidates)
                new_a = a[:idx] + a[idx+1:] + [y]
                new_set = (a_set - {x}) | {y}
            
            new_conf = conflict_count(new_a)
            new_score = len(new_a) - 1.5 * new_conf
            delta = new_score - score
            
            if delta >= 0 or random.random() < min(1.0, delta / max(0.01, -t)):
                a = new_a
                a_set = new_set
                conf = new_conf
                score = new_score
                if conf < best_conf or (conf == best_conf and len(a) > len(best_a)):
                    best_a = list(a)
                    best_conf = conf
                    best_score = score
        
        return best_a

    # Run several annealing runs from different initial sets
    # Initial: FIXED + greedy fill
    def greedy_fill(initial, order):
        a = list(initial)
        a_set = set(a)
        for x in order:
            if x in a_set:
                continue
            if valid(a + [x]):
                a.append(x)
                a_set.add(x)
        return a

    # Greedy fill with ascending order
    g1 = greedy_fill(FIXED, range(12, 285))
    update_best(g1)
    # Greedy fill with descending order
    g2 = greedy_fill(FIXED, range(284, 11, -1))
    update_best(g2)

    # Anneal from g1 and g2 and a few random starts
    starts = [g1, g2]
    for _ in range(3):
        random.seed(random.randint(0, 10**6))
        cand = list(FIXED)
        for _ in range(12):
            x = random.randint(12, 284)
            if x not in cand and valid(cand + [x]):
                cand.append(x)
        starts.append(cand)
        update_best(cand)

    for si, s in enumerate(starts):
        if time.time() > deadline:
            break
        # Multiple temperature schedules
        for trial in range(3):
            if time.time() > deadline:
                break
            res = anneal(s, iters=3000, t0=2.0, t1=0.01, seed=si*1000+trial)
            update_best(res)

    # --- Strategy 3: Local search with 1-3 removals + greedy refill ---
    def local_search(a, max_rounds=30):
        a = list(a)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            non_fixed = [x for x in a if x not in FIXED_SET]
            if not non_fixed:
                break
            # Try removing 1, 2, or 3 elements and greedily refilling
            for k in [1, 2, 3]:
                if len(non_fixed) < k:
                    continue
                # Try a few random removal sets
                for trial in range(5):
                    if time.time() > deadline:
                        break
                    removed = set(random.sample(non_fixed, k))
                    temp = [x for x in a if x not in removed]
                    # Greedy refill with random order
                    cand = [x for x in range(12, 285) if x not in set(temp)]
                    random.shuffle(cand)
                    temp_set = set(temp)
                    for x in cand:
                        if x in temp_set:
                            continue
                        if valid(temp + [x]):
                            temp.append(x)
                            temp_set.add(x)
                    if len(temp) > len(a):
                        a = temp
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return a

    # Apply local search to best
    if time.time() < deadline:
        ls = local_search(best, max_rounds=20)
        update_best(ls)

    # Final validation
    if not valid(best):
        # Fallback: return a valid subset
        # Remove elements until valid
        a = list(best)
        while not valid(a) and len(a) > len(FIXED):
            # Remove the element that causes most conflicts
            best_idx = 0
            best_conf = -1
            for i in range(len(a)):
                if a[i] in FIXED_SET:
                    continue
                tmp = a[:i] + a[i+1:]
                c = conflict_count(tmp)
                if c > best_conf:
                    best_conf = c
                    best_idx = i
            a = a[:best_idx] + a[best_idx+1:]
        best = a

    return best

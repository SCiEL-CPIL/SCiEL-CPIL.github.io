import random
import time

def _score(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    diffs = set()
    lst = list(S_set)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            a = lst[i]
            c = lst[j]
            s = a + c
            if s % 2 == 0:
                b = s // 2
                if b in S_set and b > a:
                    d = c - b  # = (c - a) / 2
                    if d > 0:
                        diffs.add(d)
    return len(diffs)

def _initial_grid():
    """4x8 grid construction."""
    M = 100
    S = set()
    for i in range(4):
        for j in range(8):
            S.add(i * M + j + 1)
    return S

def _random_individual():
    return set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Start with grid construction
    S = _initial_grid()
    best = set(S)
    best_score = _score(best)
    current_score = best_score
    
    # Also try a few random starts to see if any are better
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        S_rand = _random_individual()
        sc = _score(S_rand)
        if sc > best_score:
            best = set(S_rand)
            best_score = sc
            current_score = sc
            S = set(S_rand)
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.99999
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            T = T_min
        
        # Generate a move
        S_list = list(S)
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap two elements
            i, j = random.sample(range(32), 2)
            new_S = set(S_list)
            a = new_S.pop()
            new_S.add(S_list[j])  # placeholder, will fix
            # Actually, let's do it properly
            new_S = set(S)
            a, b = S_list[i], S_list[j]
            # Swap: remove a and b, add two new random values
            # No, swap means exchange positions - but set doesn't have positions
            # Let's do: remove one element, add another
            new_S = set(S)
            removed = S_list[i]
            new_S.discard(removed)
            new_elem = random.randint(1, 512)
            if new_elem not in new_S:
                new_S.add(new_elem)
            else:
                continue
            new_score = _score(new_S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best = set(S)
                    best_score = current_score
        elif move_type < 0.8:
            # Remove one, add one (different from above but same effect)
            new_S = set(S)
            removed = random.choice(list(S))
            new_S.discard(removed)
            new_elem = random.randint(1, 512)
            if new_elem not in new_S:
                new_S.add(new_elem)
            else:
                continue
            new_score = _score(new_S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best = set(S)
                    best_score = current_score
        else:
            # Swap: remove two, add two
            new_S = set(S)
            rem1, rem2 = random.sample(list(S), 2)
            new_S.discard(rem1)
            new_S.discard(rem2)
            add1 = random.randint(1, 512)
            add2 = random.randint(1, 512)
            if add1 not in new_S and add2 not in new_S and add1 != add2:
                new_S.add(add1)
                new_S.add(add2)
            else:
                continue
            new_score = _score(new_S)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                S = new_S
                current_score = new_score
                if current_score > best_score:
                    best = set(S)
                    best_score = current_score
        
        T *= cooling
    
    # Final local search polish on best
    if time.time() < deadline - 1.0:
        best_list = list(best)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(32):
                if time.time() > deadline - 0.5:
                    break
                old_val = best_list[i]
                for new_val in range(1, 513):
                    if new_val == old_val:
                        continue
                    if new_val in best_list:
                        continue
                    trial = list(best_list)
                    trial[i] = new_val
                    trial_set = set(trial)
                    if len(trial_set) != 32:
                        continue
                    sc = _score(trial_set)
                    if sc > best_score:
                        best_score = sc
                        best_list = trial
                        best = set(best_list)
                        improved = True
                        break
    
    return sorted(best)


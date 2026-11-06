"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_bitset(mask):
    """Count distinct d>0 such that exists a with a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 257):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _sa_bitset(init_arr, deadline, rng):
    """Simulated annealing with bitset scoring."""
    n = 32
    cur_arr = list(init_arr)
    cur_mask = 0
    for x in cur_arr:
        cur_mask |= (1 << x)
    cur_score = _score_bitset(cur_mask)
    best_arr = cur_arr[:]
    best_score = cur_score
    
    T = 25.0
    T_min = 0.05
    alpha = 0.9997
    steps_since_best = 0
    max_steps_since_best = 10000
    
    e = 2.718281828459045
    
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        move_type = rng.random()
        if move_type < 0.50:
            # Single swap
            idx = rng.randrange(n)
            old = cur_arr[idx]
            new = rng.randint(1, 512)
            if new == old or (cur_mask >> new) & 1:
                continue
            new_mask = (cur_mask & ~(1 << old)) | (1 << new)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[idx] = new
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        elif move_type < 0.75:
            # Double swap
            i1, i2 = rng.sample(range(n), 2)
            old1, old2 = cur_arr[i1], cur_arr[i2]
            new1, new2 = rng.sample(range(1, 513), 2)
            if new1 == new2:
                continue
            if (cur_mask >> new1) & 1 or (cur_mask >> new2) & 1:
                continue
            new_mask = (cur_mask & ~(1 << old1) & ~(1 << old2)) | (1 << new1) | (1 << new2)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[i1] = new1
                cur_arr[i2] = new2
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        else:
            # Local perturbation
            idx = rng.randrange(n)
            old = cur_arr[idx]
            shift = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 24, 32])
            new = old + shift
            if new < 1 or new > 512 or (cur_mask >> new) & 1:
                continue
            new_mask = (cur_mask & ~(1 << old)) | (1 << new)
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(e, delta / T):
                cur_mask = new_mask
                cur_arr[idx] = new
                cur_arr.sort()
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        
        T *= alpha
        
        if steps_since_best > max_steps_since_best:
            break
    
    return best_arr, best_score

def _hill_climb_bitset(arr, deadline, rng):
    """Exhaustive single-swap hill climbing with bitset scoring."""
    cur_arr = list(arr)
    cur_mask = 0
    for x in cur_arr:
        cur_mask |= (1 << x)
    cur_score = _score_bitset(cur_mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(32):
            old = cur_arr[idx]
            for new in range(1, 513):
                if new == old or (cur_mask >> new) & 1:
                    continue
                new_mask = (cur_mask & ~(1 << old)) | (1 << new)
                trial_score = _score_bitset(new_mask)
                if trial_score > cur_score:
                    cur_mask = new_mask
                    cur_arr[idx] = new
                    cur_arr.sort()
                    cur_score = trial_score
                    improved = True
                    break
            if improved:
                break
    return cur_arr, cur_score

def _random_pair_hill_climb(arr, deadline, rng, n_tries=3000):
    """Randomly try pairs of moves to escape local optima."""
    cur_arr = list(arr)
    cur_mask = 0
    for x in cur_arr:
        cur_mask |= (1 << x)
    cur_score = _score_bitset(cur_mask)
    
    for _ in range(n_tries):
        if time.time() > deadline:
            break
        i1, i2 = rng.sample(range(32), 2)
        old1, old2 = cur_arr[i1], cur_arr[i2]
        new1, new2 = rng.sample(range(1, 513), 2)
        if new1 == new2:
            continue
        if (cur_mask >> new1) & 1 or (cur_mask >> new2) & 1:
            continue
        new_mask = (cur_mask & ~(1 << old1) & ~(1 << old2)) | (1 << new1) | (1 << new2)
        trial_score = _score_bitset(new_mask)
        if trial_score > cur_score:
            cur_mask = new_mask
            cur_arr[i1] = new1
            cur_arr[i2] = new2
            cur_arr.sort()
            cur_score = trial_score
            # After improvement, do a quick single hill-climb
            arr2, score2 = _hill_climb_bitset(cur_arr, deadline, rng)
            if score2 > cur_score:
                cur_arr = arr2
                cur_score = score2
    
    return cur_arr, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    rng = random.Random(42)
    
    inits = []
    # Dense blocks at various positions
    inits.append(list(range(1, 33)))
    inits.append(list(range(16, 48)))
    inits.append(list(range(32, 64)))
    inits.append(list(range(64, 96)))
    # Structured with small gaps
    for d in [1, 2, 3, 4]:
        for start in [1, 10, 20, 40]:
            if start + 31*d <= 512:
                inits.append(list(range(start, start + 32*d, d))[:32])
    # Random
    for seed in [1, 2, 3, 4, 5]:
        r = random.Random(seed)
        inits.append(sorted(r.sample(range(1, 513), 32)))
    
    best_overall = None
    best_overall_score = 0
    
    for init in inits:
        if time.time() > deadline - 8.0:
            break
        arr, score = _sa_bitset(init, deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr
    
    if best_overall is None:
        best_overall = list(range(1, 33))
        mask = 0
        for x in best_overall:
            mask |= (1 << x)
        best_overall_score = _score_bitset(mask)
    
    # Hill-climb polish
    if time.time() < deadline - 5.0:
        arr, score = _hill_climb_bitset(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score
    
    # Random pair hill-climb to escape local optima
    if time.time() < deadline - 3.0:
        arr, score = _random_pair_hill_climb(best_overall, deadline, rng, n_tries=3000)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score
    
    # Final single hill-climb pass
    if time.time() < deadline - 1.0:
        arr, score = _hill_climb_bitset(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score
    
    result = sorted(set(best_overall))
    if len(result) != 32:
        result = list(range(1, 33))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
    
    return result
# EVOLVE-BLOCK-END

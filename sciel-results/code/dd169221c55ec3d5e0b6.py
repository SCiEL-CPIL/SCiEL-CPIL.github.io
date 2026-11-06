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
    
    T = 30.0
    T_min = 0.05
    alpha = 0.9995
    steps_since_best = 0
    max_steps_since_best = 15000
    
    e = 2.718281828459045
    
    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        move_type = rng.random()
        if move_type < 0.45:
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
        elif move_type < 0.70:
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
        elif move_type < 0.90:
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
        else:
            # Kick: randomize 1-3 elements to escape local optima
            n_kick = rng.randint(1, 3)
            kick_idxs = rng.sample(range(n), n_kick)
            new_mask = cur_mask
            for idx in kick_idxs:
                old = cur_arr[idx]
                new_mask &= ~(1 << old)
                new_val = rng.randint(1, 512)
                while (new_mask >> new_val) & 1:
                    new_val = rng.randint(1, 512)
                new_mask |= (1 << new_val)
                cur_arr[idx] = new_val
            cur_arr.sort()
            trial_score = _score_bitset(new_mask)
            delta = trial_score - cur_score
            # Accept kicks more aggressively when stuck
            accept_prob = 0.3 if steps_since_best > 5000 else 0.1
            if delta >= 0 or rng.random() < accept_prob:
                cur_mask = new_mask
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                # Revert
                cur_arr = list(best_arr)
                cur_mask = 0
                for x in cur_arr:
                    cur_mask |= (1 << x)
                cur_score = best_score
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
        # Try all single moves, keep best improvement
        best_idx = -1
        best_new = -1
        best_score = cur_score
        for idx in range(32):
            old = cur_arr[idx]
            for new in range(1, 513):
                if new == old or (cur_mask >> new) & 1:
                    continue
                new_mask = (cur_mask & ~(1 << old)) | (1 << new)
                trial_score = _score_bitset(new_mask)
                if trial_score > best_score:
                    best_score = trial_score
                    best_idx = idx
                    best_new = new
        if best_idx >= 0:
            old = cur_arr[best_idx]
            cur_mask = (cur_mask & ~(1 << old)) | (1 << best_new)
            cur_arr[best_idx] = best_new
            cur_arr.sort()
            cur_score = best_score
            improved = True
        # If no single move improves, try double moves
        if not improved:
            best_i1, best_i2 = -1, -1
            best_n1, best_n2 = -1, -1
            for i1 in range(32):
                if time.time() > deadline:
                    break
                old1 = cur_arr[i1]
                for i2 in range(i1 + 1, 32):
                    old2 = cur_arr[i2]
                    # Try a few random pairs
                    for _ in range(5):
                        n1, n2 = rng.sample(range(1, 513), 2)
                        if (cur_mask >> n1) & 1 or (cur_mask >> n2) & 1:
                            continue
                        new_mask = (cur_mask & ~(1 << old1) & ~(1 << old2)) | (1 << n1) | (1 << n2)
                        trial_score = _score_bitset(new_mask)
                        if trial_score > best_score:
                            best_score = trial_score
                            best_i1, best_i2 = i1, i2
                            best_n1, best_n2 = n1, n2
            if best_i1 >= 0:
                old1 = cur_arr[best_i1]
                old2 = cur_arr[best_i2]
                cur_mask = (cur_mask & ~(1 << old1) & ~(1 << old2)) | (1 << best_n1) | (1 << best_n2)
                cur_arr[best_i1] = best_n1
                cur_arr[best_i2] = best_n2
                cur_arr.sort()
                cur_score = best_score
                improved = True
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
    deadline = start_time + 145.0
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
    # Random with multiple seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r = random.Random(seed)
        inits.append(sorted(r.sample(range(1, 513), 32)))
    
    best_overall = None
    best_overall_score = 0
    
    # Phase 1: SA with multiple inits, each with its own rng seed
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        local_rng = random.Random(42 + i * 1000)
        arr, score = _sa_bitset(init, deadline, local_rng)
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# Bitset-based scoring: represent the 32-element set as a Python int bitmask.
# Bit i (0-indexed) is set iff (i+1) is in the set. So value x maps to bit (x-1).
# To check if a, a+d, a+2d are all in the set:
#   bit a-1, bit (a+d)-1, bit (a+2d)-1 all set.
# For a fixed d, we need: exists a such that bits (a-1), (a+d-1), (a+2d-1) are all set.
# This is equivalent to: (mask & (mask >> d) & (mask >> (2*d))) != 0
# where we treat the mask as having bits 0..511 corresponding to values 1..512.
# The number of such d values (d > 0) is our score.

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set (mask is bitset of values 1..512)."""
    count = 0
    # d can range from 1 to 256 (since a+2d <= 512 means d <= 255 for a>=1)
    # Actually max d: a=1, a+2d<=512 => d<=255.5, so d<=255
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_arr(mask):
    arr = []
    for x in range(1, 513):
        if mask & (1 << (x - 1)):
            arr.append(x)
    return arr

def _random_initial():
    return sorted(random.sample(range(1, 513), 32))

def _dense_initial():
    start = random.randint(1, 481)
    return list(range(start, start + 32))

def _structured_initial():
    d = random.randint(1, 15)
    start = random.randint(1, 512 - 31 * d)
    base = list(range(start, start + 32 * d, d))[:32]
    remaining = [x for x in range(1, 513) if x not in base]
    if len(base) < 32:
        extra = random.sample(remaining, 32 - len(base))
        base.extend(extra)
    return sorted(base)

def _sa_bitset(arr, deadline, rng):
    """Simulated annealing using bitset representation."""
    n = 32
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_score = cur_score
    best_arr = cur_arr[:]

    T = 50.0
    T_min = 0.5
    alpha = 0.997
    steps_since_best = 0
    max_steps_since_best = 3000

    while True:
        if time.time() > deadline:
            break
        if T < T_min:
            break

        move_type = rng.random()
        if move_type < 0.55:
            # Single swap
            idx = rng.randrange(n)
            old = cur_arr[idx]
            new = rng.randint(1, 512)
            if new == old or (cur_mask & (1 << (new - 1))):
                continue
            trial_mask = cur_mask & ~(1 << (old - 1)) | (1 << (new - 1))
            trial_score = _score_bitset(trial_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                cur_arr[idx] = new
                cur_arr.sort()
                cur_mask = trial_mask
                cur_score = trial_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_arr = cur_arr[:]
                    steps_since_best = 0
                else:
                    steps_since_best += 1
            else:
                steps_since_best += 1
        elif move_type < 0.80:
            # Double swap
            i1, i2 = rng.sample(range(n), 2)
            old1, old2 = cur_arr[i1], cur_arr[i2]
            new1, new2 = rng.sample(range(1, 513), 2)
            if new1 == new2:
                continue
            b1 = 1 << (new1 - 1)
            b2 = 1 << (new2 - 1)
            if (cur_mask & b1) or (cur_mask & b2):
                continue
            trial_mask = cur_mask & ~(1 << (old1 - 1)) & ~(1 << (old2 - 1)) | b1 | b2
            trial_score = _score_bitset(trial_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                cur_arr[i1] = new1
                cur_arr[i2] = new2
                cur_arr.sort()
                cur_mask = trial_mask
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
            # Local shift
            idx = rng.randrange(n)
            old = cur_arr[idx]
            shift = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 11])
            new = old + shift
            if new < 1 or new > 512 or (cur_mask & (1 << (new - 1))):
                continue
            trial_mask = cur_mask & ~(1 << (old - 1)) | (1 << (new - 1))
            trial_score = _score_bitset(trial_mask)
            delta = trial_score - cur_score
            if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                cur_arr[idx] = new
                cur_arr.sort()
                cur_mask = trial_mask
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

def _hill_climb_bitset(arr, deadline):
    """Greedy hill-climbing using bitset scoring."""
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_arr = cur_arr[:]
    best_score = cur_score
    rng = random.Random(123)

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for idx in range(32):
            old = cur_arr[idx]
            # Try all possible replacements
            for new in range(1, 513):
                if new == old or (cur_mask & (1 << (new - 1))):
                    continue
                trial_mask = cur_mask & ~(1 << (old - 1)) | (1 << (new - 1))
                trial_score = _score_bitset(trial_mask)
                if trial_score > cur_score:
                    cur_arr[idx] = new
                    cur_arr.sort()
                    cur_mask = trial_mask
                    cur_score = trial_score
                    best_arr = cur_arr[:]
                    best_score = cur_score
                    improved = True
                    break
            if improved:
                break
    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    rng = random.Random(42)

    # Generate diverse initial solutions
    inits = []
    inits.append(_dense_initial())
    inits.append(_structured_initial())
    inits.append(_random_initial())
    inits.append(_random_initial())
    inits.append(_structured_initial())
    inits.append(_random_initial())
    inits.append(_structured_initial())
    inits.append(_random_initial())

    best_overall = None
    best_overall_score = 0

    # Run SA on each init, but limit time per SA run
    n_inits = len(inits)
    per_init_time = (deadline - start_time) / n_inits

    for i, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        sa_deadline = min(deadline - 5.0, time.time() + remaining - 2.0)
        arr, score = _sa_bitset(init, sa_deadline, rng)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = arr

    if best_overall is None:
        best_overall = _dense_initial()
        best_overall_score = _score_bitset(_arr_to_mask(best_overall))

    # Hill-climbing polish on the best
    if time.time() < deadline - 5.0:
        arr, score = _hill_climb_bitset(best_overall, deadline)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score

    # Ensure validity: exactly 32 distinct integers in 1..512
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

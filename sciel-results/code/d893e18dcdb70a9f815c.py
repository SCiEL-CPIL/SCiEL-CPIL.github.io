"""
Low discrepancy signs on all arithmetic progressions in [1, 256].

Structural change: replace greedy 1-flip hill-climb with simulated annealing
(a stochastic global search method) that accepts worsening moves with a
temperature-dependent probability. This enables escaping local minima that
trap the greedy approach. Multi-start from diverse initializations
(Thue-Morse, Legendre symbol, random) each followed by SA.
"""
import time
import random
import math
import numpy as np

_N = 256


def _build_ap_table():
    """
    Precompute AP position lists and per-position AP membership for fast
    flip-based evaluation.
    
    Returns:
        ap_positions: list of lists, ap_positions[a] = list of positions in AP a
        pos_aps: pos_aps[p] = list of AP indices a such that p is in AP a
    """
    n = _N
    ap_positions = []
    pos_aps = [[] for _ in range(n)]
    
    for s in range(n):
        for d in range(1, n):
            positions = []
            x = s
            while x < n:
                positions.append(x)
                x += d
            if len(positions) >= 2:
                ap_positions.append(positions)
                ap_idx = len(ap_positions) - 1
                for p in positions:
                    pos_aps[p].append(ap_idx)
    
    return ap_positions, pos_aps


def _eval_max_abs(signs, ap_positions, pos_aps):
    """Compute max absolute AP sum from scratch."""
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_abs = 1
    for positions in ap_positions:
        s = int(np.sum(signs[positions]))
        a = abs(s)
        if a > max_abs:
            max_abs = a
    return max_abs


def _legendre_init(n):
    """
    Initialize using Legendre symbol modulo a large prime near n.
    For prime p, Legendre symbol (a|p) = 1 if a is a quadratic residue mod p,
    -1 otherwise, 0 if a ≡ 0 mod p. Map 0 -> +1.
    This gives a pseudo-random but structured ±1 sequence.
    """
    # Find a prime p >= n
    def is_prime(x):
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        for i in range(3, int(math.isqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        a = (i + 1) % p  # use 1-indexed positions
        if a == 0:
            signs[i] = 1
        else:
            # Euler's criterion: (a|p) = a^((p-1)/2) mod p
            ls = pow(a, (p - 1) // 2, p)
            if ls == 1:
                signs[i] = 1
            else:
                signs[i] = -1
    return signs


def _simulated_annealing(signs, ap_positions, pos_aps, max_time, rng, deadline):
    """
    Simulated annealing with fast flip evaluation.
    
    Accepts worsening moves with probability exp(-delta / T), where T is the
    current temperature that cools over time. This allows escaping local minima.
    """
    n = len(signs)
    signs = np.array(signs, dtype=np.int8)
    
    # Initialize AP_sums
    ap_sums = np.zeros(len(ap_positions), dtype=np.int32)
    for a, positions in enumerate(ap_positions):
        ap_sums[a] = int(np.sum(signs[positions]))
    
    current_max = int(np.max(np.abs(ap_sums)))
    current_max = max(current_max, 1)
    
    best_max = current_max
    best_signs = signs.copy()
    
    # Temperature schedule: start high, cool geometrically
    T_start = 10.0
    T_end = 0.1
    # Estimate number of steps we can do
    # Each step: pick random position, flip, update ~n/2 APs, compute max over ~32K APs
    # Rough estimate: ~50-100us per step in Python with numpy max
    # Budget: max_time seconds
    # Let's use a fixed number of cooling epochs
    num_epochs = 50
    cooling_rate = (T_end / T_start) ** (1.0 / num_epochs)
    
    T = T_start
    epoch = 0
    
    while time.time() < deadline:
        # One epoch: try n random flips
        for _ in range(n):
            if time.time() > deadline:
                return best_signs.tolist(), best_max
            
            p = rng.randint(0, n - 1)
            old_sign = signs[p]
            signs[p] = -old_sign
            for a in pos_aps[p]:
                ap_sums[a] -= 2 * old_sign
            
            new_max = int(np.max(np.abs(ap_sums)))
            new_max = max(new_max, 1)
            
            delta = new_max - current_max
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept move
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[p] = old_sign
                for a in pos_aps[p]:
                    ap_sums[a] += 2 * old_sign
        
        epoch += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
            # Restart with a small perturbation from best
            if epoch % 10 == 0:
                signs = best_signs.copy()
                for a, positions in enumerate(ap_positions):
                    ap_sums[a] = int(np.sum(signs[positions]))
                current_max = int(np.max(np.abs(ap_sums)))
                current_max = max(current_max, 1)
                # Small random kick
                for _ in range(rng.randint(1, 3)):
                    p = rng.randint(0, n - 1)
                    old_sign = signs[p]
                    signs[p] = -old_sign
                    for a in pos_aps[p]:
                        ap_sums[a] -= 2 * old_sign
                current_max = int(np.max(np.abs(ap_sums)))
                current_max = max(current_max, 1)
                T = T_start * 0.5  # Partial reheat
    
    return best_signs.tolist(), best_max


def _thue_morse_init(n):
    """
    Thue-Morse sequence: t(i) = parity of number of 1s in binary representation of i.
    Map even -> +1, odd -> -1.
    This is a 1-automatic sequence with low discrepancy on intervals.
    """
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        bits = bin(i).count('1')
        signs[i] = 1 if bits % 2 == 0 else -1
    return signs


def construct():
    start_time = time.time()
    total_budget = 55  # 60s limit, 5s safety margin
    
    rng = random.Random(42)
    
    # Precompute AP tables (expensive, done once)
    ap_positions, pos_aps = _build_ap_table()
    
    deadline = start_time + total_budget
    
    best_signs = None
    best_max = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. Thue-Morse
    signs_tm = _thue_morse_init(_N)
    inits.append(signs_tm.copy())
    
    # 2. Legendre symbol
    signs_leg = _legendre_init(_N)
    inits.append(signs_leg.copy())
    
    # 3. Random initializations
    for _ in range(4):
        signs_r = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
        inits.append(signs_r)
    
    # 4. Thue-Morse with small perturbations
    for _ in range(2):
        signs_p = signs_tm.copy()
        for _ in range(rng.randint(2, 5)):
            p = rng.randint(0, _N - 1)
            signs_p[p] = -signs_p[p]
        inits.append(signs_p)
    
    # Evaluate all initializations and find the best
    for signs_init in inits:
        m = _eval_max_abs(signs_init, ap_positions, pos_aps)
        if m < best_max:
            best_max = m
            best_signs = signs_init.copy()
    
    # Run simulated annealing from each of the top initializations
    # Sort by initial quality
    init_scores = []
    for i, signs_init in enumerate(inits):
        m = _eval_max_abs(signs_init, ap_positions, pos_aps)
        init_scores.append((m, i))
    init_scores.sort()
    
    # Use remaining time for SA runs
    remaining = deadline - time.time()
    if remaining > 3 and best_signs is not None:
        # Allocate time among top initializations
        num_sa_runs = min(5, len(inits))
        time_per_run = (remaining - 2) / num_sa_runs  # 2s margin
        
        for rank in range(num_sa_runs):
            if time.time() > deadline - 1:
                break
            _, idx = init_scores[rank]
            signs_start = inits[idx].copy()
            
            run_deadline = min(deadline - 0.5, time.time() + time_per_run)
            result_signs, result_max = _simulated_annealing(
                signs_start, ap_positions, pos_aps, time_per_run, rng, run_deadline
            )
            if result_max < best_max:
                best_max = result_max
                best_signs = np.array(result_signs, dtype=np.int8)
    
    return [int(x) for x in best_signs]

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _all_ap_sums(signs):
    """
    Compute max |sum| over all arithmetic progressions in [1, n].
    Uses vectorized computation for speed.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _legendre_init():
    """
    Initialize using Legendre symbol modulo a prime near 256.
    The Legendre symbol (p/n) is +1 if n is a quadratic residue mod p,
    -1 otherwise, 0 if p|n. This provides theoretically bounded discrepancy.
    """
    n = _N
    # Find a prime near n (use 257 which is prime)
    p = 257
    
    # Compute Legendre symbol for each position 1..n
    signs = np.zeros(n, dtype=np.int32)
    for i in range(n):
        pos = i + 1  # positions are 1-indexed
        if pos % p == 0:
            signs[i] = 1  # avoid zero, use +1
        else:
            # Legendre symbol: (pos/p) = pos^((p-1)/2) mod p
            ls = pow(pos, (p - 1) // 2, p)
            if ls == 1:
                signs[i] = 1
            elif ls == p - 1:
                signs[i] = -1
            else:
                signs[i] = 1  # shouldn't happen for prime p
    
    return signs.tolist()

def _simulated_annealing(signs_list, max_time, rng, T_start=10.0, T_end=0.1, cooling=0.999):
    """
    Simulated annealing: flip random positions with Metropolis acceptance.
    Can escape local minima by accepting worse solutions with probability exp(-dE/T).
    """
    n = _N
    signs = np.array(signs_list, dtype=np.int32)
    current_disc = _all_ap_sums(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    T = T_start
    
    while time.time() - start_time < max_time:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        new_disc = _all_ap_sums(signs)
        dE = new_disc - current_disc
        
        if dE <= 0:
            # Accept improvement
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Metropolis criterion: accept with probability exp(-dE/T)
            if rng.random() < np.exp(-dE / T):
                current_disc = new_disc
            else:
                # Revert
                signs[i] = old_sign
        
        # Cool down
        T = max(T * cooling, T_end)
        
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
    
    return best_signs.tolist(), best_disc

def construct():
    """
    Construct 256 signs minimizing max |sum| over all arithmetic progressions.
    Strategy: Legendre symbol init + simulated annealing with multi-start.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Phase 1: Algebraic initialization using Legendre symbol
    legendre_signs = _legendre_init()
    legendre_disc = _all_ap_sums(legendre_signs)
    
    # Phase 2: Simulated annealing from Legendre start
    remaining_time = _DEADLINE - 5 - (time.time() - start_time)
    if remaining_time > 1:
        sa_signs, sa_disc = _simulated_annealing(legendre_signs, remaining_time, rng)
        if sa_disc <= legendre_disc:
            legendre_signs = sa_signs
            legendre_disc = sa_disc
    
    # Check deadline
    if time.time() > _DEADLINE - 2:
        return legendre_signs
    
    # Phase 3: Multi-start with random inits and short SA
    best_signs = legendre_signs
    best_disc = legendre_disc
    
    for restart in range(20):
        if time.time() > _DEADLINE - 5:
            break
        
        # Random init
        random_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        
        # Short simulated annealing
        short_time = min(2.0, _DEADLINE - 5 - (time.time() - start_time))
        if short_time < 0.5:
            break
        
        sa_signs, sa_disc = _simulated_annealing(random_signs, short_time, rng, T_start=5.0, T_end=0.1, cooling=0.999)
        
        if sa_disc < best_disc:
            best_disc = sa_disc
            best_signs = sa_signs
    
    return best_signs
# EVOLVE-BLOCK-END

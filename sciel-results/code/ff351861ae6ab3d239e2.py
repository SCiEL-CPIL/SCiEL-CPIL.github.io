# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)
ZERO = Fraction(0, 1)

def _try_improve(best_idx, best_deficit, deadline):
    """Local search: 1-swap, 1-add, 1-remove moves to reduce deficit while keeping sum < 1."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Try 1-swap: remove i, add j
        for i in best_idx[:]:
            if time.time() >= deadline:
                break
            f_out = FRAC_DENOMS[i]
            sum_without = best_sum - f_out
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = sum_without + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.remove(i)
                    best_idx.append(j)
                    in_set.remove(i)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
            if changed:
                break
        
        # Try 1-add: add j if it fits
        if not changed:
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = best_sum + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.append(j)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
        
        # Try 1-remove: remove i if it improves deficit
        if not changed:
            for i in best_idx[:]:
                if time.time() >= deadline:
                    break
                f_out = FRAC_DENOMS[i]
                new_sum = best_sum - f_out
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.remove(i)
                    in_set.remove(i)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
    
    return best_idx, best_deficit

def _pairwise_swap_improve(best_idx, best_deficit, deadline):
    """Try pairwise swaps (remove two, add two) for larger jumps."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    k = len(best_idx)
    
    if k < 2:
        return best_idx, best_deficit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_remove = len(best_idx)
        for a in range(n_remove):
            if time.time() >= deadline:
                break
            for b in range(a+1, n_remove):
                i, j = best_idx[a], best_idx[b]
                sum_without_two = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j]
                for x in range(N):
                    if x in in_set:
                        continue
                    f_x = FRAC_DENOMS[x]
                    sum_after_x = sum_without_two + f_x
                    if sum_after_x >= ONE:
                        continue
                    for y in range(x+1, N):
                        if y in in_set:
                            continue
                        f_y = FRAC_DENOMS[y]
                        new_sum = sum_after_x + f_y
                        if new_sum >= ONE:
                            continue
                        new_def = ONE - new_sum
                        if new_def < best_deficit:
                            best_idx.remove(i)
                            best_idx.remove(j)
                            best_idx.append(x)
                            best_idx.append(y)
                            in_set.remove(i)
                            in_set.remove(j)
                            in_set.add(x)
                            in_set.add(y)
                            best_sum = new_sum
                            best_deficit = new_def
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_idx, best_deficit

def _triple_swap_improve(best_idx, best_deficit, deadline):
    """Try triple swaps (remove three, add three) for even larger jumps."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    k = len(best_idx)
    
    if k < 3:
        return best_idx, best_deficit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_remove = len(best_idx)
        for a in range(n_remove):
            if time.time() >= deadline:
                break
            for b in range(a+1, n_remove):
                if time.time() >= deadline:
                    break
                for c in range(b+1, n_remove):
                    i, j, m = best_idx[a], best_idx[b], best_idx[c]
                    sum_without_three = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j] - FRAC_DENOMS[m]
                    for x in range(N):
                        if x in in_set:
                            continue
                        f_x = FRAC_DENOMS[x]
                        sum_after_x = sum_without_three + f_x
                        if sum_after_x >= ONE:
                            continue
                        for y in range(x+1, N):
                            if y in in_set:
                                continue
                            f_y = FRAC_DENOMS[y]
                            sum_after_xy = sum_after_x + f_y
                            if sum_after_xy >= ONE:
                                continue
                            for z in range(y+1, N):
                                if z in in_set:
                                    continue
                                f_z = FRAC_DENOMS[z]
                                new_sum = sum_after_xy + f_z
                                if new_sum >= ONE:
                                    continue
                                new_def = ONE - new_sum
                                if new_def < best_deficit:
                                    best_idx.remove(i)
                                    best_idx.remove(j)
                                    best_idx.remove(m)
                                    best_idx.append(x)
                                    best_idx.append(y)
                                    best_idx.append(z)
                                    in_set.remove(i)
                                    in_set.remove(j)
                                    in_set.remove(m)
                                    in_set.add(x)
                                    in_set.add(y)
                                    in_set.add(z)
                                    best_sum = new_sum
                                    best_deficit = new_def
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_idx, best_deficit

def _simulated_annealing(best_idx, best_deficit, deadline, rng):
    """Simulated annealing to escape local optima."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    current_idx = list(best_idx)
    current_deficit = best_deficit
    current_sum = best_sum
    current_in_set = set(current_idx)
    
    # Temperature schedule
    t0 = 0.1
    cooling = 0.95
    min_temp = 1e-10
    
    temp = t0
    steps = 0
    
    while temp > min_temp and time.time() < deadline:
        # Generate a random move
        move_type = rng.randint(0, 2)  # 0: swap, 1: add, 2: remove
        
        if move_type == 0 and len(current_idx) > 0:
            # 1-swap
            i = rng.choice(current_idx)
            candidates = [j for j in range(N) if j not in current_in_set]
            if not candidates:
                continue
            j = rng.choice(candidates)
            
            new_sum = current_sum - FRAC_DENOMS[i] + FRAC_DENOMS[j]
            if new_sum >= ONE:
                continue
            new_def = ONE - new_sum
            
            delta = new_def - current_deficit
            if delta < 0 or rng.random() < _boltzmann_factor(delta, temp):
                # Accept move
                current_idx.remove(i)
                current_idx.append(j)
                current_in_set.remove(i)
                current_in_set.add(j)
                current_sum = new_sum
                current_deficit = new_def
                
                if current_deficit < best_deficit:
                    best_idx = list(current_idx)
                    best_deficit = current_deficit
                    best_sum = current_sum
            
        elif move_type == 1:
            # 1-add
            candidates = [j for j in range(N) if j not in current_in_set]
            if not candidates:
                continue
            j = rng.choice(candidates)
            
            new_sum = current_sum + FRAC_DENOMS[j]
            if new_sum >= ONE:
                continue
            new_def = ONE - new_sum
            
            delta = new_def - current_deficit
            if delta < 0 or rng.random() < _boltzmann_factor(delta, temp):
                # Accept move
                current_idx.append(j)
                current_in_set.add(j)
                current_sum = new_sum
                current_deficit = new_def
                
                if current_deficit < best_deficit:
                    best_idx = list(current_idx)
                    best_deficit = current_deficit
                    best_sum = current_sum
            
        elif move_type == 2 and len(current_idx) > 1:
            # 1-remove
            i = rng.choice(current_idx)
            
            new_sum = current_sum - FRAC_DENOMS[i]
            new_def = ONE - new_sum
            
            delta = new_def - current_deficit
            if delta < 0 or rng.random() < _boltzmann_factor(delta, temp):
                # Accept move
                current_idx.remove(i)
                current_in_set.remove(i)
                current_sum = new_sum
                current_deficit = new_def
                
                if current_deficit < best_deficit:
                    best_idx = list(current_idx)
                    best_deficit = current_deficit
                    best_sum = current_sum
        
        temp *= cooling
        steps += 1
    
    return best_idx, best_deficit

def _boltzmann_factor(delta, temperature):
    """Compute Boltzmann factor for accepting worse moves."""
    if delta <= 0:
        return 1.0
    # Use a simplified acceptance probability
    return min(1.0, 0.5 ** (delta * 1000 / max(temperature, 1e-10)))

def _greedy_seed():
    """Greedy: pick fractions in order until sum would exceed 1."""
    selected, total = [], Fraction()
    for i, d in enumerate(DENOMINATORS):
        if total + Fraction(1, d) < ONE:
            selected.append(i)
            total += Fraction(1, d)
    return selected, total

def _random_seed(rng, target_size=None):
    """Random subset that stays below 1."""
    if target_size is None:
        target_size = rng.randint(3, 8)
    selected = []
    total = Fraction()
    candidates = list(range(N))
    rng.shuffle(candidates)
    for idx in candidates:
        if len(selected) >= target_size:
            break
        f = FRAC_DENOMS[idx]
        if total + f < ONE:
            selected.append(idx)
            total += f
    if not selected:
        # Fallback: just pick a few small ones
        selected = [0]  # 1/2
        total = Fraction(1, 2)
    return selected, total

def _known_good_seeds():
    """Return a list of known good starting configurations."""
    seeds = []
    
    # {2,3,7,43} -> indices {0,1,5,41}
    seed1 = [0, 1, 5, 41]
    sum1 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if sum1 < ONE:
        seeds.append((seed1, sum1))
    
    # {2,3,8,23} -> indices {0,1,6,21}
    seed2 = [0, 1, 6, 21]
    sum2 = Fraction(1,2) + Fraction(1,3) + Fraction(1,8) + Fraction(1,23)
    if sum2 < ONE:
        seeds.append((seed2, sum2))
    
    # {2,3,9,17,43} -> indices {0,1,7,15,41}
    seed3 = [0, 1, 7, 15, 41]
    sum3 = Fraction(1,2) + Fraction(1,3) + Fraction(1,9) + Fraction(1,17) + Fraction(1,43)
    if sum3 < ONE:
        seeds.append((seed3, sum3))
    
    # {2,3,10,14,27} -> indices {0,1,8,12,25}
    seed4 = [0, 1, 8, 12, 25]
    sum4 = Fraction(1,2) + Fraction(1,3) + Fraction(1,10) + Fraction(1,14) + Fraction(1,27)
    if sum4 < ONE:
        seeds.append((seed4, sum4))
    
    return seeds

def construct():
    deadline = time.time() + 50  # Leave 10s safety margin
    rng = random.Random(42)
    
    # Start with greedy seed
    best_idx, best_sum = _greedy_seed()
    best_deficit = ONE - best_sum
    
    # Try known good seeds
    for seed_idx, seed_sum in _known_good_seeds():
        if seed_sum < ONE:
            seed_def = ONE - seed_sum
            if seed_def < best_deficit:
                best_idx = list(seed_idx)
                best_sum = seed_sum
                best_deficit = seed_def
    
    # Local search on initial seed
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _triple_swap_improve(best_idx, best_deficit, deadline)
    
    # Simulated annealing to escape local optima
    if time.time() < deadline - 5:
        best_idx, best_deficit = _simulated_annealing(best_idx, best_deficit, deadline, rng)
        best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
        best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    
    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        # Generate random seed with varying sizes
        target_size = rng.randint(3, 10)
        seed_idx, seed_sum = _random_seed(rng, target_size)
        
        # Local search
        idx, deficit = _try_improve(seed_idx, ONE - seed_sum, deadline)
        idx, deficit = _pairwise_swap_improve(idx, deficit, deadline)
        
        # Occasionally try triple swap and simulated annealing
        if restart_count % 3 == 0 and time.time() < deadline - 3:
            idx, deficit = _triple_swap_improve(idx, deficit, deadline)
        
        if restart_count % 5 == 0 and time.time() < deadline - 3:
            idx, deficit = _simulated_annealing(idx, deficit, deadline, rng)
            idx, deficit = _try_improve(idx, deficit, deadline)
        
        if deficit < best_deficit:
            best_idx = idx
            best_deficit = deficit
    
    # Final refinement
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    if time.time() < deadline - 1:
        best_idx, best_deficit = _triple_swap_improve(best_idx, best_deficit, deadline)
    
    return best_idx
# EVOLVE-BLOCK-END

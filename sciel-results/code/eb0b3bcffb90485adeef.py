# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)

def _local_search(best_idx, best_deficit, deadline):
    """Fast hill climbing: 1-swap, 1-add, 2-swap."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # 1. Try 1-swap
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
        
        # 2. Try 1-add
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
        
        # 3. Try 2-swap (remove 2, add 2) - simplified version for speed
        if not changed:
            k = len(best_idx)
            if k >= 2:
                # Try swapping two elements with two others
                # To keep it fast, we only try a limited number of pairs
                # Or just iterate through all pairs if time allows
                improved = False
                for a in range(k):
                    if time.time() >= deadline:
                        break
                    for b in range(a+1, k):
                        i, j = best_idx[a], best_idx[b]
                        sum_without_two = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j]
                        # Find best pair (x,y) to add
                        best_pair_def = best_deficit
                        best_x, best_y = -1, -1
                        for x in range(N):
                            if x in in_set:
                                continue
                            f_x = FRAC_DENOMS[x]
                            sum_x = sum_without_two + f_x
                            if sum_x >= ONE:
                                continue
                            for y in range(x+1, N):
                                if y in in_set:
                                    continue
                                f_y = FRAC_DENOMS[y]
                                new_sum = sum_x + f_y
                                if new_sum >= ONE:
                                    continue
                                new_def = ONE - new_sum
                                if new_def < best_pair_def:
                                    best_pair_def = new_def
                                    best_x, best_y = x, y
                        if best_x != -1:
                            best_idx.remove(i)
                            best_idx.remove(j)
                            best_idx.append(best_x)
                            best_idx.append(best_y)
                            in_set.remove(i)
                            in_set.remove(j)
                            in_set.add(best_x)
                            in_set.add(best_y)
                            best_sum = ONE - best_pair_def
                            best_deficit = best_pair_def
                            changed = True
                            improved = True
                            break
                    if improved:
                        break

    return best_idx, best_deficit

def _simulated_annealing(initial_idx, initial_deficit, deadline, rng):
    """Simulated Annealing to escape local optima."""
    current_idx = list(initial_idx)
    current_deficit = initial_deficit
    current_sum = ONE - current_deficit
    in_set = set(current_idx)
    
    best_idx = list(current_idx)
    best_deficit = current_deficit
    best_sum = current_sum
    
    T = 1.0 / 1000.0  # Initial temperature (in deficit units)
    T_min = 1.0 / 1e9
    cooling = 0.99
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline and T > T_min and steps < max_steps:
        steps += 1
        
        # Generate a random move
        move_type = rng.choice(['swap', 'add', 'remove', 'double_swap'])
        new_idx = list(current_idx)
        new_in_set = set(current_in_set) if False else set(current_idx) # Fix: use current_in_set
        new_sum = current_sum
        
        valid_move = False
        
        if move_type == 'swap' and len(current_idx) > 0:
            i = rng.choice(current_idx)
            candidates = [j for j in range(N) if j not in current_in_set]
            if candidates:
                j = rng.choice(candidates)
                new_sum = current_sum - FRAC_DENOMS[i] + FRAC_DENOMS[j]
                if new_sum < ONE:
                    new_idx.remove(i)
                    new_idx.append(j)
                    new_in_set.remove(i)
                    new_in_set.add(j)
                    valid_move = True
        
        elif move_type == 'add':
            candidates = [j for j in range(N) if j not in current_in_set]
            if candidates:
                j = rng.choice(candidates)
                new_sum = current_sum + FRAC_DENOMS[j]
                if new_sum < ONE:
                    new_idx.append(j)
                    new_in_set.add(j)
                    valid_move = True
        
        elif move_type == 'remove' and len(current_idx) > 1:
            i = rng.choice(current_idx)
            new_sum = current_sum - FRAC_DENOMS[i]
            new_idx.remove(i)
            new_in_set.remove(i)
            valid_move = True
            
        elif move_type == 'double_swap' and len(current_idx) >= 2:
            i, j = rng.sample(current_idx, 2)
            candidates = [k for k in range(N) if k not in current_in_set]
            if len(candidates) >= 2:
                x, y = rng.sample(candidates, 2)
                new_sum = current_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j] + FRAC_DENOMS[x] + FRAC_DENOMS[y]
                if new_sum < ONE:
                    new_idx.remove(i)
                    new_idx.remove(j)
                    new_idx.append(x)
                    new_idx.append(y)
                    new_in_set.remove(i)
                    new_in_set.remove(j)
                    new_in_set.add(x)
                    new_in_set.add(y)
                    valid_move = True

        if not valid_move:
            continue

        new_deficit = ONE - new_sum
        delta_e = new_deficit - current_deficit
        
        # Acceptance probability
        if delta_e < 0:
            accept = True
        else:
            # Use float for approximation in SA
            delta_f = float(delta_e)
            T_f = float(T)
            if T_f > 0:
                prob = math.exp(-delta_f / T_f)
                accept = rng.random() < prob
            else:
                accept = False

        if accept:
            current_idx = new_idx
            current_in_set = new_in_set
            current_sum = new_sum
            current_deficit = new_deficit
            
            if new_deficit < best_deficit:
                best_idx = list(new_idx)
                best_deficit = new_deficit
                best_sum = new_sum
        
        T *= cooling

    return best_idx, best_deficit

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
        selected = [0]  # 1/2
        total = Fraction(1, 2)
    return selected, total

def construct():
    deadline = time.time() + 50  # Leave 10s safety margin
    rng = random.Random(42)
    
    # 1. Generate a few diverse seeds
    seeds = []
    
    # Greedy seed
    g_idx, g_sum = _greedy_seed()
    seeds.append((g_idx, ONE - g_sum))
    
    # Known good seed
    good_seed = [0, 1, 5, 41] # 1/2, 1/3, 1/7, 1/43
    good_sum = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if good_sum < ONE:
        seeds.append((good_seed, ONE - good_sum))
        
    # Random seeds
    for _ in range(5):
        r_idx, r_sum = _random_seed(rng, rng.randint(4, 10))
        seeds.append((r_idx, ONE - r_sum))

    best_idx = seeds[0][0]
    best_deficit = seeds[0][1]
    
    # 2. For each seed, run SA then Local Search
    time_per_seed = 8.0
    
    for idx, deficit in seeds:
        if time.time() > deadline - 10:
            break
            
        sa_deadline = time.time() + time_per_seed
        if sa_deadline > deadline:
            sa_deadline = deadline
            
        # Run Simulated Annealing
        sa_idx, sa_deficit = _simulated_annealing(idx, deficit, sa_deadline, rng)
        
        # Run Local Search on the result
        ls_deadline = time.time() + (deadline - time.time()) / 2
        if ls_deadline > deadline:
            ls_deadline = deadline
            
        ls_idx, ls_deficit = _local_search(sa_idx, sa_deficit, ls_deadline)
        
        if ls_deficit < best_deficit:
            best_idx = ls_idx
            best_deficit = ls_deficit

    # 3. Final polish on the best found
    if time.time() < deadline - 2:
        final_idx, final_deficit = _local_search(best_idx, best_deficit, deadline)
        if final_deficit < best_deficit:
            best_idx = final_idx
            best_deficit = final_deficit

    return best_idx
# EVOLVE-BLOCK-END

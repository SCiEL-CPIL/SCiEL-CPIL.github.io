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
        
        # Try 1-remove: remove i if it improves deficit (only helps if sum was close to 1)
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

def construct():
    deadline = time.time() + 50  # Leave 10s safety margin
    rng = random.Random(42)
    
    # Start with greedy seed
    best_idx, best_sum = _greedy_seed()
    best_deficit = ONE - best_sum
    
    # Try known good starting point: {2,3,7,43} -> indices {0,1,5,41}
    good_seed = [0, 1, 5, 41]
    good_sum = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if good_sum < ONE:
        good_def = ONE - good_sum
        if good_def < best_deficit:
            best_idx = list(good_seed)
            best_sum = good_sum
            best_deficit = good_def
    
    # Local search on initial seed
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
        
        if deficit < best_deficit:
            best_idx = idx
            best_deficit = deficit
    
    # Final refinement
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    
    return best_idx
# EVOLVE-BLOCK-END

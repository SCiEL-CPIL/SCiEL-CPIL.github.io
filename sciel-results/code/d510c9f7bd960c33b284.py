"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)

# Compute LCM of all denominators 2..48
def _lcm(a, b):
    return a // gcd(a, b) * b

LCM = reduce(_lcm, DENOMS)
# Integer weight for each denominator: LCM // d
WEIGHTS = [LCM // d for d in DENOMS]
ONE = LCM  # represents 1 in integer space

def _sum_weights(indices):
    """Return sum of weights for given indices."""
    return sum(WEIGHTS[i] for i in indices)

def _deficit_int(indices):
    """Return deficit as integer: LCM - sum(weights)."""
    return ONE - _sum_weights(indices)

def _greedy_construct(deadline):
    """Greedy: add largest reciprocal that keeps sum < 1."""
    selected = set()
    current_sum = 0
    order = sorted(range(N), key=lambda i: -WEIGHTS[i])
    for i in order:
        if time.time() > deadline:
            break
        if i in selected:
            continue
        new_sum = current_sum + WEIGHTS[i]
        if new_sum < ONE:
            selected.add(i)
            current_sum = new_sum
    if not selected:
        selected = {0}
    return list(selected)

def _local_search_1swap(indices, deadline):
    """Local search with 1-for-1 swaps using integer arithmetic."""
    selected = set(indices)
    current_sum = _sum_weights(selected)
    best_sum = current_sum
    best_set = set(selected)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Try all removals
        for j in list(selected):
            if time.time() > deadline:
                break
            sum_without_j = current_sum - WEIGHTS[j]
            for i in range(N):
                if i in selected:
                    continue
                new_sum = sum_without_j + WEIGHTS[i]
                if new_sum < ONE and new_sum > best_sum:
                    best_sum = new_sum
                    best_set = (selected - {j}) | {i}
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        selected = set(best_set)
        current_sum = best_sum
    
    return list(best_set)

def _local_search_1for2(indices, deadline):
    """Local search with 1-for-2 swaps (remove one, add two)."""
    selected = set(indices)
    current_sum = _sum_weights(selected)
    best_sum = current_sum
    best_set = set(selected)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for j in list(selected):
            if time.time() > deadline:
                break
            sum_without_j = current_sum - WEIGHTS[j]
            # Try all pairs (i1, i2) not in selected
            unselected = [i for i in range(N) if i not in selected]
            for a in range(len(unselected)):
                if time.time() > deadline:
                    break
                i1 = unselected[a]
                for b in range(a + 1, len(unselected)):
                    i2 = unselected[b]
                    new_sum = sum_without_j + WEIGHTS[i1] + WEIGHTS[i2]
                    if new_sum < ONE and new_sum > best_sum:
                        best_sum = new_sum
                        best_set = (selected - {j}) | {i1, i2}
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
        selected = set(best_set)
        current_sum = best_sum
    
    return list(best_set)

def _randomized_greedy(deadline):
    """Randomized greedy: shuffle order, then greedily add."""
    selected = set()
    current_sum = 0
    order = list(range(N))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            break
        if i in selected:
            continue
        new_sum = current_sum + WEIGHTS[i]
        if new_sum < ONE:
            selected.add(i)
            current_sum = new_sum
    if not selected:
        selected = {0}
    return list(selected)

def _simulated_annealing(indices, deadline):
    """Simulated annealing on the integer weight space."""
    selected = set(indices)
    current_sum = _sum_weights(selected)
    best_sum = current_sum
    best_set = set(selected)
    
    temp = 0.1
    cooling = 0.995
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        # Random move: remove one and add one
        if selected:
            j = random.choice(list(selected))
            i = random.randrange(N)
            if i not in selected:
                new_sum = current_sum - WEIGHTS[j] + WEIGHTS[i]
                if new_sum < ONE:
                    delta = new_sum - current_sum
                    if delta > 0 or random.random() < temp:
                        # Accept
                        selected = (selected - {j}) | {i}
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(selected)
        temp *= cooling
    
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Phase 1: Greedy construct
    inds = _greedy_construct(deadline)
    best_deficit = _deficit_int(inds)
    best_indices = inds
    
    # Phase 2: Local search 1-for-1
    if time.time() < deadline:
        inds2 = _local_search_1swap(inds, deadline)
        d2 = _deficit_int(inds2)
        if d2 < best_deficit:
            best_deficit = d2
            best_indices = inds2
    
    # Phase 3: Local search 1-for-2
    if time.time() < deadline:
        inds3 = _local_search_1for2(best_indices, deadline)
        d3 = _deficit_int(inds3)
        if d3 < best_deficit:
            best_deficit = d3
            best_indices = inds3
    
    # Phase 4: Multiple randomized restarts with local search
    while time.time() < deadline - 5.0:
        inds_r = _randomized_greedy(deadline)
        # Apply 1-for-1 local search
        inds_ls = _local_search_1swap(inds_r, deadline)
        d_ls = _deficit_int(inds_ls)
        if d_ls < best_deficit:
            best_deficit = d_ls
            best_indices = inds_ls
        
        # Apply 1-for-2 local search
        if time.time() < deadline - 5.0:
            inds_12 = _local_search_1for2(inds_ls, deadline)
            d_12 = _deficit_int(inds_12)
            if d_12 < best_deficit:
                best_deficit = d_12
                best_indices = inds_12
        
        # Simulated annealing
        if time.time() < deadline - 5.0:
            inds_sa = _simulated_annealing(best_indices, deadline)
            d_sa = _deficit_int(inds_sa)
            if d_sa < best_deficit:
                best_deficit = d_sa
                best_indices = inds_sa
    
    # Final verification with Fraction
    exact = Fraction(0, 1)
    for i in best_indices:
        exact += Fraction(1, DENOMS[i])
    if exact >= 1:
        # Fallback: return a known valid subset
        best_indices = [0]
    
    return best_indices
# EVOLVE-BLOCK-END

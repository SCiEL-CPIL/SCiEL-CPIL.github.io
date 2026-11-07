"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt
from itertools import combinations

NMAX = 256

def _valid_addition(x, S_list):
    """Check if x can be added to set S_list without violating the constraint.
    For all distinct a,b,c in final set, a does not divide b*c.
    When adding x, check triples involving x:
    - (x, b, c): x | b*c?
    - (b, x, c): b | x*c?
    - (c, x, b): c | x*b?
    for all b,c in S_list, b != c.
    """
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy_largest_first(deadline):
    """Greedy: try numbers from largest to smallest."""
    chosen = []
    chosen_set = set()
    
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_smallest_first(deadline):
    """Greedy: try numbers from smallest to largest."""
    chosen = []
    chosen_set = set()
    
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_randomized(deadline, seed=42):
    """Greedy with randomized order."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    
    chosen = []
    chosen_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_weighted(deadline, seed=42, alpha=0.5):
    """Greedy with weighted randomization favoring larger numbers."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    # Weight proportional to (x/NMAX)^alpha, favoring larger
    weights = [(x / NMAX) ** alpha for x in candidates]
    
    chosen = []
    chosen_set = set()
    remaining = candidates[:]
    remaining_weights = weights[:]
    
    while remaining:
        if time.time() > deadline:
            break
        # Weighted random choice
        total_w = sum(remaining_weights)
        r = rng.random() * total_w
        cumulative = 0
        idx = 0
        for k in range(len(remaining)):
            cumulative += remaining_weights[k]
            if r <= cumulative:
                idx = k
                break
        x = remaining[idx]
        remaining.pop(idx)
        remaining_weights.pop(idx)
        
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _local_search(S, deadline, max_iters=2000):
    """Local search: try adding elements, removing elements, swapping."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(99999)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        if len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                # Try a few random candidates
                rng.shuffle(candidates)
                added = False
                for x in candidates[:20]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        
        # Try swapping: remove one, add another
        if S:
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:40]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
            else:
                S = S_temp + [removed]
                S_set = set(S)
    
    return best

def _two_opt_search(S, deadline, max_iters=500):
    """Try removing pairs and adding pairs to improve."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(77777)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        if len(S) < 2:
            break
        
        # Try removing two elements
        i, j = rng.sample(range(len(S)), 2)
        i, j = min(i, j), max(i, j)
        removed = [S[i], S[j]]
        S_temp = S[:i] + S[i+1:j] + S[j+1:]
        S_temp_set = set(S_temp)
        
        candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
        if not candidates:
            continue
        
        # Try adding two elements
        rng.shuffle(candidates)
        for x in candidates[:30]:
            if not _valid_addition(x, S_temp):
                continue
            S_temp2 = S_temp + [x]
            S_temp2_set = S_temp_set | {x}
            for y in candidates[:30]:
                if y == x or y in S_temp2_set:
                    continue
                if _valid_addition(y, S_temp2):
                    S = S_temp2 + [y]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    break
            else:
                continue
            break
        else:
            # No improvement, restore
            continue
    
    return best

def _simulated_annealing(S, deadline, max_iters=3000, T_start=100.0, T_end=1.0):
    """Simulated annealing for optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(55555)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        # Generate a neighbor
        op = rng.random()
        if op < 0.3 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        elif op < 0.6 and S:
            # Remove
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            # Try to re-add something
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    # Restore
                    S.append(removed)
                    S_set.add(removed)
        elif S:
            # Swap
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    # Accept with probability exp(-delta/T) where delta = -1 (worse)
                    if current_temp > 0.1 and rng.random() < 0.1:
                        S.append(x)
                        S_set.add(x)
                    else:
                        S.append(removed)
                        S_set.add(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best = None
    best_size = 0
    
    # Phase 1: Multiple greedy initializations
    greedy_results = []
    
    # Largest first
    if time.time() < deadline - 60:
        c = _greedy_largest_first(deadline - 60)
        greedy_results.append(c)
    
    # Smallest first
    if time.time() < deadline - 55:
        c = _greedy_smallest_first(deadline - 55)
        greedy_results.append(c)
    
    # Randomized with different seeds
    for seed in [1, 7, 13, 42, 99, 128, 200, 256]:
        if time.time() > deadline - 50:
            break
        c = _greedy_randomized(deadline - 50, seed=seed)
        greedy_results.append(c)
    
    # Weighted randomization
    for seed in [5, 15, 25]:
        if time.time() > deadline - 45:
            break
        for alpha in [0.3, 0.7, 1.0]:
            if time.time() > deadline - 45:
                break
            c = _greedy_weighted(deadline - 45, seed=seed, alpha=alpha)
            greedy_results.append(c)
    
    # Find best greedy result
    for c in greedy_results:
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    if best is None:
        best = list(range(129, 257))  # Fallback
        best_size = len(best)
    
    # Phase 2: Local search on best greedy result
    if time.time() < deadline - 20:
        c = _local_search(best, deadline - 20, max_iters=3000)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Phase 3: Two-opt search
    if time.time() < deadline - 15:
        c = _two_opt_search(best, deadline - 15, max_iters=500)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Phase 4: Simulated annealing
    if time.time() < deadline - 10:
        c = _simulated_annealing(best, deadline - 10, max_iters=5000)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Phase 5: More local search from perturbed states
    rng = random.Random(31337)
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Perturb the current best
        S_perturbed = best[:]
        n_remove = rng.randint(1, min(4, len(S_perturbed)))
        indices = rng.sample(range(len(S_perturbed)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            S_perturbed.pop(idx)
        
        c = _local_search(S_perturbed, deadline - 5, max_iters=500)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Phase 6: Final local search on best
    if time.time() < deadline - 2:
        c = _local_search(best, deadline - 2, max_iters=2000)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]

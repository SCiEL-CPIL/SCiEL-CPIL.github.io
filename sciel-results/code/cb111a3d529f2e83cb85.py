"""Combined program: structural B&B from inspiration 1, refinement from current program."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction
from itertools import combinations

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    """
    Find distinct pairwise coprime integers a in 1..511 maximizing sum 1/(512 - a).
    Uses greedy initialization, branch-and-bound on reduced problem, and intensive
    local search with bitset-accelerated conflict checks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    N = 512
    n_vars = N - 1  # variables 1..511, index 0 unused
    
    # Precompute weights: weight of vertex a is 1/(512 - a)
    weights_float = [0.0] * N  # index 0 unused, 1..511
    for a in range(1, N):
        weights_float[a] = 1.0 / (N - a)
    
    # Precompute prime factors for all numbers
    pfactors = [set() for _ in range(N)]
    for a in range(2, N):
        pfactors[a] = _prime_factors(a)
    
    # Build conflict bitsets for fast checks
    conflict_bitsets = [0] * N  # bitmask: bit i set means a conflicts with i+1
    for a in range(2, N):
        mask = 0
        for b in range(2, N):
            if b != a and len(pfactors[a] & pfactors[b]) > 0:
                mask |= (1 << b)
        conflict_bitsets[a] = mask
    
    def is_coprime(a, b):
        return len(pfactors[a] & pfactors[b]) == 0
    
    def can_add_bitset(candidate, current_set):
        """Check if candidate can be added to current_set using bitset."""
        mask = 0
        for x in current_set:
            mask |= (1 << x)
        return (conflict_bitsets[candidate] & mask) == 0
    
    def compute_score_float(current_set):
        return sum(weights_float[a] for a in current_set)
    
    # Step 1: Multiple greedy initializations with different orderings
    best_set = []
    best_score_float = 0.0
    
    # Ordering 1: Pure descending weight (original)
    greedy_set1 = []
    for a in range(N - 1, 0, -1):
        if time.time() > deadline - 100:
            break
        if can_add_bitset(a, greedy_set1):
            greedy_set1.append(a)
    
    score1 = compute_score_float(greedy_set1)
    if score1 > best_score_float:
        best_set = list(greedy_set1)
        best_score_float = score1
    
    # Ordering 2: Descending weight but skip multiples of small primes first
    # Try a few randomized perturbations
    random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 90:
            break
        # Perturb: randomly swap some adjacent elements in the ordering
        order = list(range(N - 1, 0, -1))
        for _ in range(20):
            i = random.randint(0, len(order) - 2)
            if random.random() < 0.5:
                order[i], order[i+1] = order[i+1], order[i]
        
        greedy_set_t = []
        for a in order:
            if can_add_bitset(a, greedy_set_t):
                greedy_set_t.append(a)
        
        score_t = compute_score_float(greedy_set_t)
        if score_t > best_score_float:
            best_set = list(greedy_set_t)
            best_score_float = score_t
    
    # Also try: prioritize numbers with fewer conflicts (more flexible)
    if time.time() < deadline - 80:
        conflict_counts = [bin(conflict_bitsets[a]).count('1') for a in range(N)]
        # Sort by weight desc, but among same weight tier, prefer fewer conflicts
        order3 = sorted(range(1, N), key=lambda x: (-weights_float[x], conflict_counts[x]))
        greedy_set3 = []
        for a in order3:
            if can_add_bitset(a, greedy_set3):
                greedy_set3.append(a)
        score3 = compute_score_float(greedy_set3)
        if score3 > best_score_float:
            best_set = list(greedy_set3)
            best_score_float = score3
    
    # Step 2: Intensive local search refinement
    all_candidates = sorted(range(1, N), key=lambda x: -weights_float[x])
    best_set_set = set(best_set)
    
    def local_search_iteration(current_set, current_set_set, current_score_float):
        """Perform one pass of local search: additions, 1-for-1, 1-for-2, 2-for-1."""
        improved = False
        
        # Single additions (try all candidates not in set)
        for a in all_candidates:
            if time.time() > deadline - 20:
                break
            if a not in current_set_set and can_add_bitset(a, current_set):
                current_set.append(a)
                current_set_set.add(a)
                current_score_float += weights_float[a]
                improved = True
        
        # 1-for-1 swaps
        for i, remove_a in enumerate(list(current_set)):
            if time.time() > deadline - 20:
                break
            for add_b in all_candidates:
                if time.time() > deadline - 20:
                    break
                if add_b == remove_a or add_b in current_set_set:
                    continue
                # Check if add_b is compatible with all other elements
                temp_set = [x for j, x in enumerate(current_set) if j != i]
                if can_add_bitset(add_b, temp_set):
                    if weights_float[add_b] > weights_float[remove_a] + 1e-12:
                        current_set[i] = add_b
                        current_set_set.remove(remove_a)
                        current_set_set.add(add_b)
                        current_score_float += weights_float[add_b] - weights_float[remove_a]
                        improved = True
                        break
        
        # 1-for-2 swaps: remove one, add two
        if time.time() < deadline - 30:
            for i, remove_a in enumerate(list(current_set)):
                if time.time() > deadline - 30:
                    break
                temp_set = [x for j, x in enumerate(current_set) if j != i]
                temp_set_set = set(temp_set)
                
                valid_cands = []
                for b in all_candidates:
                    if b not in temp_set_set and b != remove_a:
                        if can_add_bitset(b, temp_set):
                            valid_cands.append(b)
                
                # Try pairs from valid_cands
                for idx1 in range(len(valid_cands)):
                    if time.time() > deadline - 30:
                        break
                    for idx2 in range(idx1 + 1, len(valid_cands)):
                        b1, b2 = valid_cands[idx1], valid_cands[idx2]
                        # Check b1 and b2 are coprime
                        if len(pfactors[b1] & pfactors[b2]) == 0:
                            if weights_float[b1] + weights_float[b2] > weights_float[remove_a] + 1e-12:
                                current_set[i] = b1
                                current_set.append(b2)
                                current_set_set.remove(remove_a)
                                current_set_set.add(b1)
                                current_set_set.add(b2)
                                current_score_float += weights_float[b1] + weights_float[b2] - weights_float[remove_a]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        # 2-for-1 swaps: remove two, add one
        if time.time() < deadline - 30:
            n_set = len(current_set)
            for i in range(n_set):
                if time.time() > deadline - 30:
                    break
                for j in range(i + 1, n_set):
                    if time.time() > deadline - 30:
                        break
                    remove_a, remove_b = current_set[i], current_set[j]
                    temp_set = [x for k, x in enumerate(current_set) if k != i and k != j]
                    temp_set_set = set(temp_set)
                    
                    for add_c in all_candidates:
                        if time.time() > deadline - 30:
                            break
                        if add_c in temp_set_set or add_c == remove_a or add_c == remove_b:
                            continue
                        if can_add_bitset(add_c, temp_set):
                            if weights_float[add_c] > weights_float[remove_a] + weights_float[remove_b] + 1e-12:
                                # Replace i with add_c, remove j
                                current_set[i] = add_c
                                current_set.pop(j)
                                current_set_set.remove(remove_a)
                                current_set_set.remove(remove_b)
                                current_set_set.add(add_c)
                                current_score_float += weights_float[add_c] - weights_float[remove_a] - weights_float[remove_b]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return current_set, current_set_set, current_score_float, improved
    
    # Run multiple passes until no improvement or time runs out
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - 25:
            break
        best_set, best_set_set, best_score_float, improved = local_search_iteration(
            best_set, best_set_set, best_score_float
        )
        if not improved:
            break
    
    # Step 3: Simulated annealing for escaping local optima (from inspiration 1)
    if time.time() < deadline - 15:
        S = list(best_set)
        S_set = set(S)
        current_score = best_score_float
        best_SA_set = list(S)
        best_SA_score = current_score
        
        T = 0.05
        T_min = 1e-8
        cooling = 0.999
        
        rng = random.Random(123)
        
        while time.time() < deadline - 15 and T > T_min:
            # Random swap: remove one, add one
            if S:
                x = rng.choice(S)
                new_S = list(S)
                new_S.remove(x)
                new_S_set = set(new_S)
                
                # Find best vertex to add
                best_add = None
                best_add_w = -1
                for a in all_candidates:
                    if a in new_S_set:
                        continue
                    if can_add_bitset(a, new_S):
                        if weights_float[a] > best_add_w:
                            best_add_w = weights_float[a]
                            best_add = a
                
                if best_add is not None:
                    new_S.append(best_add)
                    new_S_set.add(best_add)
                    delta = weights_float[best_add] - weights_float[x]
                    
                    if delta > 0 or rng.random() < math.exp(delta / T):
                        S = new_S
                        S_set = new_S_set
                        current_score += delta
                        if current_score > best_SA_score:
                            best_SA_score = current_score
                            best_SA_set = list(S)
                            if best_SA_score > best_score_float:
                                best_score_float = best_SA_score
                                best_set = list(S)
                                best_set_set = S_set
        
        # Try 2-swap moves: remove two, add two
        if time.time() < deadline - 10:
            for _ in range(30):
                if time.time() > deadline - 10:
                    break
                
                if len(S) >= 2:
                    x, y = rng.sample(S, 2)
                    new_S = list(S)
                    new_S.remove(x)
                    new_S.remove(y)
                    new_S_set = set(new_S)
                    
                    # Find best pair to add
                    candidates = []
                    for a in all_candidates:
                        if a in new_S_set:
                            continue
                        if can_add_bitset(a, new_S):
                            candidates.append(a)
                    
                    if len(candidates) >= 2:
                        # Sort by weight descending
                        candidates.sort(key=lambda a: -weights_float[a])
                        # Try top few pairs
                        for i in range(min(10, len(candidates))):
                            for j in range(i + 1, min(10, len(candidates))):
                                a1 = candidates[i]
                                a2 = candidates[j]
                                if (conflict_bitsets[a1] >> a2) & 1:
                                    continue
                                # Check a1 and a2 don't conflict with new_S
                                if can_add_bitset(a1, new_S) and can_add_bitset(a2, new_S + [a1]):
                                    delta = weights_float[a1] + weights_float[a2] - weights_float[x] - weights_float[y]
                                    if delta > 0:
                                        new_S.append(a1)
                                        new_S.append(a2)
                                        new_S_set.add(a1)
                                        new_S_set.add(a2)
                                        current_score += delta
                                        S = new_S
                                        S_set = new_S_set
                                        if current_score > best_SA_score:
                                            best_SA_score = current_score
                                            best_SA_set = list(S)
                                            if best_SA_score > best_score_float:
                                                best_score_float = best_SA_score
                                                best_set = list(S)
                                                best_set_set = S_set
                                        break
                            else:
                                continue
                            break
    
    # Step 4: Targeted high-value element insertion
    # For each high-weight element not in best_set, try to make room by removing minimal blockers
    if time.time() < deadline - 5:
        sorted_by_weight = sorted(range(1, N), key=lambda x: -weights_float[x])
        best_set_set = set(best_set)
        
        for target in sorted_by_weight:
            if time.time() > deadline - 5:
                break
            if target in best_set_set:
                continue
            
            # Find all elements in best_set that conflict with target
            blockers = [x for x in best_set if (conflict_bitsets[target] >> x) & 1]
            
            # If no blockers, just add it
            if not blockers:
                best_set.append(target)
                best_set_set.add(target)
                best_score_float += weights_float[target]
                continue
            
            # Check if removing all blockers and adding target improves score
            blocker_score = sum(weights_float[b] for b in blockers)
            if weights_float[target] > blocker_score + 1e-12:
                for b in blockers:
                    best_set.remove(b)
                    best_set_set.remove(b)
                    best_score_float -= weights_float[b]
                best_set.append(target)
                best_set_set.add(target)
                best_score_float += weights_float[target]
                continue
            
            # Try removing subsets of blockers (up to size 2)
            if len(blockers) <= 2:
                for k in range(1, len(blockers) + 1):
                    for combo in _subsets(blockers, k):
                        combo_set = set(combo)
                        other_blockers = [x for x in best_set if x not in combo_set and (conflict_bitsets[target] >> x) & 1]
                        if not other_blockers:
                            combo_score = sum(weights_float[b] for b in combo)
                            if weights_float[target] > combo_score + 1e-12:
                                for b in combo:
                                    best_set.remove(b)
                                    best_set_set.remove(b)
                                    best_score_float -= weights_float[b]
                                best_set.append(target)
                                best_set_set.add(target)
                                best_score_float += weights_float[target]
                                break
                    if target in best_set_set:
                        break
    
    # Final validation
    final_set = list(best_set)
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if not is_coprime(final_set[i], final_set[j]):
                # Fallback to greedy_set1
                return greedy_set1
    
    return final_set

def _subsets(lst, k):
    """Generate all subsets of lst of size k."""
    if k == 0:
        yield []
        return
    if k > len(lst):
        return
    for i in range(len(lst)):
        for rest in _subsets(lst[i+1:], k-1):
            yield [lst[i]] + rest

# EVOLVE-BLOCK-END

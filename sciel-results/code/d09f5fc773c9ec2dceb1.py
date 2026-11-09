"""Reciprocal weight maximization for 3-AP-free subsets of [1,256]."""
import time
import random
from fractions import Fraction

N = 256
DEADLINE_MARGIN = 8  # seconds to leave before the 90s kill

def _precompute_conflicts():
    """Precompute for each x the set of pairs (a,c) that would form a 3-AP with x."""
    # conflicts[x] = set of (a, c) such that {a, x, c} or {x, a, c} or {a, c, x} is a 3-AP
    # Actually, for swap validity we need: if we add x, which existing elements conflict?
    # x conflicts with existing set S if there exist a, c in S such that {a, x, c} is a 3-AP.
    # So for each x, precompute all pairs (a, c) that would complete a 3-AP with x.
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x + 1) // 2 + 1):
            a = x
            b = x + d
            c = x + 2 * d
            if c <= N:
                conflicts[x].add((b, c))
        # x as middle: x-d, x, x+d
        for d in range(1, x):
            a = x - d
            b = x
            c = x + d
            if a >= 1 and c <= N:
                conflicts[x].add((a, c))
        # x as last: x-2d, x-d, x
        for d in range(1, (x + 1) // 2):
            a = x - 2 * d
            b = x - d
            c = x
            if a >= 1:
                conflicts[x].add((a, b))
    return conflicts

CONFLICTS = _precompute_conflicts()

def _swap_valid_fast(bits, x_remove, x_add):
    """Fast swap validity using precomputed conflicts."""
    # Check if adding x_add creates any 3-AP with existing elements (excluding x_remove)
    for (a, c) in CONFLICTS[x_add]:
        if a == x_remove or c == x_remove:
            continue
        if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
            return False
    return True

def _reciprocal_sum_float(bits):
    """Compute floating sum of reciprocals."""
    total = 0.0
    for i in range(N):
        if (bits >> i) & 1:
            total += 1.0 / (i + 1)
    return total

def _greedy_construction(order):
    """Greedy construction given an ordering of elements to try."""
    bits = 0
    for x in order:
        # Check if x can be added
        valid = True
        for (a, c) in CONFLICTS[x]:
            if (bits >> (a - 1)) & 1 and (bits >> (c - 1)) & 1:
                valid = False
                break
        if valid:
            bits |= (1 << (x - 1))
    return bits

def _local_search(bits, score, deadline):
    """Iterative improvement via single swaps."""
    improved = True
    while improved:
        improved = False
        cur_list = [i + 1 for i in range(N) if (bits >> i) & 1]
        for x_remove in cur_list:
            if time.time() > deadline:
                return bits, score
            for x_add in range(1, N + 1):
                if x_add == x_remove or ((bits >> (x_add - 1)) & 1):
                    continue
                if not _swap_valid_fast(bits, x_remove, x_add):
                    continue
                new_score = score - 1.0 / x_remove + 1.0 / x_add
                if new_score > score:
                    bits = (bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                    score = new_score
                    improved = True
                    break
    return bits, score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Phase 1: Multiple greedy constructions with different orderings
    best_bits = 0
    best_score = 0.0
    
    # Ordering 1: Natural order (smallest first)
    order1 = list(range(1, N + 1))
    bits1 = _greedy_construction(order1)
    score1 = _reciprocal_sum_float(bits1)
    if score1 > best_score:
        best_score = score1
        best_bits = bits1
    
    # Ordering 2: Reverse order (largest first) - less weight but may allow more small elements
    order2 = list(range(N, 0, -1))
    bits2 = _greedy_construction(order2)
    score2 = _reciprocal_sum_float(bits2)
    if score2 > best_score:
        best_score = score2
        best_bits = bits2
    
    # Ordering 3: Weighted by 1/x (same as natural for this problem since 1/x decreases)
    # Try a few random perturbations of the natural order
    rng = random.Random(123)
    for trial in range(5):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        # Randomly swap some adjacent elements to create variations
        for _ in range(20):
            i = rng.randint(0, N - 2)
            order[i], order[i + 1] = order[i + 1], order[i]
        bits = _greedy_construction(order)
        score = _reciprocal_sum_float(bits)
        if score > best_score:
            best_score = score
            best_bits = bits
    
    # Phase 2: Local search on the best greedy result
    if time.time() < deadline:
        best_bits, best_score = _local_search(best_bits, best_score, deadline)
    
    # Phase 3: Simulated annealing for further improvement
    if time.time() < deadline:
        rng_sa = random.Random(42)
        current_bits = best_bits
        current_score = best_score
        best_bits_sa = best_bits
        best_score_sa = best_score
        
        T_start = 0.005
        T_end = 0.00005
        alpha = 0.997
        
        current_T = T_start
        while current_T > T_end:
            if time.time() > deadline:
                break
            
            cur_list = [i + 1 for i in range(N) if (current_bits >> i) & 1]
            if len(cur_list) < 2:
                break
            
            idx_remove = rng_sa.randint(0, len(cur_list) - 1)
            x_remove = cur_list[idx_remove]
            
            # Choose x_add weighted toward small numbers
            candidates = [x for x in range(1, N + 1) if not ((current_bits >> (x - 1)) & 1)]
            if not candidates:
                break
            
            if rng_sa.random() < 0.8:
                # Weighted selection favoring small x
                weights = [1.0 / x for x in candidates]
                total_w = sum(weights)
                r = rng_sa.random() * total_w
                cum = 0
                x_add = candidates[-1]
                for c, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        x_add = c
                        break
            else:
                x_add = candidates[rng_sa.randint(0, len(candidates) - 1)]
            
            if x_remove == x_add:
                continue
            
            if not _swap_valid_fast(current_bits, x_remove, x_add):
                continue
            
            new_bits = (current_bits & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
            new_score = current_score - 1.0 / x_remove + 1.0 / x_add
            delta = new_score - current_score
            
            if delta > 0 or rng_sa.random() < pow(2.718281828, delta / current_T):
                current_bits = new_bits
                current_score = new_score
                if current_score > best_score_sa:
                    best_score_sa = current_score
                    best_bits_sa = current_bits
            
            current_T *= alpha
        
        if best_score_sa > best_score:
            best_bits = best_bits_sa
            best_score = best_score_sa
    
    # Phase 4: Final local search
    if time.time() < deadline:
        best_bits, best_score = _local_search(best_bits, best_score, deadline)
    
    return _bits_to_list(best_bits)

def _bits_to_list(bits):
    return [i + 1 for i in range(N) if (bits >> i) & 1]

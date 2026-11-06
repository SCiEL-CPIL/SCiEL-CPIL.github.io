"""
Optimization program for maximizing 3-term APs in a subset of [1,256] with no 4-term AP.
Uses a multi-phase approach:
1. Greedy construction to build an initial valid set
2. Iterative local search with 1-move and 2-swap improvements
3. Simulated annealing with restarts to escape local optima
"""
import time
import random
import sys

N = 256
MAX_TIME = 85  # seconds, safety margin before 90s kill

def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in set S."""
    S = set(S)
    count = 0
    for a in S:
        for b in S:
            if b > a:
                c = 2*b - a
                if c in S and c > b:
                    count += 1
    return count

def has_4ap(S):
    """Check if set S contains any nonconstant 4-term arithmetic progression."""
    S = set(S)
    elems = sorted(S)
    n = len(elems)
    # For each pair (i, j) with i < j, check if the AP continues
    # 4-term AP: a, a+d, a+2d, a+3d
    # We check all possible (a, d) pairs
    for a in S:
        for d in range(1, N // 3 + 1):
            a2 = a + d
            a3 = a + 2*d
            a4 = a + 3*d
            if a2 in S and a3 in S and a4 in S:
                return True
    return False

def greedy_construct():
    """
    Greedy construction: add elements in order of highest marginal 3-AP gain
    while maintaining no-4-AP constraint.
    """
    S = set()
    remaining = set(range(1, N+1))
    
    while remaining:
        best_gain = -1
        best_elem = None
        for x in remaining:
            test_S = S | {x}
            if has_4ap(test_S):
                continue
            gain = count_3aps(test_S) - count_3aps(S)
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        if best_elem is None:
            break
        S.add(best_elem)
        remaining.remove(best_elem)
    
    return S

def local_search_1move(S, max_iters=10000):
    """
    Local search with single-add and single-remove moves.
    Returns improved set.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    n = N
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding elements
        for x in range(1, n+1):
            if x not in S:
                test_S = S | {x}
                if not has_4ap(test_S):
                    new_score = count_3aps(test_S)
                    if new_score > current_score:
                        S = test_S
                        current_score = new_score
                        improved = True
                        break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue
        
        # Try removing elements
        for x in list(S):
            test_S = S - {x}
            new_score = count_3aps(test_S)
            if new_score > current_score:
                S = test_S
                current_score = new_score
                improved = True
                break
        
        if not improved:
            break
    
    return best_S

def simulated_annealing(S, initial_temp=10.0, cooling=0.995, max_iters=50000):
    """
    Simulated annealing with add, remove, and swap moves.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    T = initial_temp
    n = N
    
    for it in range(max_iters):
        move_type = random.random()
        moved = False
        
        if move_type < 0.3:
            # Add move
            candidates = [x for x in range(1, n+1) if x not in S]
            if candidates:
                x = random.choice(candidates)
                test_S = S | {x}
                if not has_4ap(test_S):
                    new_score = count_3aps(test_S)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < _exp(delta / T):
                        S = test_S
                        current_score = new_score
                        moved = True
        elif move_type < 0.5:
            # Remove move
            if S:
                x = random.choice(list(S))
                test_S = S - {x}
                new_score = count_3aps(test_S)
                delta = new_score - current_score
                if delta > 0 or random.random() < _exp(delta / T):
                    S = test_S
                    current_score = new_score
                    moved = True
        else:
            # Swap move
            in_S = list(S)
            not_in_S = [x for x in range(1, n+1) if x not in S]
            if in_S and not_in_S:
                x = random.choice(in_S)
                y = random.choice(not_in_S)
                test_S = (S - {x}) | {y}
                if not has_4ap(test_S):
                    new_score = count_3aps(test_S)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < _exp(delta / T):
                        S = test_S
                        current_score = new_score
                        moved = True
        
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_S

def _exp(x):
    """Safe exponent for acceptance probability."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    import math
    return math.exp(x)

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    S = greedy_construct()
    score = count_3aps(S)
    best_S = set(S)
    best_score = score
    
    # Phase 2: Local search refinement
    S = local_search_1move(S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME * 0.3:
        return sorted(best_S)
    
    # Phase 3: Simulated annealing with restarts
    remaining_time = MAX_TIME - elapsed
    n_restarts = max(1, int(remaining_time / 5))
    
    for r in range(n_restarts):
        if time.time() - start_time > MAX_TIME - 5:
            break
        
        # Perturb the best solution
        perturbed = set(best_S)
        # Randomly add/remove a few elements
        for _ in range(random.randint(3, 10)):
            if random.random() < 0.5:
                candidates = [x for x in range(1, N+1) if x not in perturbed]
                if candidates:
                    perturbed.add(random.choice(candidates))
            else:
                if perturbed:
                    perturbed.discard(random.choice(list(perturbed)))
        
        # Check validity
        if has_4ap(perturbed):
            # Try to fix by removing problematic elements
            for x in list(perturbed):
                if not has_4ap(perturbed - {x}):
                    perturbed.discard(x)
                    break
        
        # Run SA from perturbed state
        sa_S = simulated_annealing(perturbed, initial_temp=5.0, cooling=0.998, max_iters=20000)
        sa_score = count_3aps(sa_S)
        
        if sa_score > best_score:
            best_score = sa_score
            best_S = set(sa_S)
        
        # Local search on SA result
        ls_S = local_search_1move(sa_S, max_iters=2000)
        ls_score = count_3aps(ls_S)
        if ls_score > best_score:
            best_score = ls_score
            best_S = set(ls_S)
    
    # Final verification
    if has_4ap(best_S):
        # Fallback to greedy
        best_S = greedy_construct()
    
    return sorted(best_S)

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, 3-APs: {count_3aps(result)}")
    print(f"Has 4-AP: {has_4ap(result)}")
    print(f"First 20: {result[:20]}")

import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Use numpy for speed
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score


def simulated_annealing(initial_set, deadline, max_iter=10000):
    """Simulated annealing to escape local optima"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    current_score = compute_score(U)
    best_score = current_score
    best_set = U[:]
    
    # Temperature schedule
    T0 = 0.1
    T_min = 0.0001
    cooling_rate = 0.999
    
    T = T0
    stagnant = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(U)
        max_val = max(U)
        
        # Generate a random move - focused on small perturbations
        move_type = random.random()
        
        if move_type < 0.7:
            # Replace one element with nearby value
            i = random.randint(1, n - 1)  # Don't replace 0
            # Small perturbation around current value
            step = max(1, max_val // 10)
            new_val = max(0, U[i] + random.randint(-step, step))
            
            if new_val in U and new_val != U[i]:
                continue
            
            new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])
                
        elif move_type < 0.85:
            # Add an element - try to extend pattern
            if len(U) >= 20:
                continue
            # Try values that might fit the pattern
            if random.random() < 0.4:
                new_val = max_val + random.randint(1, max(1, max_val // 5))
            elif random.random() < 0.7:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 2)
            
            if new_val in U:
                continue
            
            new_U = sorted(U + [new_val])
            
        elif move_type < 0.95:
            # Remove an element
            if len(U) <= 3:
                continue
            i = random.randint(1, n - 1)  # Don't remove 0
            new_U = U[:i] + U[i+1:]
            
        else:
            # Replace two elements with nearby values
            if n < 4:
                continue
            i, j = random.sample(range(1, n), 2)
            step = max(1, max_val // 10)
            new_i = max(0, U[i] + random.randint(-step, step))
            new_j = max(0, U[j] + random.randint(-step, step))
            
            new_U = sorted(set(U[:i] + [new_i] + U[i+1:j] + [new_j] + U[j+1:]))
            if 0 not in new_U:
                new_U = sorted([0] + [x for x in new_U if x > 0])
        
        if len(new_U) < 2:
            continue
        
        new_score = compute_score(new_U)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
            # Accept the move
            current_score = new_score
            U = new_U
            
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_set = U[:]
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1
        
        # Cool down
        T *= cooling_rate
        
        # Early termination if stuck
        if stagnant > 300 and T < 0.005:
            break
    
    return best_score, best_set

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets of size n"""
    best_score = 0.0
    best_set = None
    # Try all subsets of {0, 1, ..., max_val} of size n that contain 0
    candidates = list(range(1, max_val + 1))
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_score, best_set

def parametric_families(deadline):
    """Try various parametric families"""
    best_score = 0.0
    best_set = None
    
    def try_set(U):
        nonlocal best_score, best_set
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        if len(U) < 2:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 1: {0, 1, 2, ..., n-1} - arithmetic progression
    for n in range(3, 60):
        if time.time() > deadline:
            break
        try_set(list(range(n)))
    
    # Family 2: {0, 1, 3, 7, 15, ...} - Mersenne-like (2^k - 1)
    # Limit n to avoid extremely large numbers that slow down Python set fallback
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k - 1 for k in range(1, n)])
    
    # Family 3: {0, 1, 2, 4, 8, ...} - powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        try_set([0] + [2**k for k in range(1, n)])
    
    # Family 4: {0, 1, 3, 6, 10, ...} - triangular numbers
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k*(k+1)//2 for k in range(1, n)])
    
    # Family 5: {0, 1, 2, 4, 7, 11, ...} - quadratic
    for n in range(3, 40):
        if time.time() > deadline:
            break
        try_set([0] + [k*k//2 for k in range(1, n)])
    
    # Family 6: Mixed - {0} + small dense set + sparse large elements
    for base_n in range(3, 15):
        if time.time() > deadline:
            break
        for gap in range(2, 30):
            if time.time() > deadline:
                break
            for extra in range(0, 8):
                if time.time() > deadline:
                    break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                try_set(U)
    
    # Family 7: Geometric progression with base b
    for b in range(2, 15):
        if time.time() > deadline:
            break
        for n in range(3, 15):
            if time.time() > deadline:
                break
            try_set([0] + [b**k for k in range(1, n)])
    
    # Family 8: {0, a, 2a, ..., (m-1)a, m*a*b, (m+1)*a*b, ...}
    for m in range(3, 12):
        if time.time() > deadline:
            break
        for b in range(2, 10):
            if time.time() > deadline:
                break
            for extra in range(0, 6):
                if time.time() > deadline:
                    break
                U = list(range(m))
                for i in range(extra):
                    U.append(m * b * (i + 1))
                try_set(U)
    
    # Family 9: Sparse sets with controlled growth
    for start in range(1, 10):
        if time.time() > deadline:
            break
        for ratio in [2, 3, 4, 5]:
            if time.time() > deadline:
                break
            for n in range(3, 15):
                if time.time() > deadline:
                    break
                U = [0]
                val = start
                for _ in range(n - 1):
                    U.append(val)
                    val *= ratio
                try_set(U)
    
    # Family 10: Two-block structure {0, ..., a} ∪ {b, b+1, ..., b+c}
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 2, a * 5 + 5):
            if time.time() > deadline:
                break
            for c in range(1, 10):
                if time.time() > deadline:
                    break
                U = list(range(a + 1)) + list(range(b, b + c + 1))
                try_set(U)
    
    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement with smarter moves"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    stagnant = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try replacing each element with smart candidates
        for i in range(1, n):  # Don't try to replace 0
            if time.time() > deadline:
                break
            # Generate focused replacement candidates
            candidates = set()
            # Nearby values (small steps)
            step = max(1, max_val // 20)
            for d in range(-step, step + 1, max(1, step // 5)):
                candidates.add(max(0, U[i] + d))
            # Multiplicative neighbors
            candidates.add(max(1, U[i] * 2))
            if U[i] > 1:
                candidates.add(max(1, U[i] // 2))
            # Interpolation between neighbors
            if i > 1 and i < n - 1:
                candidates.add((U[i-1] + U[i+1]) // 2)
            # Random in reasonable range
            candidates.add(random.randint(1, max_val * 3))
            
            for new_val in sorted(candidates):
                if time.time() > deadline:
                    break
                if new_val == 0 and i > 0:
                    continue
                if new_val in U and new_val != U[i]:
                    continue
                new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if 0 not in new_U:
                    continue
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    stagnant = 0
                    break
        
        if not improved:
            stagnant += 1
            
            # Try adding a new element
            if len(U) < 25 and stagnant < 15:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    # Smart addition
                    if random.random() < 0.3:
                        new_val = max_val + random.randint(1, max(1, max_val // 3))
                    elif random.random() < 0.6:
                        new_val = random.randint(1, max_val)
                    else:
                        new_val = random.randint(max_val + 1, max_val * 2)
                    if new_val in U or new_val == 0:
                        continue
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        stagnant = 0
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3 and stagnant < 15:
                    for _ in range(8):
                        if time.time() > deadline:
                            break
                        idx = random.choice([i for i in range(1, len(U))])
                        new_U = U[:idx] + U[idx+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            stagnant = 0
                            break
            
            if not improved and stagnant > 50:
                break
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_score = 0.0
    best_set = [0, 1, 3, 9]  # Known good starting point
    
    # 1. Quick exhaustive search for small sets
    for n in range(3, 6):
        if time.time() > deadline - 30:
            break
        max_val = 20 if n <= 4 else 15
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Try key parametric families (focused)
    if time.time() < deadline - 25:
        s, U = parametric_families(deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 3. Hill climb from best found so far (primary refinement)
    if time.time() < deadline - 20:
        s, U = hill_climb(best_set, deadline, max_iter=5000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 4. Simulated annealing for global exploration
    if time.time() < deadline - 15:
        s, U = simulated_annealing(best_set, deadline, max_iter=10000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 5. Final hill climb polish
    if time.time() < deadline - 5:
        s, U = hill_climb(best_set, deadline, max_iter=3000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 6. Try a few more starting points if time remains
    if time.time() < deadline - 3:
        starts = [
            [0, 1, 2, 4, 8],
            [0, 1, 3, 7, 15],
            [0, 1, 3, 9, 27],
            [0, 1, 4, 9, 16],
        ]
        for U0 in starts:
            if time.time() > deadline - 2:
                break
            s, U2 = hill_climb(U0, deadline, max_iter=1000)
            if s > best_score:
                best_score = s
                best_set = U2
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

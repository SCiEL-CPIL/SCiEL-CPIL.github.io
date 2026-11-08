import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return -1.0
    
    # Compute sumset and diffset
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    r = dc / sc
    if r <= 1:
        return -1.0
    
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(r) / log(D)

def score_set_fast(U):
    """Faster score computation using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return -1.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairs (i,j) with i <= j
    sums = set()
    for i in range(n):
        sums.update((U_arr[i] + U_arr[i:]).tolist())
    
    # Diffset: all pairs (i,j) with i < j, plus 0
    diffs = set()
    for i in range(n):
        d = (U_arr[i+1:] - U_arr[i]).tolist()
        diffs.update(d)
        diffs.update(-x for x in d)
    diffs.add(0)
    
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    r = dc / sc
    if r <= 1:
        return -1.0
    
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(r) / log(D)

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustively search for best set of size n with elements up to max_val."""
    best_score = -1
    best_set = None
    
    if n < 2:
        return best_score, best_set
    
    # We need 0 in the set, so choose n-1 other elements from 1..max_val
    from itertools import combinations
    
    count = 0
    for combo in combinations(range(1, max_val + 1), n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
        count += 1
    
    return best_score, best_set

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def local_search(U, deadline, max_iterations=5000):
    """Local search with simulated annealing."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    temperature = 0.05
    cooling = 0.999
    min_temp = 0.001
    
    no_improve = 0
    max_no_improve = 200
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break
        if temperature < min_temp:
            break
        
        move_type = random.random()
        new_U = list(U)
        n = len(U)
        
        if move_type < 0.4:
            # Add an element
            if n < 30:
                max_val = max(U) if U else 10
                # Try values near existing elements or at strategic positions
                candidates = []
                for x in U:
                    for d in range(1, 6):
                        candidates.append(x + d)
                        candidates.append(max(1, x - d))
                # Also try some random values
                for _ in range(10):
                    candidates.append(random.randint(1, max_val + 20))
                # And some larger values
                for _ in range(5):
                    candidates.append(random.randint(max_val + 1, max_val * 2 + 10))
                
                random.shuffle(candidates)
                added = False
                for c in candidates[:30]:
                    if c <= 0 or c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score or random.random() < np.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
                        added = True
                        break
                if not added:
                    no_improve += 1
        elif move_type < 0.7:
            # Remove an element
            if n > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score or random.random() < np.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
            else:
                no_improve += 1
        else:
            # Replace an element
            if n > 2:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                max_val = max(U)
                candidates = []
                for d in range(1, 15):
                    candidates.append(old_val + d)
                    candidates.append(max(1, old_val - d))
                for _ in range(5):
                    candidates.append(random.randint(1, max_val + 30))
                
                random.shuffle(candidates)
                replaced = False
                for new_val in candidates[:20]:
                    if new_val <= 0 or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score or random.random() < np.exp((new_score - current_score) / temperature):
                        U = new_U
                        current_score = new_score
                        replaced = True
                        break
                if not replaced:
                    no_improve += 1
            else:
                no_improve += 1
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(U)
            no_improve = 0
        else:
            no_improve += 1
        
        temperature *= cooling
    
    return best_score, best_set

def fine_polish(U, deadline):
    """Fine-grained local search on the best set."""
    current = sorted(set(U))
    current_score = score_set(current)
    best_score = current_score
    best_set = list(current)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 2 and iteration < 1000:
        improved = False
        iteration += 1
        n = len(current)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old_val = current[i]
            for delta in range(-20, 21):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 25 and time.time() < deadline - 2:
            max_val = max(current)
            for new_val in range(max(1, max_val - 15), max_val + 16):
                if new_val <= 0 or new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break
        
        # Try removing an element
        if not improved and n > 4 and time.time() < deadline - 2:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # 1. Try structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],       # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128],            # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127],               # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84],               # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13],                 # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54],               # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],       # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243],                   # Powers of 3
        [0, 1, 5, 10, 15, 20, 25, 30],               # Arithmetic progression
        [0, 2, 5, 9, 14, 20, 27, 35],                # Quadratic
        [0, 1, 3, 6, 10, 15, 21, 28],                # Triangular
        [0, 1, 2, 5, 10, 17, 26, 37],                # Near quadratic
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Geometric search
    b_candidates = [b_int / 10.0 for b_int in range(12, 80)]
    for b in b_candidates:
        if time.time() > deadline - 60:
            break
        for m in range(2, 18):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 3. Exhaustive search for small sets (n=3,4,5)
    # This can find optimal small sets that local search might miss
    for n in [3, 4, 5]:
        if time.time() > deadline - 50:
            break
        # For n=3: choose 2 from 1..30
        # For n=4: choose 3 from 1..25
        # For n=5: choose 4 from 1..20
        max_vals = {3: 30, 4: 25, 5: 20}
        mv = max_vals[n]
        es, es_set = exhaustive_search_small(n, mv, deadline)
        if es > best_score:
            best_score = es
            best_set = es_set
    
    # 4. Local search with multiple restarts
    seeds = [best_set]
    # Add some diverse seeds
    for cand in structured_candidates[:5]:
        if cand != best_set:
            seeds.append(cand)
    
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search(U, deadline - 15)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # 5. Fine polish on the best set
    if time.time() < deadline - 5:
        polish_score, polish_set = fine_polish(best_set, deadline)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set = sorted([0] + best_set)
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

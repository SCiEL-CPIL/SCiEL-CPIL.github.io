import itertools
import math
import time
import numpy as np

def _score_set(U):
    """Compute the score for a set U of nonnegative integers."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))  # ensure uniqueness, keep 0
    U.sort()
    n = len(U)
    if n < 1:
        return 0.0
    
    # Compute sumset |U+U| using numpy for speed
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    P = len(np.unique(sums))
    
    # Compute difference set |U-U|
    diffs = U_arr[:, None] - U_arr[None, :]
    D = len(np.unique(diffs))
    
    max_U = max(U)
    if max_U == 0:
        return 0.0
    
    ratio = D / P
    if ratio <= 1:
        return 0.0
    
    denom = math.log(2 * max_U + 1)
    if denom == 0:
        return 0.0
    
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets
    def search_exhaustive(k, max_val):
        nonlocal best_score, best_set
        for combo in itertools.combinations(range(1, max_val + 1), k - 1):
            U = [0] + list(combo)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
            
            if time.time() > deadline:
                return True
        
        return False
    
    # Expanded search due to faster numpy scoring
    search_configs = [
        (2, 30),
        (3, 30),
        (4, 30),
        (5, 25),
        (6, 20),
        (7, 15),
        (8, 12),
    ]
    
    for k, max_val in search_configs:
        if search_exhaustive(k, max_val):
            break
    
    # Phase 2: Greedy growth + local search
    def greedy_grow(U):
        nonlocal best_score, best_set
        U = list(U)
        improved = True
        while improved:
            improved = False
            current_score = _score_set(U)
            # Try adding each possible new element
            for x in range(1, 80):
                if x not in U:
                    new_U = U + [x]
                    score = _score_set(new_U)
                    if score > current_score + 1e-9:
                        U = new_U
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = U
                        if time.time() > deadline:
                            return
            # Try removing an element if it helps
            for x in list(U):
                if x == 0:
                    continue
                new_U = [y for y in U if y != x]
                if len(new_U) >= 2:
                    score = _score_set(new_U)
                    if score > current_score + 1e-9:
                        U = new_U
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = U
                        if time.time() > deadline:
                            return
        
        return U

    best_set = greedy_grow(best_set)
    
    # Phase 3: Local search with swaps and two-element additions
    if time.time() < deadline - 15:
        U = list(best_set)
        no_improve = 0
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            current_score = _score_set(U)
            improved = False
            
            # Try single element addition
            for x in range(1, 100):
                if x not in U:
                    new_U = U + [x]
                    score = _score_set(new_U)
                    if score > current_score + 1e-9:
                        U = new_U
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = U
                        break
            
            # Try single element removal
            if not improved:
                for x in list(U):
                    if x == 0:
                        continue
                    new_U = [y for y in U if y != x]
                    if len(new_U) >= 2:
                        score = _score_set(new_U)
                        if score > current_score + 1e-9:
                            U = new_U
                            current_score = score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_set = U
                            break
            
            # Try swap: remove one, add another
            if not improved:
                for x in list(U):
                    if x == 0:
                        continue
                    for y in range(1, 100):
                        if y not in U:
                            new_U = [z for z in U if z != x] + [y]
                            score = _score_set(new_U)
                            if score > current_score + 1e-9:
                                U = new_U
                                current_score = score
                                improved = True
                                if score > best_score:
                                    best_score = score
                                    best_set = U
                                break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve >= 3:
                    break
            else:
                no_improve = 0
    
    # Phase 4: Final polish - fine-tune the largest element
    if time.time() < deadline - 5:
        U = list(best_set)
        U.sort()
        # Try adjusting the max element by small amounts
        max_U = U[-1]
        for delta in range(-5, 6):
            if delta == 0:
                continue
            new_max = max_U + delta
            if new_max <= 0:
                continue
            new_U = U[:-1] + [new_max]
            if len(new_U) == len(set(new_U)):
                score = _score_set(new_U)
                if score > best_score + 1e-9:
                    best_score = score
                    best_set = new_U
    
    return best_set

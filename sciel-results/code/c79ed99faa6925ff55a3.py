import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Try structured candidates with geometric growth
    for base in [2, 3, 4, 5, 7]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # Try {0, 1, a, a*b} type sets
    for a in range(2, 40):
        if time.time() > deadline:
            break
        for b in range(2, 25):
            if time.time() > deadline:
                break
            c = a * b
            if c > 800:
                break
            try_set([0, 1, a, c])
    
    # Try {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 40):
            if time.time() > deadline:
                break
            for c in range(b + 1, 120):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 80
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 55):
            if time.time() > deadline:
                break
            for c in range(b + 1, 81):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 45
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 35):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 46):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try 6-element sets with geometric growth
    for a in range(1, 7):
        if time.time() > deadline:
            break
        for b in range(a + 1, 12):
            if time.time() > deadline:
                break
            for c in range(b + 1, 20):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 30):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 50):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Try sets with one large element and small dense part
    for small_max in range(5, 25):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, small_max * 6 + 1):
            if time.time() > deadline:
                break
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Try {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(3, 18):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 6 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Try arithmetic progressions
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for d in range(1, 20):
            if time.time() > deadline:
                break
            for n in range(4, 9):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                try_set(U)
    
    # Local improvement: hill-climb on the best set found
    # Try replacing each element with nearby values, adding elements, or removing elements
    for iteration in range(50):
        if time.time() > deadline:
            break
        improved = False
        U_sorted = sorted(set(best_U))
        n = len(U_sorted)
        
        # Try replacing each element (except 0) with nearby values
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = U_sorted[i]
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Check if new_val would cause a duplicate
                if new_val in U_sorted:
                    continue
                new_U = U_sorted[:i] + [new_val] + U_sorted[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        # Try adding a new element
        if time.time() <= deadline:
            max_val = U_sorted[-1]
            for new_val in range(1, max_val + 30):
                if time.time() > deadline:
                    break
                if new_val in U_sorted:
                    continue
                new_U = sorted(U_sorted + [new_val])
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        # Try removing a non-zero element
        if time.time() <= deadline and n > 2:
            for i in range(1, n):
                if time.time() > deadline:
                    break
                new_U = U_sorted[:i] + U_sorted[i+1:]
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_U

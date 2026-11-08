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
    deadline = start_time + 135  # safety margin before 150s kill
    
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
    
    # Geometric sequences: {0, 1, base, base^2, ..., base^(n-1)}
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for n in range(3, 9):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            if not try_set(U):
                break
    
    # {0, 1, a, a*b} type multiplicative structures
    for a in range(2, 40):
        if time.time() > deadline:
            break
        for b in range(2, 25):
            if time.time() > deadline:
                break
            c = a * b
            if c > 800:
                break
            if not try_set([0, 1, a, c]):
                break
    
    # {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 50):
            if time.time() > deadline:
                break
            for c in range(b + 1, 150):
                if time.time() > deadline:
                    break
                if not try_set([0, 1, a, b, c]):
                    break
    
    # Dense small part + one large element: {0, 1, ..., k} + {m}
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 5 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            if not try_set(U):
                break
    
    # Small dense part {0, 1, ..., small_max} + large
    for small_max in range(5, 20):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, small_max * 5 + 1):
            if time.time() > deadline:
                break
            U = list(range(small_max + 1)) + [large]
            if not try_set(U):
                break
    
    # Arithmetic progressions {0, d, 2d, ..., (n-1)d}
    for d in range(1, 15):
        if time.time() > deadline:
            break
        for n in range(4, 8):
            if time.time() > deadline:
                break
            U = [d * i for i in range(n)]
            if not try_set(U):
                break
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 80
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if time.time() > deadline:
                break
            for c in range(b + 1, 81):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 50
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 35):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 51):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # 6-element sets with geometric-like growth
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 10):
            if time.time() > deadline:
                break
            for c in range(b + 1, 15):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 25):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 40):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Additive chains {0, a, a+b, a+b+c, ...}
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for b in range(1, 20):
            if time.time() > deadline:
                break
            for c in range(1, 25):
                if time.time() > deadline:
                    break
                if not try_set([0, a, a + b, a + b + c]):
                    break
                if not try_set([0, a, a + b, a + b + c, a + b + 2*c]):
                    break
    
    # 3-element sets for comparison
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if not try_set([0, a, b]):
                break
    
    return best_U

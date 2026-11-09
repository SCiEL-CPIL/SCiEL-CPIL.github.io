import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 30
    # Precompute 1/i as Fractions for exact arithmetic
    inv = [Fraction(0) for _ in range(N)]
    for i in range(N):
        inv[i] = Fraction(1, i + 1)
    
    best_val = None
    best_coeffs = None
    best_abs = None
    
    def evaluate(coeffs):
        s = Fraction(0)
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * inv[i]
        if s == 0:
            return None, None
        return s, abs(s)
    
    def random_valid():
        while True:
            c = [random.choice([-1, 0, 1]) for _ in range(N)]
            s, a = evaluate(c)
            if s is not None:
                return c, s, a
    
    # Initial random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        c, s, a = random_valid()
        if best_abs is None or a < best_abs:
            best_abs = a
            best_val = s
            best_coeffs = c[:]
    
    # Local search from best
    def local_search(coeffs, max_iters=5000):
        nonlocal best_abs, best_val, best_coeffs
        c = coeffs[:]
        s, a = evaluate(c)
        if s is None:
            return
        if best_abs is None or a < best_abs:
            best_abs = a
            best_val = s
            best_coeffs = c[:]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return
            # Pick a random position to change
            pos = random.randint(0, N - 1)
            old = c[pos]
            new = random.choice([-1, 0, 1])
            if new == old:
                continue
            c[pos] = new
            s2, a2 = evaluate(c)
            if s2 is None:
                c[pos] = old
                continue
            if a2 < a:
                a = a2
                s = s2
                if best_abs is None or a < best_abs:
                    best_abs = a
                    best_val = s
                    best_coeffs = c[:]
            else:
                c[pos] = old
        return
    
    # Run local search from multiple starting points
    num_ls = 50
    for i in range(num_ls):
        if time.time() > deadline:
            break
        if i == 0:
            c = best_coeffs[:]
        else:
            c, s, a = random_valid()
            if best_abs is not None and a > best_abs * 2:
                # Perturb the best instead
                c = best_coeffs[:]
                for _ in range(random.randint(1, 3)):
                    pos = random.randint(0, N - 1)
                    c[pos] = random.choice([-1, 0, 1])
                s, a = evaluate(c)
                if s is None:
                    c, s, a = random_valid()
        local_search(c, max_iters=3000)
    
    # Final check
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    return best_coeffs

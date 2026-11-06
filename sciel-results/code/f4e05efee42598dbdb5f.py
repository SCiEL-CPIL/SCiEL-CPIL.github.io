"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import random
from collections import deque

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def _evaluate(coeffs):
    """Return (numerator, denominator) of the signed sum."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1  # term is c / (i+1)
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def construct():
    deadline = time.time() + 55  # safety margin before 90s kill
    n = 30
    L = _lcm_range(n)  # lcm(1..30)
    weights = [L // (i + 1) for i in range(n)]
    
    # Trivial solution: 1/29 - 1/30 = 1/(29*30) = 1/870
    # N = L/29 - L/30 = L * (30-29)/(29*30) = L/870
    # But let's just compute: best_coeffs = [0]*28 + [1, -1]
    best_coeffs = [0] * 28 + [1, -1]
    best_N = sum(c * w for c, w in zip(best_coeffs, weights))
    best_abs_N = abs(best_N)
    
    # Try to find smaller nonzero |N|
    # Strategy: use a meet-in-the-middle-like approach or smart local search
    # Given 30 variables each with 3 choices, total space is 3^30 ~ 2e14 - too large for full enumeration
    # But we can use beam search or priority-based exploration
    
    # Let's use a best-first search with a priority queue (min-heap by |N|)
    # But state space is still large. Let's use a smarter approach.
    
    # Observation: the weights are L/(i+1) for i=0..29
    # L = lcm(1,2,...,30) = 2329089562800
    # weights[i] = L/(i+1)
    
    # Key insight: we want to find coefficients c_i in {-1,0,1} such that
    # N = sum c_i * weights[i] is nonzero and as small as possible in absolute value.
    
    # Since weights are all divisors of L, and L is huge, the smallest nonzero |N|
    # could be very small (like 1) if we can find a subset that sums to exactly 1 or -1.
    
    # Let's use a systematic approach:
    # 1. Start with the trivial solution
    # 2. Try local moves that reduce |N|
    # 3. Use randomized restarts with hill climbing
    
    rng = random.Random(12345)
    
    def compute_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def local_search(coeffs, max_iters=None):
        """Hill climbing with random restarts within the search."""
        current = list(coeffs)
        current_N = compute_N(current)
        best_local = list(current)
        best_local_N = current_N
        best_local_abs = abs(current_N)
        
        if max_iters is None:
            max_iters = 50000
        
        iters = 0
        stuck_count = 0
        while iters < max_iters and time.time() < deadline - 2:
            iters += 1
            # Pick a random position to change
            idx = rng.randrange(n)
            old_val = current[idx]
            # Try a new value different from old
            choices = [v for v in (-1, 0, 1) if v != old_val]
            new_val = rng.choice(choices)
            delta = (new_val - old_val) * weights[idx]
            new_N = current_N + delta
            new_abs = abs(new_N)
            
            if new_abs < best_local_abs:
                current[idx] = new_val
                current_N = new_N
                best_local = list(current)
                best_local_N = new_N
                best_local_abs = new_abs
                stuck_count = 0
                if best_local_abs == 1:
                    break
            else:
                # With some probability, accept to escape local minima
                if rng.random() < 0.05:
                    current[idx] = new_val
                    current_N = new_N
                stuck_count += 1
            
            if stuck_count > 5000:
                break
        
        return best_local, best_local_N
    
    # Multi-start local search
    # Start with the trivial solution and several random starts
    starts = [best_coeffs]
    for _ in range(50):
        starts.append([rng.choice([-1, 0, 1]) for _ in range(n)])
    
    for start in starts:
        if time.time() > deadline - 3:
            break
        local_best, local_N = local_search(start)
        local_abs = abs(local_N)
        if local_abs < best_abs_N:
            best_coeffs = local_best
            best_N = local_N
            best_abs_N = local_abs
            if best_abs_N == 1:
                break
    
    # Additional targeted search: try to combine pairs
    # For each pair (i, j), check if there's a combination that gives small N
    # Actually, let's do a more systematic pairwise improvement
    
    if time.time() < deadline - 5 and best_abs_N > 1:
        # Try pairwise and triple moves
        current = list(best_coeffs)
        current_N = best_N
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try changing two positions at once
            for i in range(n):
                for j in range(i + 1, n):
                    for ci in [-1, 0, 1]:
                        for cj in [-1, 0, 1]:
                            if ci == current[i] and cj == current[j]:
                                continue
                            delta = (ci - current[i]) * weights[i] + (cj - current[j]) * weights[j]
                            new_N = current_N + delta
                            new_abs = abs(new_N)
                            if new_abs < abs(current_N):
                                current[i] = ci
                                current[j] = cj
                                current_N = new_N
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if abs(current_N) < best_abs_N:
            best_coeffs = current
            best_N = current_N
            best_abs_N = abs(current_N)
    
    # Final attempt: if we haven't found something very small, try more random restarts
    if time.time() < deadline - 3 and best_abs_N > 100:
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            start = [rng.choice([-1, 0, 1]) for _ in range(n)]
            local_best, local_N = local_search(start, max_iters=20000)
            local_abs = abs(local_N)
            if local_abs < best_abs_N:
                best_coeffs = local_best
                best_N = local_N
                best_abs_N = local_abs
                if best_abs_N == 1:
                    break
    
    # Convert best_N to (num, den)
    num = best_N
    den = L
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    
    # Ensure den is positive
    if den < 0:
        num = -num
        den = -den
    
    # Verify nonzero
    if num == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs

# EVOLVE-BLOCK-END

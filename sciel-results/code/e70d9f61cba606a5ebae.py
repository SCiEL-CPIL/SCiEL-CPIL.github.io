"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    n = 30
    # We need to find coefficients c_i in {-1, 0, 1} for i=1..30
    # such that S = sum(c_i / i) != 0 and |S| is minimized.
    
    # The LCM of 1..30 is:
    # 30! / (product of prime powers in 30!) ... actually let's compute it
    # LCM(1..30) = 2329089562800
    
    # Let's compute LCM properly
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    
    # L = 2329089562800
    # S = sum(c_i / i) = (sum(c_i * L/i)) / L
    # Let w_i = L / i (integer)
    # We want to minimize |sum(c_i * w_i)| subject to sum(c_i * w_i) != 0
    
    weights = [L // i for i in range(1, 31)]
    # weights[i-1] = L/i for i=1..30
    
    # This is a subset sum / signed subset sum problem.
    # We want to find c_i in {-1,0,1} to make sum(c_i * w_i) as close to 0 as possible, but not 0.
    
    # Meet in the middle approach:
    # Split into two halves of 15 each.
    # For each half, enumerate all 3^15 = 14348907 possibilities.
    # That's about 14M per half, which might be too slow in Python.
    # 3^15 = 14348907, that's a lot but let's try to be smarter.
    
    # Actually, let's think about the magnitudes.
    # w_1 = L = 2329089562800
    # w_30 = L/30 = 77636318760
    # The weights vary enormously.
    
    # A better approach: since w_1 is huge, c_1 is either -1, 0, or 1.
    # If c_1 != 0, then |sum| >= w_1 - sum_{i=2}^{30} w_i (if all others are -c_1)
    # sum_{i=2}^{30} w_i = L * sum(1/i for i=2..30) = L * (H_30 - 1)
    # H_30 ≈ 3.995, so H_30 - 1 ≈ 2.995
    # w_1 = L, sum_{i=2}^{30} w_i ≈ 2.995 * L
    # So if c_1 = 1, min |sum| >= L - 2.995L < 0, so we can potentially get close to 0.
    
    # Let's use a different strategy. Since the weights are very different in magnitude,
    # we can use a greedy/DP approach or random search with local optimization.
    
    # Actually, let's try a randomized approach with local search.
    # Start with random assignments, then try flipping individual coefficients to improve.
    
    import random
    random.seed(42)
    
    best_val = None
    best_coeffs = None
    
    def evaluate(coeffs):
        """Compute the signed sum as a Fraction, return (abs_value, coeffs_copy)"""
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        return abs(s), s
    
    # Try random initial solutions
    num_attempts = 500000
    attempt = 0
    
    while time.time() < deadline:
        # Generate random coefficients
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Check if sum is zero
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        
        if s == 0:
            continue
        
        abs_s = abs(s)
        
        if best_val is None or abs_s < best_val:
            best_val = abs_s
            best_coeffs = coeffs[:]
        
        # Local search: try flipping each coefficient
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    coeffs[i] = new_c
                    s_new = Fraction(0, 1)
                    for j, c in enumerate(coeffs):
                        if c != 0:
                            s_new += Fraction(c, j + 1)
                    if s_new == 0:
                        coeffs[i] = old_c
                        continue
                    abs_new = abs(s_new)
                    if abs_new < best_val:
                        best_val = abs_new
                        best_coeffs = coeffs[:]
                        improved = True
                    else:
                        coeffs[i] = old_c
                if time.time() > deadline:
                    break
        
        attempt += 1
        if attempt % 100 == 0:
            pass  # progress tracking
    
    if best_coeffs is None:
        # Fallback: return the seed program's answer
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END

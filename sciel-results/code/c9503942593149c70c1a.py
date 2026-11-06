"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _evaluate(coeffs):
    """Compute the signed harmonic sum as (num, den) in reduced form."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def _abs_val(num, den):
    return abs(num) / den

def construct():
    deadline = time.time() + 50
    
    # Default: 1/29 - 1/30 = 1/870
    best_val = float('inf')
    best_coeffs = [0] * 28 + [1, -1]
    
    def update(coeffs):
        nonlocal best_val, best_coeffs
        num, den = _evaluate(coeffs)
        if num == 0:
            return
        val = _abs_val(num, den)
        if val < best_val:
            best_val = val
            best_coeffs = list(coeffs)
    
    # Phase 1: Systematic pair search - try all pairs (i, j) with i < j
    # Sum = 1/i - 1/j = (j-i)/(i*j), minimize |j-i|/(i*j)
    # For fixed difference d, minimize d/(i*(i+d)) -> maximize i*(i+d)
    # Best is largest i, i.e., i=28, j=29: 1/28-1/29 = 1/812
    # Or i=29, j=30: 1/29-1/30 = 1/870
    # But also consider 1/i + 1/j - 1/k etc.
    
    # Try all pairs
    for i in range(1, 30):
        for j in range(i+1, 31):
            coeffs = [0] * 30
            coeffs[i-1] = 1
            coeffs[j-1] = -1
            update(coeffs)
    
    # Try pairs with same sign: 1/i + 1/j
    for i in range(1, 30):
        for j in range(i+1, 31):
            coeffs = [0] * 30
            coeffs[i-1] = 1
            coeffs[j-1] = 1
            update(coeffs)
    
    # Phase 2: Systematic triple search
    # Try 1/i - 1/j - 1/k, 1/i + 1/j - 1/k, etc.
    # Focus on triples where indices are close to each other
    for i in range(1, 31):
        for j in range(i+1, 31):
            for k in range(j+1, 31):
                # Only check if time allows
                if time.time() > deadline - 10:
                    break
                # Try sign patterns: +-- , -++, +-+ , -+- 
                for signs in [(1,-1,-1), (-1,1,1), (1,-1,1), (-1,1,-1)]:
                    coeffs = [0] * 30
                    coeffs[i-1] = signs[0]
                    coeffs[j-1] = signs[1]
                    coeffs[k-1] = signs[2]
                    update(coeffs)
    
    # Phase 3: Random local search
    random.seed(42)
    
    # Start from best found so far
    current = list(best_coeffs)
    update(current)
    
    # Also try random starts
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Random sparse starting point
        current = [0] * 30
        num_ones = random.randint(2, 6)
        indices = random.sample(range(30), num_ones)
        for idx in indices:
            current[idx] = random.choice([-1, 1])
        update(current)
        
        # Local search: flip, add, remove coefficients
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            action = random.random()
            if action < 0.4:
                # Flip a coefficient
                idx = random.randint(0, 29)
                current[idx] = -current[idx] if current[idx] != 0 else random.choice([-1, 1])
            elif action < 0.7:
                # Set a random position to 0 or +/-1
                idx = random.randint(0, 29)
                current[idx] = random.choice([-1, 0, 1])
            else:
                # Swap two coefficients
                i, j = random.sample(range(30), 2)
                current[i], current[j] = current[j], current[i]
            
            update(current)
    
    # Phase 4: More targeted search - try to find small sums with many terms
    # The key insight: sum of 1/i with signs can be very small if terms nearly cancel
    # Try greedy approaches with random perturbations
    
    for restart in range(100):
        if time.time() > deadline - 3:
            break
        
        # Start with a random subset of high-index terms (smaller values)
        current = [0] * 30
        # Focus on indices 15-29 (smaller harmonic terms)
        pool = list(range(15, 30))
        random.shuffle(pool)
        num_terms = random.randint(3, 8)
        for idx in pool[:num_terms]:
            current[idx] = random.choice([-1, 1])
        update(current)
        
        # Local search
        for step in range(1000):
            if time.time() > deadline - 3:
                break
            
            # Random mutation
            idx = random.randint(0, 29)
            old_val = current[idx]
            current[idx] = random.choice([-1, 0, 1])
            
            num, den = _evaluate(current)
            if num != 0:
                val = _abs_val(num, den)
                if val < best_val:
                    best_val = val
                    best_coeffs = list(current)
                else:
                    # Revert with some probability (simulated annealing-like)
                    if random.random() > 0.5:
                        current[idx] = old_val
            else:
                current[idx] = old_val
    
    # Phase 5: Try specific constructions
    # 1/n - 1/(n+1) - 1/(n+2) + ... alternating patterns
    for start in range(1, 25):
        for length in range(3, 10):
            if start + length > 31:
                break
            coeffs = [0] * 30
            for k in range(length):
                coeffs[start-1+k] = 1 if k % 2 == 0 else -1
            update(coeffs)
    
    return best_coeffs
# EVOLVE-BLOCK-END

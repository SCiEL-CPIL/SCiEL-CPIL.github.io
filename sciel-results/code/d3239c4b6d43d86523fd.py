"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction

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
    start_time = time.time()
    deadline = start_time + 55.0
    n = 30
    
    # Precompute weights for integer arithmetic
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    weights = [L // (i + 1) for i in range(n)]
    
    def get_energy(coeffs):
        s = 0
        for i in range(n):
            s += coeffs[i] * weights[i]
        return abs(s)
    
    # Initial solution
    best_coeffs = [0] * 28 + [1, -1]
    best_energy = get_energy(best_coeffs)
    
    # Simulated Annealing
    T = 100000.0
    T_min = 1.0
    cooling = 0.999
    
    current_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
    if all(c == 0 for c in current_coeffs):
        current_coeffs[0] = 1
    current_energy = get_energy(current_coeffs)
    
    if current_energy > 0 and current_energy < best_energy:
        best_energy = current_energy
        best_coeffs = list(current_coeffs)
        
    while time.time() < deadline and T > T_min:
        new_coeffs = list(current_coeffs)
        
        if random.random() < 0.8:
            idx = random.randint(0, n - 1)
            old_val = new_coeffs[idx]
            if old_val == 0:
                new_val = random.choice([-1, 1])
            else:
                new_val = random.choice([0, -old_val])
            new_coeffs[idx] = new_val
        else:
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % n
            new_coeffs[idx1] = random.choice([-1, 0, 1])
            new_coeffs[idx2] = random.choice([-1, 0, 1])
            
        new_energy = get_energy(new_coeffs)
        
        if new_energy == 0:
            current_coeffs = new_coeffs
            current_energy = 0
        else:
            delta = new_energy - current_energy
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_coeffs = new_coeffs
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_coeffs = list(current_coeffs)
        
        T *= cooling
        
    # Final local search
    current_coeffs = list(best_coeffs)
    current_energy = get_energy(current_coeffs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            old_val = current_coeffs[i]
            for new_val in [-1, 0, 1]:
                if new_val == old_val:
                    continue
                current_coeffs[i] = new_val
                new_energy = get_energy(current_coeffs)
                if new_energy > 0 and new_energy < current_energy:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_coeffs = list(current_coeffs)
                    improved = True
                else:
                    current_coeffs[i] = old_val
    
    num, den = _evaluate(best_coeffs)
    if num == 0:
        best_coeffs = [0] * 28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END

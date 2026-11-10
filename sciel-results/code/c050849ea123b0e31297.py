import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    gcd_cache = {}
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        key = (m1, m2)
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(m1, m2)
        return gcd_cache[key]

    def is_compatible(current, r_new, m_new):
        for r_old, m_old in current:
            g = get_gcd(m_old, m_new)
            if (r_new - r_old) % g == 0:
                return False
        return True

    best_solution = []
    best_size = 0
    
    # 1. Deterministic Greedy (Natural Order)
    current = []
    for m in moduli:
        for r in range(m):
            if is_compatible(current, r, m):
                current.append((r, m))
                break
    if len(current) > best_size:
        best_size = len(current)
        best_solution = current[:]

    # 2. Deterministic Greedy (Reverse Order)
    current = []
    for m in reversed(moduli):
        for r in range(m):
            if is_compatible(current, r, m):
                current.append((r, m))
                break
    if len(current) > best_size:
        best_size = len(current)
        best_solution = current[:]

    # 3. Randomized Search
    while time.time() < deadline:
        moduli_copy = moduli[:]
        random.shuffle(moduli_copy)
        current = []
        for m in moduli_copy:
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(current, r, m):
                    current.append((r, m))
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_solution = current[:]
            
    return [[r, m] for r, m in best_solution]

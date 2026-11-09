import math
import time
import random

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return False
    return ((r2 - r1) % g) != 0

def verify_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        if m1 < 2 or m1 > MAX_MOD or r1 < 0 or r1 >= m1:
            return False
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if m1 == m2:
                return False
            if not is_disjoint(r1, m1, r2, m2):
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin before 60s kill
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    best_solution = []
    
    # Strategy: Greedy with random restarts
    # We try to build a solution by adding moduli one by one.
    # We shuffle the moduli to get different orders.
    # For each modulus, we try random residues.
    
    num_restarts = 1000 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle moduli to vary the order of selection
        # Prioritize moduli that are "hard" to place? Or just random?
        # Let's try random order first.
        current_moduli = all_moduli[:]
        random.shuffle(current_moduli)
        
        current_pairs = []
        success = True
        
        for m in current_moduli:
            if time.time() > deadline:
                success = False
                break
                
            # Try to find a residue r for m that is disjoint from all in current_pairs
            # We try random residues. If we fail after some tries, we skip m or restart?
            # Skipping m allows us to keep the rest.
            
            found = False
            # Try a few random residues
            # If m is small, we can check all. If m is large, random is better.
            if m <= 20:
                residues_to_try = list(range(m))
                random.shuffle(residues_to_try)
            else:
                residues_to_try = [random.randint(0, m-1) for _ in range(50)]
                # Ensure we try 0 and m-1 as well?
                if 0 not in residues_to_try: residues_to_try.append(0)
                if m-1 not in residues_to_try: residues_to_try.append(m-1)
                
            for r in residues_to_try:
                ok = True
                for (r2, m2) in current_pairs:
                    if not is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    current_pairs.append([r, m])
                    found = True
                    break
                    
            if not found:
                # Option 1: Skip this modulus
                # Option 2: Restart
                # Skipping is better to accumulate more pairs
                pass
        
        if len(current_pairs) > len(best_solution):
            # Verify before updating best to be safe
            if verify_solution(current_pairs):
                best_solution = current_pairs[:]
                
    if best_solution:
        return best_solution
        
    # Fallback: Greedy approach with even moduli
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    pairs = []
    for m in sorted(even_moduli, reverse=True):
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                if not is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                pairs.append([r, m])
                break
    
    if pairs and verify_solution(pairs):
        return pairs
    
    # Final fallback
    return [[m // 2, m] for m in [2, 4, 8, 16, 32, 64, 128]]

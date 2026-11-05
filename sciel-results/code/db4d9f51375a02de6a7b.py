import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    # Precompute gcds
    gcds = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            m1 = moduli[i]
            m2 = moduli[j]
            gcds[(m1, m2)] = math.gcd(m1, m2)
            
    def is_compatible(r_new, m_new, selected):
        for r_k, m_k in selected:
            g = gcds.get((m_new, m_k), math.gcd(m_new, m_k))
            if r_new % g == r_k % g:
                return False
        return True

    def solve_greedy(order):
        selected = []
        for m in order:
            # Try residues
            # Heuristic: try residues that are "far" from existing ones?
            # Just try all
            found = False
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    found = True
                    break
            if not found:
                # If we can't place m, skip it? 
                # For maximizing count, we should skip and try next.
                pass
        return selected

    # We want to maximize the number of selected moduli.
    # Greedy might get stuck.
    # Let's try multiple random orders and heuristics.
    
    best_solution = []
    best_len = 0
    
    # Heuristic 1: Primes first? No, primes have gcd 1 with others.
    # We need moduli with gcd > 1.
    # So we should prioritize moduli with many small prime factors?
    # Or moduli that are multiples of 2, 3, etc.
    
    # Let's sort by number of divisors descending (more connected)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_counts = {m: num_divisors(m) for m in moduli}
    
    orders = []
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    
    random.seed(42)
    for _ in range(20):
        shuffled = moduli.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
        sol = solve_greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            
    # If greedy is not enough, we might need backtracking.
    # But given the score jump potential, let's see if we can do better.
    # The current score is 2.57 (18 classes).
    # If we can get 20+, it's an improvement.
    
    # Let's try a simple backtracking for the best order found, allowing removal of previous choices?
    # Too complex for now.
    
    return [[r, m] for r, m in best_solution]

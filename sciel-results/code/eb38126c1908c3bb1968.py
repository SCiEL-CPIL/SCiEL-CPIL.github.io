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
            found = False
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    found = True
                    break
            if not found:
                pass
        return selected

    # We want to maximize the number of selected moduli.
    best_solution = []
    best_len = 0
    
    # Heuristic: Primes first? No, primes have gcd 1 with others.
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
    
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    div_counts = {m: num_divisors(m) for m in moduli}
    pf_counts = {m: count_prime_factors(m) for m in moduli}
    
    orders = []
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    orders.append(sorted(moduli, key=lambda m: -pf_counts[m]))
    orders.append(sorted(moduli, key=lambda m: pf_counts[m]))
    
    random.seed(42)
    for _ in range(30):
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
    
    # Backtracking search to improve
    def branch_and_bound(selected, remaining_moduli):
        nonlocal best_solution, best_len, deadline
        
        if time.time() > deadline:
            return
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_solution = list(selected)
        
        if not remaining_moduli:
            return
        
        # Upper bound: current + remaining
        if len(selected) + len(remaining_moduli) <= best_len:
            return
        
        # Try each remaining modulus
        for idx, m in enumerate(remaining_moduli):
            if time.time() > deadline:
                return
            
            # Try residues
            found = False
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    branch_and_bound(selected, remaining_moduli[:idx] + remaining_moduli[idx+1:])
                    selected.pop()
                    found = True
                    break
    
    # Start B&B from the best greedy solution
    if best_len > 0:
        # Convert to set of selected moduli for efficiency
        selected_moduli = set(m for r, m in best_solution)
        remaining = [m for m in moduli if m not in selected_moduli]
        
        # Try B&B from best solution
        current_selected = list(best_solution)
        branch_and_bound(current_selected, remaining)
        
        # Also try B&B from empty with different orderings
        for order in orders[:5]:  # Only first few to save time
            if time.time() > deadline:
                break
            branch_and_bound([], order)
    
    return [[r, m] for r, m in best_solution]

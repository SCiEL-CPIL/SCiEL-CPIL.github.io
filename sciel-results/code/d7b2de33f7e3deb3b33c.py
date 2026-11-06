"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Moduli we can choose from: 2..128
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(selected, r_new, m_idx_new):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected."""
        for (r_sel, m_idx_sel) in selected:
            g = gcd_table[m_idx_sel][m_idx_new]
            if (r_new - r_sel) % g == 0:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy: process moduli in given order, pick first valid residue for each."""
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            for r in range(m):
                if can_add(selected, r, m_idx):
                    selected.append((r, m_idx))
                    break
        return selected
    
    def try_add_to_solution(selected):
        """Try to add any missing modulus with a valid residue. Returns new solution or None."""
        selected_set = set(si[1] for si in selected)
        for m_idx in range(n_moduli):
            if m_idx in selected_set:
                continue
            m = moduli[m_idx]
            for r in range(m):
                if can_add(selected, r, m_idx):
                    return selected + [(r, m_idx)]
        return None
    
    def local_search_additions(selected, max_iter=2000):
        """Iteratively try to add more moduli."""
        best = selected[:]
        best_len = len(best)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            new_sol = try_add_to_solution(best)
            if new_sol is not None:
                best = new_sol
                best_len += 1
            else:
                break
        
        return best
    
    def local_search_swap(selected, max_iter=5000):
        """Try to improve by swapping residues to enable more additions."""
        best = selected[:]
        best_len = len(best)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new modulus
            new_sol = try_add_to_solution(best)
            if new_sol is not None:
                best = new_sol
                best_len += 1
                improved = True
                continue
            
            # Try swapping a residue to enable future additions
            for i, (r_old, m_idx) in enumerate(best):
                m = moduli[m_idx]
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    # Check if new residue is disjoint from all others
                    ok = True
                    for j, (r_j, m_idx_j) in enumerate(best):
                        if i == j:
                            continue
                        g = gcd_table[m_idx][m_idx_j]
                        if (r_new - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        # Try this swap and see if it enables an addition
                        test_best = best[:]
                        test_best[i] = (r_new, m_idx)
                        new_sol = try_add_to_solution(test_best)
                        if new_sol is not None:
                            best = new_sol
                            best_len += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy with moduli in ascending order
    order_asc = list(range(n_moduli))
    sol1 = greedy_construct(order_asc)
    sol1 = local_search_swap(sol1, max_iter=3000)
    
    if time.time() > deadline:
        return [[r, moduli[m_idx]] for r, m_idx in sol1]
    
    # Strategy 2: Greedy with moduli sorted by number of prime factors
    def num_prime_factors(n):
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
    
    order_by_pf = sorted(range(n_moduli), key=lambda i: num_prime_factors(moduli[i]))
    sol2 = greedy_construct(order_by_pf)
    sol2 = local_search_swap(sol2, max_iter=3000)
    
    if time.time() > deadline:
        best_sol = max([sol1, sol2], key=len)
        return [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Strategy 3: Greedy with moduli in descending order
    order_desc = list(range(n_moduli - 1, -1, -1))
    sol3 = greedy_construct(order_desc)
    sol3 = local_search_swap(sol3, max_iter=3000)
    
    if time.time() > deadline:
        best_sol = max([sol1, sol2, sol3], key=len)
        return [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Strategy 4: Multiple random orderings
    random.seed(42)
    best_sol = max([sol1, sol2, sol3], key=len)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        random_order = list(range(n_moduli))
        random.shuffle(random_order)
        sol_r = greedy_construct(random_order)
        sol_r = local_search_swap(sol_r, max_iter=2000)
        if len(sol_r) > len(best_sol):
            best_sol = sol_r
    
    # Final local search on the best solution
    best_sol = local_search_swap(best_sol, max_iter=5000)
    
    # Convert to required format
    result = [[r, moduli[m_idx]] for r, m_idx in best_sol]
    
    # Ensure nonempty
    if not result:
        return [[1, 2]]
    
    return result
# EVOLVE-BLOCK-END

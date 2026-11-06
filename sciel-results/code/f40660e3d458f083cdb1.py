"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
import sys

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses a model-and-optimize approach: precompute constraints, use structured search.
    """
    start_time = time.time()
    time_limit = 85.0  # Leave buffer for overhead
    
    MAX_M = 128
    MODULI = list(range(2, MAX_M + 1))
    
    # Precompute gcd table for all pairs of moduli
    gcd_table = {}
    for i, m1 in enumerate(MODULI):
        for m2 in MODULI[i:]:
            g = math.gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            if m1 != m2:
                gcd_table[(m2, m1)] = g

    def find_best_residue(m, chosen):
        """Find a residue r for m that is compatible with chosen. Returns r or None."""
        if not chosen:
            return 0
            
        forbidden = [False] * m
        for r2, m2 in chosen:
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if g == 1:
                return None # Impossible
            # Mark r such that r % g == r2 % g
            start = r2 % g
            for r in range(start, m, g):
                forbidden[r] = True
                
        for r in range(m):
            if not forbidden[r]:
                return r
        return None

    def try_add_to_solution(sol, used, m):
        """Try to add modulus m to sol. Returns True if successful."""
        # Check gcd constraints
        for r2, m2 in sol:
            if gcd_table.get((m, m2), math.gcd(m, m2)) == 1:
                return False
        # Find residue
        r = find_best_residue(m, sol)
        if r is not None:
            sol.append((r, m))
            used.add(m)
            return True
        return False

    # Initial construction: Greedy
    # Start with empty set.
    # At each step, pick the modulus m (from remaining) that minimizes calculate_cost(m, current_set).
    # If cost is inf, skip.
    
    current_set = []
    used_moduli = set()
    remaining = list(MODULI)
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
            
        best_m = None
        best_cost = float('inf')
        
        for m in remaining:
            if m in used_moduli:
                continue
            # Cost: sum of m // gcd(m, m2) for all m2 in current_set
            # If any gcd is 1, cost is inf
            cost = 0
            for r2, m2 in current_set:
                g = gcd_table.get((m, m2), math.gcd(m, m2))
                if g == 1:
                    cost = float('inf')
                    break
                cost += m // g
            
            if cost < best_cost:
                best_cost = cost
                best_m = m
        
        if best_m is None or best_cost == float('inf'):
            break
            
        r = find_best_residue(best_m, current_set)
        if r is not None:
            current_set.append((r, best_m))
            used_moduli.add(best_m)
            remaining.remove(best_m)
        else:
            remaining.remove(best_m)
            
    best_solution = list(current_set)
    best_size = len(best_solution)
    
    # Local Search
    for iteration in range(20000):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        # Phase 1: Try to add any unused modulus
        unused = [m for m in MODULI if m not in used_moduli]
        random.shuffle(unused)
        for m in unused:
            if try_add_to_solution(best_solution, used_moduli, m):
                best_size = len(best_solution)
                improved = True
                break
                
        if improved:
            continue
            
        # Phase 2: Try to swap (remove 1, add 1 or 2)
        if len(best_solution) > 0:
            idx = random.randint(0, len(best_solution) - 1)
            removed_pair = best_solution.pop(idx)
            removed_m = removed_pair[1]
            used_moduli.discard(removed_m)
            
            candidates = [m for m in MODULI if m not in used_moduli]
            random.shuffle(candidates)
            
            added_count = 0
            for m in candidates:
                if added_count >= 2:
                    break
                if try_add_to_solution(best_solution, used_moduli, m):
                    added_count += 1
                    
            if added_count >= 1:
                best_size = len(best_solution)
                improved = True
            else:
                best_solution.append(removed_pair)
                used_moduli.add(removed_m)
    
    return [[r, m] for r, m in best_solution]

# EVOLVE-BLOCK-END

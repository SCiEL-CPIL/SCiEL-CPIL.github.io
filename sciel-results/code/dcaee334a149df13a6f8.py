"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

MAX_MODULUS = 128
TIME_LIMIT = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Moduli we can choose from: 2..128
    moduli = list(range(2, MAX_MODULUS + 1))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * (n_moduli + 1) for _ in range(n_moduli + 1)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def can_add(residue, modulus, idx, selected):
        """Check if we can add [residue, modulus] to the selected set."""
        for r2, m2_idx in selected:
            m2 = moduli[m2_idx]
            g = gcd_table[idx][m2_idx]
            if (residue - r2) % g == 0:
                return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an order of moduli."""
        selected = []
        for idx in order:
            m = moduli[idx]
            # Try residues in random order
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if can_add(r, m, idx, selected):
                    selected.append((r, idx))
                    break
        return selected
    
    def local_search(initial_solution, time_budget):
        """Improve a solution via local search."""
        if not initial_solution:
            return []
        
        selected = list(initial_solution)
        selected_moduli = set(idx for _, idx in selected)
        
        # Find a better solution
        while time.time() < time_budget:
            improved = False
            # Try swapping: remove one, add one new
            for remove_idx in range(len(selected)):
                r_rem, m_rem_idx = selected[remove_idx]
                m_rem = moduli[m_rem_idx]
                
                # Try adding a new modulus
                for new_idx in range(n_moduli):
                    if new_idx in selected_moduli:
                        continue
                    m_new = moduli[new_idx]
                    
                    # Check if new can be added to remaining
                    remaining = [s for i, s in enumerate(selected) if i != remove_idx]
                    
                    residues = list(range(m_new))
                    random.shuffle(residues)
                    
                    for r_new in residues:
                        if can_add(r_new, m_new, new_idx, remaining):
                            # Check if we can re-add the removed one
                            new_selected = remaining + [(r_new, new_idx)]
                            remaining2 = [s for s in new_selected if s[1] != new_idx]
                            
                            residues_rem = list(range(m_rem))
                            random.shuffle(residues_rem)
                            
                            for r_rem_new in residues_rem:
                                if can_add(r_rem_new, m_rem, m_rem_idx, remaining2):
                                    # Full swap
                                    final_selected = remaining2 + [(r_rem_new, m_rem_idx), (r_new, new_idx)]
                                    selected = final_selected
                                    selected_moduli = set(idx for _, idx in selected)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try adding a new modulus without removing
                for new_idx in range(n_moduli):
                    if new_idx in selected_moduli:
                        continue
                    m_new = moduli[new_idx]
                    residues = list(range(m_new))
                    random.shuffle(residues)
                    for r_new in residues:
                        if can_add(r_new, m_new, new_idx, selected):
                            selected.append((r_new, new_idx))
                            selected_moduli.add(new_idx)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return selected
    
    best_solution = []
    best_count = 0
    
    # Strategy 1: Multiple greedy restarts with different orderings
    deadline = start_time + TIME_LIMIT
    
    # First, try a deterministic greedy with priority to "harder" moduli
    # Moduli with more divisors (less prime) are harder to fit
    def difficulty_score(m):
        """Score moduli by how many other moduli share a factor with them."""
        count = 0
        for i in range(n_moduli):
            if i != moduli.index(m) if m in moduli else -1:
                g = math.gcd(m, moduli[i])
                if g > 1:
                    count += 1
        return count
    
    # Sort by difficulty (most constrained first)
    moduli_index_map = {m: i for i, m in enumerate(moduli)}
    difficulty_order = sorted(range(n_moduli), key=lambda i: -difficulty_score(moduli[i]))
    
    # Try deterministic greedy first
    sol = greedy_construct(difficulty_order)
    if len(sol) > best_count:
        best_count = len(sol)
        best_solution = sol
    
    # Try randomized restarts
    attempts = 0
    while time.time() < deadline - 5.0:  # Leave 5s for local search
        attempts += 1
        order = list(range(n_moduli))
        random.shuffle(order)
        sol = greedy_construct(order)
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = sol
        
        # Occasional local search on good solutions
        if len(sol) >= best_count - 1 and time.time() < deadline - 3.0:
            improved_sol = local_search(sol, time.time() + min(2.0, deadline - time.time() - 1.0))
            if len(improved_sol) > best_count:
                best_count = len(improved_sol)
                best_solution = improved_sol
    
    # Final local search on best solution
    if best_solution and time.time() < deadline - 1.0:
        improved_sol = local_search(best_solution, deadline - 0.5)
        if len(improved_sol) > best_count:
            best_count = len(improved_sol)
            best_solution = improved_sol
    
    # Convert to required format
    if not best_solution:
        # Fallback: seed solution
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    result = [[r, moduli[idx]] for r, idx in best_solution]
    
    # Verify the solution
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            g = math.gcd(m1, m2)
            if (r1 - r2) % g == 0:
                # Invalid - should not happen, but return seed if it does
                return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return result
# EVOLVE-BLOCK-END

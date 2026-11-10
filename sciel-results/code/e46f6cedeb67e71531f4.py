"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def is_valid(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m_j, r_j) in selected.
        """
        for m_j, r_j in selected:
            g = gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True

    def find_valid_residue(selected, m):
        """
        Find a valid residue for modulus m given the selected set.
        Returns the residue or None if no valid residue exists.
        """
        for r in range(m):
            if is_valid(selected, m, r):
                return r
        return None

    def greedy_pass(seed=None):
        """
        Greedy: process moduli in order of decreasing weight (smallest m first).
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random(42)
        
        selected = []
        moduli = list(range(20, 201))
        
        # Process in order of smallest m first (highest 1/m)
        for m in moduli:
            if time.time() > deadline - 10.0:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((m, r))
        
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    def local_search(selected, max_iters=100):
        """
        Local search: try to add moduli not in the set, or swap.
        """
        selected = list(selected)
        current_score = sum(1.0 / m for m, r in selected)
        m_set = {m for m, r in selected}
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try to add any missing modulus
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                r = find_valid_residue(selected, m_add)
                if r is not None:
                    selected.append((m_add, r))
                    m_set.add(m_add)
                    current_score += 1.0 / m_add
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one, try to add two or more
                for idx in range(len(selected)):
                    if time.time() > deadline - 5.0:
                        return selected, current_score
                    
                    m_rem, r_rem = selected[idx]
                    others = selected[:idx] + selected[idx+1:]
                    others_m_set = {m for m, r in others}
                    
                    best_new_score = current_score - 1.0 / m_rem
                    best_new_selected = None
                    
                    for m_add in range(20, 201):
                        if m_add in others_m_set:
                            continue
                        r = find_valid_residue(others, m_add)
                        if r is not None:
                            new_selected = others + [(m_add, r)]
                            new_score = sum(1.0 / m for m, r in new_selected)
                            if new_score > best_new_score + 1e-12:
                                best_new_score = new_score
                                best_new_selected = new_selected
                    
                    if best_new_selected is not None and best_new_score > current_score + 1e-12:
                        selected = best_new_selected
                        current_score = best_new_score
                        m_set = {m for m, r in selected}
                        improved = True
                        break
        
        return selected, current_score

    best_score = 0.0
    best_solution = []

    # Run greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        sol, score = greedy_pass(seed=seed)
        if score > best_score:
            best_score = score
            best_solution = sol

    # Apply local search to the best solution
    if best_solution and time.time() < deadline - 10.0:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    return [list(x) for x in best_solution]
# EVOLVE-BLOCK-END

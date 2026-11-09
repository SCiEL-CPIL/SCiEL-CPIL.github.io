import time
import math
from math import gcd
from itertools import combinations
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def _is_valid_solution(solution):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _greedy_on_clique(clique_moduli, deadline, num_restarts=1000):
    """
    Greedily assign residues to a set of moduli (clique).
    Returns the best solution found.
    """
    best_solution = []
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle the order of moduli to explore different assignments
        moduli_order = list(clique_moduli)
        random.shuffle(moduli_order)
        
        current_solution = []
        
        for m in moduli_order:
            if time.time() > deadline:
                break
                
            # Try to find a residue r for m that is compatible with all in current_solution
            # Shuffle residues to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                compatible = True
                for r_prev, m_prev in current_solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    current_solution.append([r, m])
                    found = True
                    break
            
            # If no residue found, we just skip this modulus for this restart
        
        if len(current_solution) > len(best_solution):
            best_solution = [list(x) for x in current_solution]
            
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Generate cliques: multiples of small primes
    # The largest clique is multiples of 2
    cliques = []
    for p in [2, 3, 5, 7, 11, 13]:
        clique = [m for m in range(2, MAXIMUM + 1) if m % p == 0]
        if len(clique) > len(best_solution):
             cliques.append(clique)
    
    # Sort cliques by size descending to try the best ones first
    cliques.sort(key=lambda x: len(x), reverse=True)
    
    # Strategy 1: Greedy on the largest cliques
    for clique in cliques:
        if time.time() > deadline:
            break
        sol = _greedy_on_clique(clique, deadline, num_restarts=500)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    if not best_solution:
        return [[0, 2]]
        
    return best_solution

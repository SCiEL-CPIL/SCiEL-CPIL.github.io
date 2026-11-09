import time
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 88.0

def _check_pair(r1, m1, r2, m2):
    g = gcd(m1, m2)
    return r1 % g != r2 % g

def _is_valid_solution(solution):
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
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Candidate moduli: All even numbers
    # This ensures no two are coprime, which is a necessary condition for disjointness.
    # If two moduli are coprime, they always intersect.
    moduli = list(range(2, MAXIMUM + 1, 2))
    
    best_solution = []
    
    # Strategy: Randomized Local Search
    # We want to assign residues to all moduli in 'moduli'
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Initialize residues
        # Using 1 is a good start? Or random?
        # Let's try random
        residues = {m: random.randint(0, m - 1) for m in moduli}
        
        # Local search
        current_solution = [[residues[m], m] for m in moduli]
        
        # Check if valid
        if _is_valid_solution(current_solution):
            best_solution = current_solution
            break
        
        # If not, try to fix conflicts
        # We'll do a simple iterative improvement
        # Find a conflict, change a residue, repeat
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Find a conflicting pair
            conflict_found = False
            for i in range(len(current_solution)):
                for j in range(i + 1, len(current_solution)):
                    r1, m1 = current_solution[i]
                    r2, m2 = current_solution[j]
                    if not _check_pair(r1, m1, r2, m2):
                        # Conflict!
                        # Randomly change r1 or r2
                        if random.random() < 0.5:
                            new_r = random.randint(0, m1 - 1)
                            if new_r != r1:
                                current_solution[i][0] = new_r
                                conflict_found = True
                                break
                        else:
                            new_r = random.randint(0, m2 - 1)
                            if new_r != r2:
                                current_solution[j][0] = new_r
                                conflict_found = True
                                break
                if conflict_found:
                    break
            
            if not conflict_found:
                # No conflicts, valid solution
                if len(current_solution) > len(best_solution):
                    best_solution = current_solution
                break

    # If we didn't find a full solution, try to find a valid subset
    # Greedy approach on the moduli
    if len(best_solution) < len(moduli):
        solution = []
        used_moduli = set()
        # Try to add moduli one by one
        # Order: maybe larger moduli first? Or smaller?
        # Smaller moduli have fewer residue choices, so maybe harder?
        # Let's try random order
        random.shuffle(moduli)
        
        for m in moduli:
            if time.time() > deadline:
                break
            found = False
            for _ in range(100): # Try 100 random residues
                r = random.randint(0, m - 1)
                compatible = True
                for (r_prev, m_prev) in solution:
                    if not _check_pair(r, m, r_prev, m_prev):
                        compatible = False
                        break
                if compatible:
                    solution.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
            if not found:
                # Try all residues
                for r in range(m):
                    compatible = True
                    for (r_prev, m_prev) in solution:
                        if not _check_pair(r, m, r_prev, m_prev):
                            compatible = False
                            break
                    if compatible:
                        solution.append([r, m])
                        used_moduli.add(m)
                        found = True
                        break
        
        if len(solution) > len(best_solution):
            best_solution = solution

    if not best_solution:
        return [[0, 2]]
    
    return best_solution

import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 55.0
    
    # Candidate moduli: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    
    best_solution = []
    best_count = 0
    
    # Helper to check if a full assignment is valid
    def is_valid_assignment(assignments):
        # assignments is list of (r, m)
        for i in range(len(assignments)):
            r_i, m_i = assignments[i]
            for j in range(i+1, len(assignments)):
                r_j, m_j = assignments[j]
                g = math.gcd(m_i, m_j)
                if (r_i - r_j) % g == 0:
                    return False
        return True

    # CSP Solver: Try to find a valid assignment for a given set of moduli
    # Uses backtracking with MRV and forward checking
    def solve_csp(moduli_set, time_limit=1.0):
        start = time.time()
        # Order moduli by size? Smaller moduli have smaller domains, so MRV suggests starting with smallest.
        # But larger moduli are less constrained by gcd? 
        # Let's sort by modulus size ascending.
        sorted_moduli = sorted(moduli_set)
        
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start > time_limit:
                return None
            
            if idx == len(sorted_moduli):
                return assignment.copy()
                
            m = sorted_moduli[idx]
            # Try residues in random order or natural order?
            # Natural order might be deterministic. Random might help escape local traps?
            # But we want to find ANY solution.
            # Let's try residues 0..m-1
            
            for r in range(m):
                # Check compatibility with already assigned
                compatible = True
                for m2, r2 in assignment.items():
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        compatible = False
                        break
                
                if compatible:
                    assignment[m] = r
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    del assignment[m]
                    
            return None
            
        return backtrack(0)

    # Greedy with CSP solving
    # We try to add moduli one by one.
    # To maximize, we should probably try to add "hard" moduli first? Or "easy"?
    # If we add easy ones first, we might block hard ones.
    # If we add hard ones first, we have more freedom for easy ones.
    # So sort moduli by "difficulty". 
    # Difficulty could be the number of conflicts with other potential moduli?
    # Or just size? Small moduli are "hard" (small domain).
    # So let's try to add small moduli first?
    # Wait, if I add 2, 4, 6, I get stuck.
    # If I add 128, 126, ..., I might get stuck later.
    
    # Let's try multiple strategies:
    # 1. Ascending order
    # 2. Descending order
    # 3. Random order
    
    for strategy in ['asc', 'desc', 'rand']:
        current_moduli = []
        current_solution = {}
        
        candidates = moduli.copy()
        if strategy == 'asc':
            candidates.sort()
        elif strategy == 'desc':
            candidates.sort(reverse=True)
        else:
            random.shuffle(candidates)
            
        for m in candidates:
            # Try to add m to current set
            new_moduli = current_moduli + [m]
            # Solve CSP for new_moduli
            # Use a short time limit for each step to keep total time low
            # If we have 64 moduli, and each takes 0.1s, that's 6.4s.
            # We have 55s. So 0.8s per step is safe.
            sol = solve_csp(new_moduli, time_limit=0.5)
            
            if sol is not None:
                current_moduli.append(m)
                current_solution = sol
            else:
                # Skip m
                pass
                
        # Check if this solution is better
        if len(current_moduli) > best_count:
            best_count = len(current_moduli)
            best_solution = current_solution
            
        if best_count == len(moduli):
            break
            
    # Convert to list of lists
    return [[r, m] for m, r in best_solution.items()]
# EVOLVE-BLOCK-END

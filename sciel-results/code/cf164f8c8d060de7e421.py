import random
import time
import math
from fractions import Fraction

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds to be safe
    
    best_score = 0.0
    best_solution = None
    all_moduli = list(range(20, 201))
    
    # Precompute gcds for speed? 
    # gcd is fast in Python 3.9+
    
    def is_valid_solution(sol):
        """Check if the solution is pairwise disjoint."""
        n = len(sol)
        for i in range(n):
            m1, r1 = sol[i]
            for j in range(i + 1, n):
                m2, r2 = sol[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def calculate_score(sol):
        return sum(1.0 / m for m, r in sol)

    def find_valid_residue(m, current_sol):
        """Find a residue r for modulus m such that (m, r) is disjoint from all in current_sol."""
        # Try residues 0 to m-1
        # Optimization: if m is large, this is slow. But m <= 200.
        for r in range(m):
            valid = True
            for m2, r2 in current_sol:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                return r
        return -1

    def greedy_init(seed):
        random.seed(seed)
        sol = []
        # Sort moduli by ascending value to prioritize high 1/m
        # But maybe we should be strategic.
        # Let's try adding moduli in increasing order.
        for m in all_moduli:
            # Try to add m
            # We want to pick a residue that is "least constraining" for future smaller moduli?
            # Actually, future moduli are larger, so they have more residues.
            # Constraints from small moduli are tighter (smaller gcds usually? No, gcd(m_small, m_large) can be anything).
            # Let's just pick the first valid residue found.
            r = find_valid_residue(m, sol)
            if r != -1:
                sol.append([m, r])
        
        # Now try to improve by swapping in smaller moduli if we missed any?
        # Greedy usually takes all small ones.
        return sol

    # Initial greedy run
    initial_sol = greedy_init(42)
    if not is_valid_solution(initial_sol):
        # Fallback to seed
        initial_sol = [[20 * j, j - 1] for j in range(1, 11)]
    
    current_sol = [list(x) for x in initial_sol]
    current_score = calculate_score(current_sol)
    
    if current_score > best_score:
        best_score = current_score
        best_solution = [list(x) for x in current_sol]

    # Simulated Annealing
    # We want to maximize score.
    # Moves:
    # 1. Change residue of a random element
    # 2. Swap residues of two random elements
    # 3. Add a random unused modulus
    # 4. Remove a random element
    # 5. Swap moduli of two random elements (if residues fit)
    
    def get_unused_moduli(sol):
        used = {m for m, r in sol}
        return [m for m in all_moduli if m not in used]

    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.0
            # Random restart with perturbation
            # Remove a random subset of elements to allow reshuffling
            if len(current_sol) > 10:
                for _ in range(5):
                    idx = random.randint(0, len(current_sol) - 1)
                    current_sol.pop(idx)
                current_score = calculate_score(current_sol)
            # Re-greedy fill
            # Sort current sol by m
            current_sol.sort(key=lambda x: x[0])
            # Try to add back small moduli
            used = {m for m, r in current_sol}
            for m in all_moduli:
                if m not in used:
                    r = find_valid_residue(m, current_sol)
                    if r != -1:
                        current_sol.append([m, r])
            
            if current_score > best_score:
                best_score = current_score
                best_solution = [list(x) for x in current_sol]
                # If we found a very good one, break? No, keep polishing.
        
        # Pick a move
        move_type = random.random()
        if move_type < 0.3:
            # Change residue
            if not current_sol: continue
            idx = random.randint(0, len(current_sol) - 1)
            m, r = current_sol[idx]
            # Try a few random residues
            for _ in range(10):
                new_r = random.randint(0, m - 1)
                if new_r == r: continue
                # Check validity
                valid = True
                for j, (m2, r2) in enumerate(current_sol):
                    if j == idx: continue
                    g = math.gcd(m, m2)
                    if new_r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    old_r = current_sol[idx][1]
                    current_sol[idx][1] = new_r
                    # Score doesn't change with residue change
                    # But it might enable future moves.
                    # Accept always? Yes, it's a neutral move.
                    break
        elif move_type < 0.5:
            # Swap residues
            if len(current_sol) < 2: continue
            i, j = random.sample(range(len(current_sol)), 2)
            m1, r1 = current_sol[i]
            m2, r2 = current_sol[j]
            # Swap
            current_sol[i][1] = r2
            current_sol[j][1] = r1
            # Check validity
            valid = True
            # Check pair (i, j)
            g = math.gcd(m1, m2)
            if r2 % g == r1 % g:
                valid = False
            # Check i against others
            if valid:
                for k, (mk, rk) in enumerate(current_sol):
                    if k == i or k == j: continue
                    g_ik = math.gcd(m1, mk)
                    if r2 % g_ik == rk % g_ik:
                        valid = False
                        break
            if valid:
                for k, (mk, rk) in enumerate(current_sol):
                    if k == i or k == j: continue
                    g_jk = math.gcd(m2, mk)
                    if r1 % g_jk == rk % g_jk:
                        valid = False
                        break
            if not valid:
                # Revert
                current_sol[i][1] = r1
                current_sol[j][1] = r2
            else:
                # Accept (neutral move)
                pass
        elif move_type < 0.7:
            # Add modulus
            unused = get_unused_moduli(current_sol)
            if unused:
                m = random.choice(unused)
                r = find_valid_residue(m, current_sol)
                if r != -1:
                    current_sol.append([m, r])
                    new_score = calculate_score(current_sol)
                    # Accept if better or with probability exp(-delta/T) if worse?
                    # Adding always improves score (1/m > 0). So always accept if valid.
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = [list(x) for x in current_sol]
        elif move_type < 0.85:
            # Remove modulus
            if current_sol:
                idx = random.randint(0, len(current_sol) - 1)
                removed = current_sol.pop(idx)
                current_score = calculate_score(current_sol)
                # This worsens score, but might allow adding better ones later
                # We don't update best here
        else:
            # Swap moduli
            if len(current_sol) < 2: continue
            i, j = random.sample(range(len(current_sol)), 2)
            m1, r1 = current_sol[i]
            m2, r2 = current_sol[j]
            
            # We swap moduli, keep residues?
            # New pair i: (m2, r1), New pair j: (m1, r2)
            # Must have r1 < m2 and r2 < m1
            if r1 < m2 and r2 < m1:
                # Check validity of new config
                # Temporarily swap
                current_sol[i][0] = m2
                current_sol[j][0] = m1
                
                valid = True
                # Check i against all
                for k, (mk, rk) in enumerate(current_sol):
                    if k == i: continue
                    g = math.gcd(m2, mk)
                    if r1 % g == rk % g:
                        valid = False
                        break
                if valid:
                    for k, (mk, rk) in enumerate(current_sol):
                        if k == j: continue
                        g = math.gcd(m1, mk)
                        if r2 % g == rk % g:
                            valid = False
                            break
                
                if not valid:
                    # Revert
                    current_sol[i][0] = m1
                    current_sol[j][0] = m2
                else:
                    new_score = calculate_score(current_sol)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_solution = [list(x) for x in current_sol]
                    else:
                        # Revert
                        current_sol[i][0] = m1
                        current_sol[j][0] = m2
            else:
                # Residues don't fit, try to adjust residues? 
                # Skip for now
                pass

        T *= cooling
        
        # If we have a very good score, we might want to tighten search
        # But let it run.

    # Final polish: try to add any remaining moduli
    if best_solution:
        used = {m for m, r in best_solution}
        for m in all_moduli:
            if m not in used:
                r = find_valid_residue(m, best_solution)
                if r != -1:
                    best_solution.append([m, r])
                    best_score = calculate_score(best_solution)

    return best_solution

# EVOLVE-BLOCK-END

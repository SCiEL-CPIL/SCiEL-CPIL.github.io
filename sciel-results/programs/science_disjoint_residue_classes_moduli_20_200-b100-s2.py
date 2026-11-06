import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 110s to be safe
    deadline = time.time() + 110
    
    max_m = 200
    min_m = 20
    
    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Helper to check if (m, r) is valid against a list of selected (m, r)
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    # Greedy initialization
    def greedy_init():
        selected = []
        for m in range(min_m, max_m + 1):
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
        return selected

    # Simulated Annealing
    def simulated_annealing(initial_solution):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-6
        
        selected_set = set((m, r) for m, r in current_sol)
        # To speed up random selection of unselected moduli, we can keep a list
        # But we need to check validity anyway.
        
        while time.time() < deadline:
            if T < min_T:
                break
                
            # Generate a neighbor
            move_type = random.random()
            
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False
            
            if move_type < 0.7:
                # Add a modulus
                # Choose a random unselected modulus, biased towards small m
                # We can just pick a random m from 20..200 and check if it's already there
                m_add = random.randint(min_m, max_m)
                if any(m == m_add for m, _ in new_sol):
                    continue
                
                # Try a random valid residue
                # To be faster, we can try a few random r's
                r_add = random.randint(0, m_add - 1)
                if is_valid(m_add, r_add, new_sol):
                    new_sol.append([m_add, r_add])
                    new_score += 1.0 / m_add
                    valid_move = True
            elif move_type < 0.9:
                # Remove a modulus
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            else:
                # Change residue
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                
                # Try a few random new residues
                for _ in range(5):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    # Check if new_r is valid for m_chg against all OTHERS
                    # We must check against all (m2, r2) where m2 != m_chg
                    temp_sol = [x for i, x in enumerate(new_sol) if i != idx]
                    if is_valid(m_chg, new_r, temp_sol):
                        # Update score? Score doesn't change for residue change
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            
            if not valid_move:
                continue
                
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_sol = new_sol
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]
            
            T *= cooling
            
        return best_sol

    # Run multiple restarts
    best_solution = []
    best_score = 0.0
    
    # 1. Deterministic Greedy
    sol = greedy_init()
    sol = simulated_annealing(sol)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
        
    # 2. Randomized Greedy Restarts
    random.seed(42)
    while time.time() < deadline - 5:
        # Randomized greedy: pick random valid r instead of first
        sol = []
        for m in range(min_m, max_m + 1):
            valid_rs = [r for r in range(m) if is_valid(m, r, sol)]
            if valid_rs:
                sol.append([m, random.choice(valid_rs)])
        
        sol = simulated_annealing(sol)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

    return best_solution

# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")

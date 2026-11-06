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
    # idx_to_skip: index in selected to ignore (for swap/residue change)
    def is_valid(m, r, selected, idx_to_skip=-1):
        for i, (m2, r2) in enumerate(selected):
            if i == idx_to_skip:
                continue
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    # Greedy initialization: Pick smallest m with any valid r
    # Randomized version to allow different starts
    def greedy_init(randomize=False):
        selected = []
        for m in range(min_m, max_m + 1):
            valid_rs = []
            for r in range(m):
                if is_valid(m, r, selected):
                    valid_rs.append(r)
            if valid_rs:
                if randomize:
                    selected.append([m, random.choice(valid_rs)])
                else:
                    selected.append([m, valid_rs[0]])
        return selected

    # Simulated Annealing
    def simulated_annealing(initial_solution):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        T = 0.5 # Start with higher temperature to explore more
        cooling = 0.999
        min_T = 1e-8
        
        # Precompute list of all possible moduli
        all_moduli = list(range(min_m, max_m + 1))
        
        while time.time() < deadline:
            if T < min_T:
                break
                
            move_type = random.random()
            
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False
            
            if move_type < 0.6:
                # Add a modulus
                selected_ms = set(m for m, _ in new_sol)
                unselected = [m for m in all_moduli if m not in selected_ms]
                if not unselected:
                    continue
                
                # Bias towards smaller moduli: weight = 1/m
                # We want to pick small m.
                # Let's just pick the smallest unselected modulus with high probability.
                if random.random() < 0.5:
                    m_add = min(unselected)
                else:
                    # Weighted random
                    weights = [1.0/m for m in unselected]
                    # Normalize not needed for random.choices
                    m_add = random.choices(unselected, weights=weights)[0]
                
                # Find a valid residue
                r_add = None
                # Try random residues first
                for _ in range(5):
                    r_try = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_try, new_sol):
                        r_add = r_try
                        break
                
                if r_add is None:
                    # Exhaustive search for a valid r
                    for r_try in range(m_add):
                        if is_valid(m_add, r_try, new_sol):
                            r_add = r_try
                            break
                
                if r_add is not None:
                    new_sol.append([m_add, r_add])
                    new_score += 1.0 / m_add
                    valid_move = True
            elif move_type < 0.8:
                # Remove a modulus
                if not new_sol:
                    continue
                # Bias towards removing larger moduli (less value)
                # Sort by m descending, pick from top half?
                # Or just random
                idx = random.randint(0, len(new_sol) - 1)
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            else:
                # Change residue of an existing modulus
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                
                # Try to find a different valid residue
                new_r = None
                for _ in range(10):
                    r_try = random.randint(0, m_chg - 1)
                    if r_try == old_r:
                        continue
                    if is_valid(m_chg, r_try, new_sol, idx):
                        new_r = r_try
                        break
                
                if new_r is not None:
                    new_sol[idx] = [m_chg, new_r]
                    valid_move = True
            
            if not valid_move:
                continue
                
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
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
    
    # 1. Deterministic Greedy + SA
    sol = greedy_init(randomize=False)
    sol = simulated_annealing(sol)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
        
    # 2. Randomized Greedy Restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 2:
        # Randomized greedy: pick random valid r
        sol = greedy_init(randomize=True)
        
        sol = simulated_annealing(sol)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
        restarts += 1
        
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

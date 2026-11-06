import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 115s to be safe
    deadline = time.time() + 115
    
    # Precompute gcd table for m in [20, 200]
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Helper function to check if adding (m, r) is valid given current set
    def is_valid_add(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    # Helper function to calculate score
    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)
    
    # Multi-start greedy with local search
    best_score = 0.0
    best_solution = []
    
    # Greedy fill with optional randomization of residue choice
    def greedy_fill(order, randomize_residue=False):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_rs = []
            for r in range(m):
                if is_valid_add(m, r, selected):
                    if randomize_residue:
                        valid_rs.append(r)
                    else:
                        selected.append([m, r])
                        break
            if randomize_residue and valid_rs:
                selected.append([m, random.choice(valid_rs)])
        return selected

    # Local search: try to improve the solution
    def local_search(selected):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = calc_score(selected)
            
            # Sort selected by m descending (largest m first to remove)
            sorted_selected = sorted(selected, key=lambda x: -x[0])
            
            for i in range(len(sorted_selected)):
                if time.time() > deadline:
                    break
                m_remove, r_remove = sorted_selected[i]
                
                # Create a new list without this element
                new_selected = [x for x in selected if x[0] != m_remove]
                
                # Try to add any m that is smaller than m_remove and not in new_selected
                # and is valid
                added_any = False
                present_ms = set(m for m, _ in new_selected)
                
                # Check candidates in increasing order
                for m_add in range(20, m_remove):
                    if m_add in present_ms:
                        continue
                    # Find a valid r for m_add
                    found_r = None
                    for r in range(m_add):
                        if is_valid_add(m_add, r, new_selected):
                            found_r = r
                            break
                    if found_r is not None:
                        new_selected.append([m_add, found_r])
                        added_any = True
                        break # Only add the best one (smallest m)
                        
                if added_any:
                    new_score = calc_score(new_selected)
                    if new_score > current_score + 1e-9:
                        selected = new_selected
                        improved = True
                        break # Restart the while loop
            
            if not improved:
                break
        
        return selected

    # Main loop: Multi-start
    # 1. Deterministic increasing order (baseline)
    order = list(range(20, 201))
    sol = greedy_fill(order, randomize_residue=False)
    sol = local_search(sol)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
    
    # 2. Random restarts
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Generate a random permutation with noise favoring small m
        order = list(range(20, 201))
        noisy_order = sorted(order, key=lambda m: m + random.uniform(0, 100))
        
        sol = greedy_fill(noisy_order, randomize_residue=True)
        sol = local_search(sol)
        
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

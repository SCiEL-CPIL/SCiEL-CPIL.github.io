import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 115s to be safe
    deadline = time.time() + 115
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    moduli = list(range(min_m, max_m + 1))
    
    # Helper to check validity of (m, r) against a set of (m, r) pairs
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(m, selected):
        valid_rs = []
        for r in range(m):
            if is_valid(m, r, selected):
                valid_rs.append(r)
        return valid_rs

    def calculate_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_construct(randomize=False):
        selected = []
        # Process moduli in ascending order (highest weight first)
        for m in moduli:
            valid_rs = get_valid_residues(m, selected)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    # Heuristic: pick r that minimizes conflicts with next 5 moduli
                    best_r = valid_rs[0]
                    min_conflicts = float('inf')
                    for r in valid_rs:
                        conflicts = 0
                        next_moduli = [mm for mm in moduli if mm > m][:5]
                        for mm in next_moduli:
                            g = gcd_table[m][mm]
                            conflicts += mm // g
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            best_r = r
                    r = best_r
                selected.append((m, r))
        return selected

    def local_search(initial_selected):
        selected = [tuple(p) for p in initial_selected]
        current_score = calculate_score(selected)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any unselected modulus
            for m in moduli:
                if m in [s[0] for s in selected]:
                    continue
                valid_rs = get_valid_residues(m, selected)
                if valid_rs:
                    new_selected = selected + [(m, valid_rs[0])]
                    new_score = current_score + 1.0 / m
                    if new_score > current_score:
                        selected = new_selected
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue

            # 2. Try to swap a selected modulus with a smaller unselected one
            selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
            
            swap_found = False
            for m_old, r_old in selected_sorted:
                for m_new in moduli:
                    if m_new >= m_old:
                        break
                    if m_new in [s[0] for s in selected]:
                        continue
                    
                    temp_selected = [s for s in selected if s[0] != m_old]
                    valid_rs = get_valid_residues(m_new, temp_selected)
                    if valid_rs:
                        new_score = current_score - 1.0 / m_old + 1.0 / m_new
                        if new_score > current_score:
                            selected = temp_selected + [(m_new, valid_rs[0])]
                            current_score = new_score
                            improved = True
                            swap_found = True
                            break
                if swap_found:
                    break
            
            if improved:
                continue
                
            # 3. Random kick
            if time.time() < deadline - 10:
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected = [s for s in selected if s != removed]
                    current_score = calculate_score(selected)
                    improved = True

        return selected, current_score

    def try_fill_all_randomized():
        # Inspiration 1: Try to find a valid residue assignment for ALL moduli
        # This is the optimal solution if feasible.
        # Use randomized greedy with restarts.
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
                
            sol = {}
            success = True
            
            # Iterate from smallest to largest m (most constrained first)
            for m in range(min_m, max_m + 1):
                # Generate random permutation of residues
                residues = list(range(m))
                random.shuffle(residues)
                
                assigned = False
                for r in residues:
                    valid = True
                    for m2, r2 in sol.items():
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            valid = False
                            break
                            
                    if valid:
                        sol[m] = r
                        assigned = True
                        break
                        
                if not assigned:
                    success = False
                    break
            
            if success:
                return [[m, r] for m, r in sol.items()]
                
        return None

    best_solution = []
    best_score = 0.0
    
    # Phase 1: Try to get the full set (optimal if feasible)
    full_sol = try_fill_all_randomized()
    if full_sol:
        # Verify and score
        # It's guaranteed valid by construction, but let's double check score
        score = calculate_score(full_sol)
        if score > best_score:
            best_score = score
            best_solution = full_sol
            
    # Phase 2: If full set not found or we want to improve (though full set is max possible score sum 1/m),
    # we run the Greedy + Local Search approach which might find a high-scoring subset 
    # if the full set is infeasible, or confirm the full set.
    # Actually, if full set is feasible, it IS the optimal solution because all weights are positive.
    # So if we found it, we are done.
    # If we didn't find it, it might be infeasible, or just hard to find.
    # The greedy+local search approach finds the best subset.
    
    if not best_solution:
        # Run multiple iterations of Greedy + Local Search
        num_restarts = 50
        
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            is_rand = (i % 2 == 1)
            initial = greedy_construct(randomize=is_rand)
            
            sol, score = local_search(initial)
            
            if score > best_score:
                best_score = score
                best_solution = sol

    # Convert to required format
    if best_solution:
        return [[m, r] for m, r in best_solution]
    
    # Fallback: simple greedy
    initial = greedy_construct(randomize=False)
    return [[m, r] for m, r in initial]

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

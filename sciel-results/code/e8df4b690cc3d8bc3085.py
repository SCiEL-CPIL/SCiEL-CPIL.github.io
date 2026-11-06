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

    def find_residues(moduli):
        """
        Backtracking solver to find valid residues for a given set of moduli.
        Returns list of [m, r] or None if infeasible.
        Optimized for speed: orders moduli by 'constraint tightness' (higher gcd with others?)
        """
        n = len(moduli)
        if n == 0:
            return []
        
        # Sort moduli to find conflicts early.
        # Heuristic: Moduli with many large gcds with others should be assigned first?
        # Or just sort by modulus size?
        # Let's sort by descending modulus size to handle larger ranges first? 
        # Actually, smaller moduli have fewer options (0..m-1), so they are more constrained.
        # Let's sort by ascending modulus size.
        sorted_indices = sorted(range(n), key=lambda i: moduli[i])
        
        assigned = [None] * n
        # conflicts[i][j] = gcd(moduli[i], moduli[j])
        
        # Precompute gcds for the specific set
        # To speed up, we can just compute on the fly or use the global table
        def backtrack(k):
            if k == n:
                return True
            
            idx = sorted_indices[k]
            m = moduli[idx]
            
            # Try all residues
            # Optimization: If we have already assigned some, we can restrict r
            # But simple iteration is fine for m <= 200
            for r in range(m):
                # Check against all previously assigned
                valid = True
                for j in range(k):
                    idx_j = sorted_indices[j]
                    m_j = moduli[idx_j]
                    r_j = assigned[idx_j]
                    g = gcd_table[m][m_j]
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    assigned[idx] = r
                    if backtrack(k + 1):
                        return True
                    assigned[idx] = None
            return False

        if backtrack(0):
            return [[moduli[i], assigned[i]] for i in range(n)]
        return None

    def calc_score(moduli):
        return sum(1.0 / m for m in moduli)

    def local_search_moduli(initial_moduli, deadline):
        """
        Local search on the set of moduli.
        Moves: Add, Remove, Swap.
        Uses find_residues to check feasibility.
        """
        current_moduli = list(initial_moduli)
        current_sol = find_residues(current_moduli)
        if not current_sol:
            return None, 0.0
            
        current_score = calc_score(current_moduli)
        best_moduli = list(current_moduli)
        best_score = current_score
        best_sol = current_sol
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding a modulus
            # Prioritize small moduli
            candidates = [m for m in range(min_m, max_m + 1) if m not in current_moduli]
            random.shuffle(candidates)
            # Sort by weight desc for greedy attempt? 
            # Let's try random order to escape local optima, but bias towards small
            candidates.sort(key=lambda m: -1.0/m) 
            # Mix: take top 10, then random rest
            top_k = candidates[:10]
            rest = candidates[10:]
            random.shuffle(rest)
            candidates = top_k + rest
            
            for m_add in candidates:
                if time.time() > deadline:
                    break
                new_moduli = current_moduli + [m_add]
                new_sol = find_residues(new_moduli)
                if new_sol:
                    new_score = calc_score(new_moduli)
                    if new_score > current_score:
                        current_moduli = new_moduli
                        current_sol = new_sol
                        current_score = new_score
                        improved = True
                        # If we improved, we can break and restart the loop to re-evaluate
                        # Or continue to see if we can add more? 
                        # Greedy add: let's continue adding if possible
                        # But we need to update 'candidates' list? 
                        # For simplicity, break and restart outer loop
                        break
            
            if not improved:
                # 2. Try removing a modulus
                # Remove large moduli first (low weight)
                indices = list(range(len(current_moduli)))
                random.shuffle(indices)
                # Sort by weight asc (largest m first)
                indices.sort(key=lambda i: current_moduli[i])
                
                for idx in indices:
                    if time.time() > deadline:
                        break
                    m_rem = current_moduli[idx]
                    new_moduli = [m for i, m in enumerate(current_moduli) if i != idx]
                    new_sol = find_residues(new_moduli)
                    if new_sol:
                        new_score = calc_score(new_moduli)
                        # We only accept removal if it allows adding something better later?
                        # Or if it's a dead end? 
                        # In this simple local search, we only accept if score improves.
                        # Removing a modulus *decreases* score. So we never accept pure removal
                        # unless it enables a swap.
                        # So pure removal is not useful for maximization unless we are stuck 
                        # and want to explore a different basin.
                        # Let's implement Swap instead.
                        pass
            
            if not improved:
                # 3. Try swapping: remove one, add one
                # Random swap
                for _ in range(50): # Limit swaps per iteration
                    if time.time() > deadline:
                        break
                    if not current_moduli:
                        break
                    idx = random.randint(0, len(current_moduli) - 1)
                    m_rem = current_moduli[idx]
                    
                    # Choose m_add
                    m_add = random.randint(min_m, max_m)
                    if m_add == m_rem or m_add in current_moduli:
                        continue
                        
                    new_moduli = [m for i, m in enumerate(current_moduli) if i != idx]
                    new_moduli.append(m_add)
                    
                    new_sol = find_residues(new_moduli)
                    if new_sol:
                        new_score = calc_score(new_moduli)
                        if new_score > current_score:
                            current_moduli = new_moduli
                            current_sol = new_sol
                            current_score = new_score
                            improved = True
                            break
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_moduli = current_moduli
            best_sol = current_sol
            
        return best_sol, best_score

    def initial_solution(deadline):
        """
        Greedy construction of moduli set.
        """
        selected_moduli = []
        # Try to add moduli in increasing order of m (decreasing weight)
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            test_moduli = selected_moduli + [m]
            if find_residues(test_moduli):
                selected_moduli.append(m)
        
        return selected_moduli

    # 1. Initial Solution
    init_deadline = deadline - 30
    init_moduli = initial_solution(init_deadline)
    
    # 2. Local Search
    ls_deadline = deadline - 10
    best_sol, best_score = local_search_moduli(init_moduli, ls_deadline)
    
    # 3. Simulated Annealing on Moduli Set
    # If time permits, run SA
    if time.time() < deadline - 5:
        current_moduli = list(best_sol and [x[0] for x in best_sol] or init_moduli)
        if not current_moduli:
            current_moduli = init_moduli
            
        current_sol = find_residues(current_moduli)
        if not current_sol:
            current_sol = best_sol
            current_moduli = [x[0] for x in current_sol]
            
        current_score = calc_score(current_moduli)
        best_sa_sol = current_sol
        best_sa_score = current_score
        
        T = 0.1
        cooling = 0.999
        
        while time.time() < deadline - 5:
            if T < 1e-6:
                break
                
            # Move: Random Add, Remove, or Swap
            move_type = random.random()
            new_moduli = list(current_moduli)
            valid_move = False
            
            if move_type < 0.4:
                # Add
                m_add = random.randint(min_m, max_m)
                if m_add not in new_moduli:
                    new_moduli.append(m_add)
                    valid_move = True
            elif move_type < 0.7:
                # Remove
                if new_moduli:
                    idx = random.randint(0, len(new_moduli) - 1)
                    new_moduli.pop(idx)
                    valid_move = True
            else:
                # Swap
                if new_moduli:
                    idx = random.randint(0, len(new_moduli) - 1)
                    m_rem = new_moduli[idx]
                    m_add = random.randint(min_m, max_m)
                    if m_add not in new_moduli and m_add != m_rem:
                        new_moduli[idx] = m_add
                        valid_move = True
            
            if not valid_move:
                continue
                
            new_sol = find_residues(new_moduli)
            if new_sol:
                new_score = calc_score(new_moduli)
                delta = new_score - current_score
                if delta > 0 or (delta > -1.0 and random.random() < math.exp(delta / T)):
                    current_moduli = new_moduli
                    current_sol = new_sol
                    current_score = new_score
                    
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa_sol = current_sol
            else:
                # Infeasible: treat as very bad move
                delta = -10.0
                if random.random() < math.exp(delta / T):
                    current_moduli = new_moduli
                    current_score = -10.0 # Penalty state? 
                    # Better to just reject infeasible moves in SA for this problem
                    # Since we need a valid solution.
                    pass
            
            T *= cooling
        
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_sol = best_sa_sol

    # Final polish: Try to add any remaining moduli greedily
    if best_sol:
        final_moduli = [x[0] for x in best_sol]
        for m in range(min_m, max_m + 1):
            if m not in final_moduli:
                test = final_moduli + [m]
                sol = find_residues(test)
                if sol:
                    best_sol = sol
                    best_score = calc_score(final_moduli + [m])
                    final_moduli.append(m)

    return best_sol

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

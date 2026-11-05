"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i+1, n_moduli):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    # Precompute blocked residue masks for fast compatibility checking
    # For each pair (m, n) with m != n, and each residue class c mod gcd(m,n),
    # store a bitmask of which residues of n are blocked
    # Also precompute for each m, the list of residues 0..m-1
    residue_lists = {m: list(range(m)) for m in moduli}
    
    # Precompute: for each m, n, the gcd
    # And for each m, n, c: the set of residues in [0,n) congruent to c mod gcd(m,n)
    # We'll use bitmasks for speed
    blocked_mask_cache = {}
    for m in moduli:
        for n in moduli:
            if m == n:
                continue
            g = gcd_table.get((min(m,n), max(m,n)), math.gcd(m,n))
            for c in range(g):
                mask = 0
                for r in range(c, n, g):
                    mask |= (1 << r)
                blocked_mask_cache[(m, n, c)] = mask
    
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        a, b = min(m1, m2), max(m1, m2)
        return gcd_table.get((a, b), math.gcd(a, b))
    
    def score_solution(sol):
        return sum(Fraction(1, m) for m, r in sol)
    
    def is_compatible(m1, r1, m2, r2):
        """Check if (m1, r1) and (m2, r2) are disjoint."""
        if m1 == m2:
            return r1 != r2
        g = get_gcd(m1, m2)
        return r1 % g != r2 % g
    
    def get_valid_residues_fast(m, selected_pairs):
        """Get valid residues for modulus m given selected pairs, using bitmask for speed."""
        if not selected_pairs:
            return list(range(m))
        
        # Start with all residues valid
        valid_mask = (1 << m) - 1
        
        for m_sel, r_sel in selected_pairs:
            if m_sel == m:
                valid_mask &= ~(1 << r_sel)
            else:
                g = get_gcd(m, m_sel)
                r_mod = r_sel % g
                mask = blocked_mask_cache.get((m_sel, m, r_mod), 0)
                valid_mask &= ~mask
        
        # Convert mask to list
        valid = [r for r in range(m) if valid_mask & (1 << r)]
        return valid
    
    def greedy_construct(order, residue_strategy='first'):
        """
        Construct a solution by processing moduli in the given order.
        """
        selected = []
        
        for m in order:
            valid_residues = get_valid_residues_fast(m, selected)
            if not valid_residues:
                continue
            
            if len(valid_residues) == 1:
                r = valid_residues[0]
            elif residue_strategy == 'random':
                r = random.choice(valid_residues)
            elif residue_strategy == 'min_damage':
                # Pick residue minimizing damage to future moduli
                best_r = valid_residues[0]
                min_damage = float('inf')
                
                for r_candidate in valid_residues:
                    damage = 0
                    for n in order:
                        if n == m:
                            continue
                        if n in [mm for mm, _ in selected]:
                            continue
                        g = get_gcd(m, n)
                        r_mod = r_candidate % g
                        ap_size = n // g + (1 if n % g > r_mod else 0)
                        damage += ap_size
                        if damage >= min_damage:
                            break
                    if damage < min_damage:
                        min_damage = damage
                        best_r = r_candidate
                        if min_damage == 0:
                            break
                r = best_r
            else:
                r = valid_residues[0]
            
            selected.append([m, r])
        
        return selected
    
    def try_add_moduli(sol, unselected):
        """Try to add as many unselected moduli as possible to sol."""
        sol = [x[:] for x in sol]
        sel_set = set(m for m, _ in sol)
        
        # Sort unselected by reciprocal (largest first) for better packing
        unselected_sorted = sorted(unselected, key=lambda m: -1.0/m)
        
        for m_new in unselected_sorted:
            if m_new in sel_set:
                continue
            valid = get_valid_residues_fast(m_new, sol)
            if valid:
                r_new = valid[0]
                sol.append([m_new, r_new])
                sel_set.add(m_new)
        
        return sol
    
    def local_search(sol, max_iters=500):
        """
        Local search: try changing residues or swapping to enable adding new moduli.
        """
        best = [x[:] for x in sol]
        best_score_val = score_solution(best)
        selected_moduli = set(m for m, r in best)
        unselected_moduli = [m for m in moduli if m not in selected_moduli]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not unselected_moduli:
                break
                
            # Strategy 1: Change a random selected modulus's residue
            idx = random.randint(0, len(best) - 1)
            m, r = best[idx]
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            compatible = True
            for i, (m2, r2) in enumerate(best):
                if i == idx:
                    continue
                g = get_gcd(m, m2)
                if r_prime % g == r2 % g:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            new_sol = [x[:] for x in best]
            new_sol[idx] = [m, r_prime]
            
            # Try to add unselected moduli
            new_sol = try_add_moduli(new_sol, unselected_moduli)
            
            new_score = score_solution(new_sol)
            if new_score > best_score_val:
                best = new_sol
                best_score_val = new_score
                selected_moduli = set(m for m, r in best)
                unselected_moduli = [m for m in moduli if m not in selected_moduli]
        
        return best
    
    def simulated_annealing(initial_sol, max_iters=10000):
        """
        Simulated annealing with aggressive add-then-evaluate.
        """
        best = [x[:] for x in initial_sol]
        best_score_val = score_solution(best)
        current = [x[:] for x in initial_sol]
        current_score_val = best_score_val
        
        T = 0.5
        cooling = 0.999
        min_T = 1e-5
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            m, r = current[idx]
            
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            compatible = True
            for i, (m2, r2) in enumerate(current):
                if i == idx:
                    continue
                g = get_gcd(m, m2)
                if r_prime % g == r2 % g:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            new_sol = [x[:] for x in current]
            new_sol[idx] = [m, r_prime]
            
            # Aggressively try to add any newly available moduli
            sel_set = set(m2 for m2, _ in new_sol)
            unselected = [m2 for m2 in moduli if m2 not in sel_set]
            new_sol = try_add_moduli(new_sol, unselected)
            
            new_score_val = score_solution(new_sol)
            delta = new_score_val - current_score_val
            
            if delta >= 0 or random.random() < math.exp(-float(delta) / max(T, 1e-10)):
                current = new_sol
                current_score_val = new_score_val
                if current_score_val > best_score_val:
                    best = [x[:] for x in current]
                    best_score_val = current_score_val
            
            T = max(T * cooling, min_T)
        
        return best
    
    # Strategy:
    # 1. Multiple greedy constructions with different orders
    # 2. Local search
    # 3. Simulated annealing
    # 4. Random restarts
    
    best_solution = []
    best_score = Fraction(0, 1)
    
    # Phase 1: Greedy with different orders
    # Sorted ascending
    order_sorted = sorted(moduli)
    sol_greedy = greedy_construct(order_sorted, residue_strategy='min_damage')
    sol_greedy = try_add_moduli(sol_greedy, [m for m in moduli if m not in set(mm for mm, _ in sol_greedy)])
    score_greedy = score_solution(sol_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_solution = sol_greedy
    
    # Sorted descending
    order_desc = sorted(moduli, reverse=True)
    sol_greedy2 = greedy_construct(order_desc, residue_strategy='min_damage')
    sol_greedy2 = try_add_moduli(sol_greedy2, [m for m in moduli if m not in set(mm for mm, _ in sol_greedy2)])
    score_greedy2 = score_solution(sol_greedy2)
    if score_greedy2 > best_score:
        best_score = score_greedy2
        best_solution = sol_greedy2
    
    # Phase 2: Local search
    sol_ls = local_search(best_solution, max_iters=500)
    score_ls = score_solution(sol_ls)
    if score_ls > best_score:
        best_score = score_ls
        best_solution = sol_ls
    
    # Phase 3: Simulated annealing
    sol_sa = simulated_annealing(best_solution, max_iters=8000)
    score_sa = score_solution(sol_sa)
    if score_sa > best_score:
        best_score = score_sa
        best_solution = sol_sa
    
    # Phase 4: More local search after SA
    sol_ls2 = local_search(best_solution, max_iters=300)
    score_ls2 = score_solution(sol_ls2)
    if score_ls2 > best_score:
        best_score = score_ls2
        best_solution = sol_ls2
    
    # Phase 5: Random restarts
    for restart in range(15):
        if time.time() > deadline:
            break
        
        order_rand = moduli[:]
        random.shuffle(order_rand)
        sol_r = greedy_construct(order_rand, residue_strategy='random')
        sol_r = try_add_moduli(sol_r, [m for m in moduli if m not in set(mm for mm, _ in sol_r)])
        score_r = score_solution(sol_r)
        
        sol_r_ls = local_search(sol_r, max_iters=150)
        score_r_ls = score_solution(sol_r_ls)
        
        if score_r_ls > best_score:
            best_score = score_r_ls
            best_solution = sol_r_ls
        elif score_r > best_score:
            best_score = score_r
            best_solution = sol_r
    
    # Phase 6: Final SA polish
    if time.time() < deadline - 15:
        sol_final_sa = simulated_annealing(best_solution, max_iters=5000)
        score_final = score_solution(sol_final_sa)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final_sa
        
        sol_final_ls = local_search(best_solution, max_iters=200)
        score_final_ls = score_solution(sol_final_ls)
        if score_final_ls > best_score:
            best_score = score_final_ls
            best_solution = sol_final_ls
    
    return best_solution
# EVOLVE-BLOCK-END

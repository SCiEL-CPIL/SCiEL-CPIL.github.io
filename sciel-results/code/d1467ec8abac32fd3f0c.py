"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    moduli_set = set(moduli)
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i+1, n_moduli):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        a, b = min(m1, m2), max(m1, m2)
        return gcd_table.get((a, b), math.gcd(a, b))
    
    # Precompute inverse moduli as floats for fast scoring
    inv_moduli = {m: 1.0 / m for m in moduli}
    
    def score_solution(sol):
        return sum(inv_moduli[m] for m, r in sol)
    
    def get_valid_residues(m, selected_pairs):
        """Get valid residues for modulus m given selected pairs."""
        if not selected_pairs:
            return list(range(m))
        blocked = [False] * m
        for m_sel, r_sel in selected_pairs:
            if m_sel == m:
                blocked[r_sel] = True
            else:
                g = get_gcd(m, m_sel)
                r_mod = r_sel % g
                for r in range(r_mod, m, g):
                    blocked[r] = True
        return [r for r in range(m) if not blocked[r]]
    
    def check_compatible(m, r, selected_pairs, skip_idx=None):
        """Check if (m, r) is compatible with all selected_pairs (optionally skipping one)."""
        for i, (m2, r2) in enumerate(selected_pairs):
            if i == skip_idx:
                continue
            g = get_gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def greedy_construct(order, residue_strategy='first'):
        """
        Construct a solution by processing moduli in the given order.
        """
        selected = []
        
        for m in order:
            valid_residues = get_valid_residues(m, selected)
            if not valid_residues:
                continue
            
            if len(valid_residues) == 1:
                r = valid_residues[0]
            elif residue_strategy == 'random':
                r = random.choice(valid_residues)
            elif residue_strategy == 'min_damage':
                # Pick residue minimizing total newly-blocked residues for future moduli
                best_r = valid_residues[0]
                min_damage = float('inf')
                sel_set = set(mm for mm, _ in selected)
                
                for r_candidate in valid_residues:
                    damage = 0
                    for n in order:
                        if n in sel_set or n == m:
                            continue
                        g = get_gcd(m, n)
                        r_mod = r_candidate % g
                        ap_size = (n - 1 - r_mod) // g + 1
                        damage += ap_size
                        if damage >= min_damage:
                            break
                    if damage < min_damage:
                        min_damage = damage
                        best_r = r_candidate
                        if min_damage == 0:
                            break
                r = best_r
            else:  # 'first'
                r = valid_residues[0]
            
            selected.append([m, r])
        
        return selected
    
    def try_add_moduli(sol, unselected_moduli):
        """Try to add as many unselected moduli as possible to sol."""
        new_sol = [x[:] for x in sol]
        sel_set = set(m for m, r in new_sol)
        changed = True
        while changed:
            changed = False
            for m_new in unselected_moduli:
                if m_new in sel_set:
                    continue
                found_r = None
                for r_new in range(m_new):
                    if check_compatible(m_new, r_new, new_sol):
                        found_r = r_new
                        break
                if found_r is not None:
                    new_sol.append([m_new, found_r])
                    sel_set.add(m_new)
                    changed = True
        return new_sol
    
    def targeted_insertion(sol, target_moduli):
        """
        Try to add specific target moduli by adjusting residues of existing moduli.
        For each target, find which existing moduli block it, then try changing their residues.
        """
        best = [x[:] for x in sol]
        best_score_val = score_solution(best)
        
        for target_m in target_moduli:
            if time.time() > deadline:
                break
            if target_m in set(m for m, r in best):
                continue
            
            # Find which residues block target_m
            blocking_idx = []
            for i, (m_sel, r_sel) in enumerate(best):
                if m_sel == target_m:
                    continue
                g = get_gcd(target_m, m_sel)
                r_sel_mod = r_sel % g
                # Check if any residue of target_m is compatible
                compatible = False
                for r_target in range(target_m):
                    if r_target % g != r_sel_mod:
                        compatible = True
                        break
                if not compatible:
                    blocking_idx.append(i)
            
            if not blocking_idx:
                # No blockers, just add it
                for r_target in range(target_m):
                    if check_compatible(target_m, r_target, best):
                        best.append([target_m, r_target])
                        best_score_val = score_solution(best)
                        break
                continue
            
            # Try changing one blocker's residue at a time
            for bi in blocking_idx:
                if time.time() > deadline:
                    break
                m_block, r_block = best[bi]
                
                # Try all possible new residues for the blocker
                for r_new in range(m_block):
                    if r_new == r_block:
                        continue
                    if not check_compatible(m_block, r_new, best, skip_idx=bi):
                        continue
                    
                    trial = [x[:] for x in best]
                    trial[bi] = [m_block, r_new]
                    
                    # Check if target can now be added
                    for r_target in range(target_m):
                        if check_compatible(target_m, r_target, trial):
                            trial.append([target_m, r_target])
                            trial_score = score_solution(trial)
                            if trial_score > best_score_val:
                                best = trial
                                best_score_val = trial_score
                            break
                    else:
                        continue
                    break
        return best
    
    def local_search(sol, max_iters=200):
        """
        Local search: try changing residues of selected moduli to enable adding new moduli.
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
                
            # Pick a random selected modulus to perturb
            idx = random.randint(0, len(best) - 1)
            m, r = best[idx]
            
            # Pick a new random residue
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            # Check compatibility with all other selected
            if not check_compatible(m, r_prime, best, skip_idx=idx):
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
    
    def simulated_annealing(initial_sol, max_iters=5000):
        """
        Simulated annealing: perturb residues, accept worse states with probability exp(-delta/T).
        """
        best = [x[:] for x in initial_sol]
        best_score_val = score_solution(best)
        current = [x[:] for x in initial_sol]
        current_score_val = best_score_val
        
        T = 1.0
        cooling = 0.9995
        min_T = 1e-4
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random selected modulus
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            m, r = current[idx]
            
            # Pick a new random residue
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            # Check compatibility
            if not check_compatible(m, r_prime, current, skip_idx=idx):
                continue
            
            new_sol = [x[:] for x in current]
            new_sol[idx] = [m, r_prime]
            
            # Try to greedily fill any newly available moduli
            sel_set = set(m2 for m2, _ in new_sol)
            unselected = [m for m in moduli if m not in sel_set]
            new_sol = try_add_moduli(new_sol, unselected)
            
            new_score_val = score_solution(new_sol)
            delta = new_score_val - current_score_val
            
            if delta >= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current = new_sol
                current_score_val = new_score_val
                if current_score_val > best_score_val:
                    best = [x[:] for x in current]
                    best_score_val = current_score_val
            
            T = max(T * cooling, min_T)
        
        return best
    
    # Strategy:
    # 1. Greedy with min_damage residue selection (sorted by increasing m)
    # 2. Targeted insertion for high-value missing moduli
    # 3. Local search on the result
    # 4. Simulated annealing from the best
    # 5. Multiple random restarts
    
    best_solution = []
    best_score = 0.0
    
    # Phase 1: Greedy with min_damage
    order_sorted = sorted(moduli)
    sol_greedy = greedy_construct(order_sorted, residue_strategy='min_damage')
    score_greedy = score_solution(sol_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_solution = sol_greedy
    
    # Phase 2: Targeted insertion - try to add highest value missing moduli
    selected_set = set(m for m, r in best_solution)
    missing = [m for m in moduli if m not in selected_set]
    missing_sorted = sorted(missing, key=lambda m: -inv_moduli[m])  # highest value first
    sol_targeted = targeted_insertion(best_solution, missing_sorted[:30])
    score_targeted = score_solution(sol_targeted)
    if score_targeted > best_score:
        best_score = score_targeted
        best_solution = sol_targeted
    
    # Phase 3: Local search
    sol_ls = local_search(best_solution, max_iters=300)
    score_ls = score_solution(sol_ls)
    if score_ls > best_score:
        best_score = score_ls
        best_solution = sol_ls
    
    # Phase 4: More targeted insertion
    selected_set = set(m for m, r in best_solution)
    missing = [m for m in moduli if m not in selected_set]
    missing_sorted = sorted(missing, key=lambda m: -inv_moduli[m])
    sol_targeted2 = targeted_insertion(best_solution, missing_sorted[:20])
    score_targeted2 = score_solution(sol_targeted2)
    if score_targeted2 > best_score:
        best_score = score_targeted2
        best_solution = sol_targeted2
    
    # Phase 5: Simulated annealing
    sol_sa = simulated_annealing(best_solution, max_iters=3000)
    score_sa = score_solution(sol_sa)
    if score_sa > best_score:
        best_score = score_sa
        best_solution = sol_sa
    
    # Phase 6: More local search after SA
    sol_ls2 = local_search(best_solution, max_iters=200)
    score_ls2 = score_solution(sol_ls2)
    if score_ls2 > best_score:
        best_score = score_ls2
        best_solution = sol_ls2
    
    # Phase 7: Random restarts with different strategies
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random order greedy
        order_rand = moduli[:]
        random.shuffle(order_rand)
        sol_r = greedy_construct(order_rand, residue_strategy='random')
        score_r = score_solution(sol_r)
        
        # Local search on random solution
        sol_r_ls = local_search(sol_r, max_iters=100)
        score_r_ls = score_solution(sol_r_ls)
        
        # Targeted insertion on random solution
        selected_set_r = set(m for m, r in sol_r_ls)
        missing_r = [m for m in moduli if m not in selected_set_r]
        missing_r_sorted = sorted(missing_r, key=lambda m: -inv_moduli[m])
        sol_r_t = targeted_insertion(sol_r_ls, missing_r_sorted[:15])
        score_r_t = score_solution(sol_r_t)
        
        best_candidate = max(score_r, score_r_ls, score_r_t)
        if best_candidate == score_r_t and score_r_t > best_score:
            best_score = score_r_t
            best_solution = sol_r_t
        elif best_candidate == score_r_ls and score_r_ls > best_score:
            best_score = score_r_ls
            best_solution = sol_r_ls
        elif best_candidate == score_r and score_r > best_score:
            best_score = score_r
            best_solution = sol_r
    
    # Phase 8: Final targeted insertion and SA polish from best
    if time.time() < deadline - 15:
        selected_set = set(m for m, r in best_solution)
        missing = [m for m in moduli if m not in selected_set]
        missing_sorted = sorted(missing, key=lambda m: -inv_moduli[m])
        sol_final_t = targeted_insertion(best_solution, missing_sorted[:25])
        score_final_t = score_solution(sol_final_t)
        if score_final_t > best_score:
            best_score = score_final_t
            best_solution = sol_final_t
        
        sol_final_sa = simulated_annealing(best_solution, max_iters=2000)
        score_final = score_solution(sol_final_sa)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final_sa
        
        # Final local search
        sol_final_ls = local_search(best_solution, max_iters=100)
        score_final_ls = score_solution(sol_final_ls)
        if score_final_ls > best_score:
            best_score = score_final_ls
            best_solution = sol_final_ls
    
    return best_solution
# EVOLVE-BLOCK-END

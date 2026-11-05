"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    moduli_set = set(moduli)
    
    # Precompute gcd for all pairs of moduli
    gcd_table = {}
    for i in range(n_moduli):
        for j in range(i+1, n_moduli):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    # Precompute for each pair (m, n): the gcd and the list of residues in [0,n) that are congruent to r mod g
    # For efficiency, we precompute the AP membership: for each (m, n), and each possible r % gcd(m,n),
    # which residues of n are blocked
    # Actually, let's precompute: for each m, n, g=gcd(m,n), and for each residue class c in [0,g),
    # the set of residues in [0,n) that are ≡ c (mod g)
    blocked_residues_cache = {}
    for m in moduli:
        for n in moduli:
            if m == n:
                continue
            g = gcd_table.get((min(m,n), max(m,n)), math.gcd(m,n))
            for c in range(g):
                residues = set(range(c, n, g))
                blocked_residues_cache[(m, n, c)] = residues
    
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
    
    def get_valid_residues(m, selected_pairs):
        """Get valid residues for modulus m given selected pairs."""
        if not selected_pairs:
            return list(range(m))
        valid = list(range(m))
        for m_sel, r_sel in selected_pairs:
            if m_sel == m:
                # Same modulus, just exclude r_sel
                if r_sel in valid:
                    valid.remove(r_sel)
            else:
                g = get_gcd(m, m_sel)
                r_mod = r_sel % g
                # Remove all residues in valid that are ≡ r_mod (mod g)
                to_remove = [r for r in valid if r % g == r_mod]
                for r in to_remove:
                    valid.remove(r)
        return valid
    
    def greedy_construct(order, residue_strategy='first'):
        """
        Construct a solution by processing moduli in the given order.
        residue_strategy: 'first', 'random', or 'min_damage'
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
                
                for r_candidate in valid_residues:
                    damage = 0
                    for n in order[order.index(m)+1:]:
                        if n in [mm for mm, _ in selected]:
                            continue
                        g = get_gcd(m, n)
                        r_mod = r_candidate % g
                        # Count how many residues of n would be blocked
                        # that are not already blocked by existing selections
                        # For now, approximate by counting all residues in the AP
                        # This is an upper bound on actual new damage
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
            else:  # 'first'
                r = valid_residues[0]
            
            selected.append([m, r])
        
        return selected
    
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
            
            # Check if any unselected modulus can now be added
            added = False
            for m_new in unselected_moduli:
                # Find a valid residue for m_new
                found_r = None
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in new_sol:
                        g = get_gcd(m_new, m_sel)
                        if r_new % g == r_sel % g:
                            ok = False
                            break
                    if ok:
                        found_r = r_new
                        break
                
                if found_r is not None:
                    new_sol.append([m_new, found_r])
                    added = True
                    break
            
            if added:
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
        
        selected_moduli = set(m for m, r in current)
        
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
            
            # Try to greedily fill any newly available moduli
            sel_set = set(m2 for m2, _ in new_sol)
            for m_new in moduli:
                if m_new in sel_set:
                    continue
                found_r = None
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in new_sol:
                        g = get_gcd(m_new, m_sel)
                        if r_new % g == r_sel % g:
                            ok = False
                            break
                    if ok:
                        found_r = r_new
                        break
                if found_r is not None:
                    new_sol.append([m_new, found_r])
                    sel_set.add(m_new)
            
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
    # 1. Greedy with min_damage residue selection (sorted by increasing m)
    # 2. Local search on the result
    # 3. Simulated annealing from the best
    # 4. Multiple random restarts
    
    best_solution = []
    best_score = Fraction(0, 1)
    
    # Phase 1: Greedy with min_damage
    order_sorted = sorted(moduli)
    sol_greedy = greedy_construct(order_sorted, residue_strategy='min_damage')
    score_greedy = score_solution(sol_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_solution = sol_greedy
    
    # Phase 2: Local search
    sol_ls = local_search(best_solution, max_iters=300)
    score_ls = score_solution(sol_ls)
    if score_ls > best_score:
        best_score = score_ls
        best_solution = sol_ls
    
    # Phase 3: Simulated annealing
    sol_sa = simulated_annealing(best_solution, max_iters=3000)
    score_sa = score_solution(sol_sa)
    if score_sa > best_score:
        best_score = score_sa
        best_solution = sol_sa
    
    # Phase 4: More local search after SA
    sol_ls2 = local_search(best_solution, max_iters=200)
    score_ls2 = score_solution(sol_ls2)
    if score_ls2 > best_score:
        best_score = score_ls2
        best_solution = sol_ls2
    
    # Phase 5: Random restarts with different strategies
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
        
        if score_r_ls > best_score:
            best_score = score_r_ls
            best_solution = sol_r_ls
        elif score_r > best_score:
            best_score = score_r
            best_solution = sol_r
    
    # Phase 6: Final SA polish from best
    if time.time() < deadline - 10:
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

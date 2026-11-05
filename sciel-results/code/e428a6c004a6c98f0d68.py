"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def score_solution(selected):
        return sum(1.0 / m for m, r in selected)
    
    def greedy_initial():
        """Greedy: pick moduli in order of 1/m (smallest first), choose first valid residue."""
        selected = []
        for m in moduli:  # already sorted ascending
            if time.time() > deadline:
                break
            for r in range(m):
                if is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def try_add_modulus(selected, m):
        """Try to add modulus m by finding a valid residue."""
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None
    
    def simulated_annealing(selected, time_limit=100.0):
        """
        Simulated annealing on residue assignments.
        State: for each modulus in moduli, either it's included (with a residue) or excluded.
        Moves: change residue of an included modulus, add an excluded modulus, remove an included one.
        """
        selected = [list(x) for x in selected]
        selected_set = set(m for m, r in selected)
        current_score = score_solution(selected)
        best_solution = [list(x) for x in selected]
        best_score = current_score
        
        # Build index map
        idx_map = {m: i for i, (m, r) in enumerate(selected)}
        
        T = 0.1  # initial temperature
        T_min = 1e-6
        cooling_rate = 0.9995
        
        while time.time() < start_time + time_limit and T > T_min:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5 and len(selected) > 0:
                # Change residue of a random included modulus
                idx = random.randint(0, len(selected) - 1)
                m_old, r_old = selected[idx]
                r_new = random.randint(0, m_old - 1)
                if r_new == r_old:
                    continue
                
                # Check compatibility with all others
                compatible = True
                for jdx, (m_j, r_j) in enumerate(selected):
                    if jdx == idx:
                        continue
                    g = math.gcd(m_old, m_j)
                    if (r_new - r_j) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Accept if improves, or with probability exp(-delta/T)
                # Score change is 0 (same moduli, just different residue)
                # So always accept valid moves
                selected[idx][1] = r_new
                
            elif move_type < 0.85:
                # Try to add a missing modulus
                missing = [m for m in moduli if m not in selected_set]
                if not missing:
                    continue
                m_add = random.choice(missing)
                r_add = try_add_modulus(selected, m_add)
                if r_add is None:
                    continue
                
                # This always improves score
                selected.append([m_add, r_add])
                selected_set.add(m_add)
                current_score += 1.0 / m_add
                idx_map[m_add] = len(selected) - 1
                
            else:
                # Remove a random modulus (to explore)
                if not selected:
                    continue
                idx = random.randint(0, len(selected) - 1)
                m_rem, r_rem = selected[idx]
                delta = -1.0 / m_rem
                
                # Accept removal with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    selected.pop(idx)
                    selected_set.discard(m_rem)
                    current_score += delta
                    # Rebuild idx_map
                    idx_map = {m: i for i, (m, r) in enumerate(selected)}
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_solution = [list(x) for x in selected]
            
            # Cool down
            T *= cooling_rate
        
        return best_solution
    
    def two_opt_swap(selected):
        """
        Try to improve by removing one modulus and adding another (or two-for-two swaps).
        """
        selected = [list(x) for x in selected]
        selected_set = set(m for m, r in selected)
        current_score = score_solution(selected)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try single add
            for m in moduli:
                if m in selected_set:
                    continue
                r = try_add_modulus(selected, m)
                if r is not None:
                    selected.append([m, r])
                    selected_set.add(m)
                    current_score += 1.0 / m
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swap: remove m1, add m2 where 1/m2 > 1/m1
            for idx in range(len(selected)):
                m1 = selected[idx][0]
                for m2 in moduli:
                    if m2 in selected_set or m2 == m1:
                        continue
                    if 1.0 / m2 <= 1.0 / m1:
                        continue
                    
                    # Temporarily remove m1
                    saved = selected.pop(idx)
                    selected_set.discard(m1)
                    
                    r2 = try_add_modulus(selected, m2)
                    if r2 is not None:
                        selected.append([m2, r2])
                        selected_set.add(m2)
                        improved = True
                        break
                    else:
                        selected.insert(idx, saved)
                        selected_set.add(m1)
                
                if improved:
                    break
        
        return selected
    
    # Phase 1: Greedy initialization
    best_solution = greedy_initial()
    best_score = score_solution(best_solution)
    
    # Phase 2: Two-opt swap to add missing moduli
    if time.time() < deadline - 30.0:
        sol = two_opt_swap(best_solution)
        sc = score_solution(sol)
        if sc > best_score:
            best_solution = sol
            best_score = sc
    
    # Phase 3: Simulated annealing for residue optimization
    if time.time() < deadline - 10.0:
        sa_time = min(80.0, deadline - time.time() - 10.0)
        if sa_time > 5.0:
            sol = simulated_annealing(best_solution, time_limit=sa_time)
            sc = score_solution(sol)
            if sc > best_score:
                best_solution = sol
                best_score = sc
    
    # Phase 4: Final two-opt polish
    if time.time() < deadline - 5.0:
        sol = two_opt_swap(best_solution)
        sc = score_solution(sol)
        if sc > best_score:
            best_solution = sol
            best_score = sc
    
    return best_solution
# EVOLVE-BLOCK-END

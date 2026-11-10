import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    def is_valid_pair(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def find_valid_residue(m, selected, randomize=True, num_trials=100):
        """Find a valid residue for m given selected moduli/residues."""
        if not selected:
            return 0

        if randomize:
            candidates = list(range(m))
            random.shuffle(candidates)
            for _ in range(min(num_trials, m)):
                if candidates:
                    r = candidates.pop()
                    ok = True
                    for m_prime, r_prime in selected:
                        if not is_valid_pair(m, r, m_prime, r_prime):
                            ok = False
                            break
                    if ok:
                        return r

            # Fallback: Sequential search
            for r in range(m):
                ok = True
                for m_prime, r_prime in selected:
                    if not is_valid_pair(m, r, m_prime, r_prime):
                        ok = False
                        break
                if ok:
                    return r
            return None
        else:
            for r in range(m):
                ok = True
                for m_prime, r_prime in selected:
                    if not is_valid_pair(m, r, m_prime, r_prime):
                        ok = False
                        break
                if ok:
                    return r
            return None

    def compute_conflict_score(m, all_moduli):
        """Compute how many residues m blocks for each other modulus.
        Lower score = fewer conflicts = better to include earlier."""
        total_conflicts = 0
        for m2 in all_moduli:
            if m2 == m:
                continue
            g = math.gcd(m, m2)
            # m blocks g residues out of m2
            total_conflicts += g
        return total_conflicts

    def greedy_build(order, randomize_residues=True):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected, randomize_residues)
            if r is not None:
                selected.append([m, r])
        return selected

    def try_add_missing(solution, max_attempts=50):
        """Try to add missing moduli, prioritizing small ones (high weight)."""
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        
        added = 0
        for m in missing:
            if time.time() > deadline:
                break
            if added >= max_attempts:
                break
            
            r = find_valid_residue(m, solution, randomize=True, num_trials=50)
            if r is not None:
                solution.append([m, r])
                added += 1
        return solution

    def local_search_1opt(solution, max_iterations=200):
        """Try to add any missing modulus that fits."""
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()  # Prioritize small moduli
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            for m in missing[:]:
                if m in selected_moduli:
                    continue
                r = find_valid_residue(m, solution, randomize=True, num_trials=30)
                if r is not None:
                    solution.append([m, r])
                    selected_moduli.add(m)
                    missing.remove(m)
                    improved = True
                    break
        return solution

    def local_search_2opt(solution, max_swaps=100):
        """Try to swap one modulus for another to improve the score."""
        selected_moduli = set(m for m, r in solution)
        missing = [m for m in range(20, 201) if m not in selected_moduli]
        missing.sort()
        
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            best_gain = 0
            best_swap = None
            
            # Try removing each selected modulus and adding a missing one
            for idx in range(len(solution)):
                m_remove, r_remove = solution[idx]
                remaining = solution[:idx] + solution[idx+1:]
                
                for m_add in missing[:20]:  # Only check small missing moduli
                    r_add = find_valid_residue(m_add, remaining, randomize=False)
                    if r_add is not None:
                        gain = 1.0/m_add - 1.0/m_remove
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (idx, m_remove, r_remove, m_add, r_add)
            
            if best_swap is not None:
                idx, m_remove, r_remove, m_add, r_add = best_swap
                solution.pop(idx)
                solution.append([m_add, r_add])
                selected_moduli.discard(m_remove)
                selected_moduli.add(m_add)
                if m_add in missing:
                    missing.remove(m_add)
                if m_remove not in missing:
                    missing.append(m_remove)
                    missing.sort()
            else:
                break
        
        return solution

    def score_solution(solution):
        if not solution:
            return 0.0
        return sum(1.0 / m for m, r in solution)

    best_result = []
    best_score = 0.0
    
    all_moduli = list(range(20, 201))
    
    # Precompute conflict scores to guide ordering
    conflict_scores = {m: compute_conflict_score(m, all_moduli) for m in all_moduli}
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # Strategy 1: Ascending order (prioritize small moduli)
        # Strategy 2: Conflict-aware ordering (fewer conflicts first)
        # Strategy 3: Randomized
        
        if num_restarts % 3 == 1:
            order = all_moduli[:]  # Ascending
        elif num_restarts % 3 == 2:
            # Sort by conflict score, then by modulus
            order = sorted(all_moduli, key=lambda m: (conflict_scores[m], m))
        else:
            order = all_moduli[:]
            random.shuffle(order)
        
        solution = greedy_build(order, randomize_residues=True)
        
        # Repair: Try to add missing moduli
        solution = try_add_missing(solution, max_attempts=50)
        
        # Local search: 1-opt (add missing) and 2-opt (swap)
        solution = local_search_1opt(solution, max_iterations=100)
        solution = local_search_2opt(solution, max_swaps=50)
        
        current_score = score_solution(solution)
        if current_score > best_score:
            best_score = current_score
            best_result = [list(x) for x in solution]
            
        if time.time() > deadline - 5:
            break

    # Final validation
    if best_result:
        for i in range(len(best_result)):
            for j in range(i + 1, len(best_result)):
                m1, r1 = best_result[i]
                m2, r2 = best_result[j]
                if not is_valid_pair(m1, r1, m2, r2):
                    return [[20, 0]]
    
    return best_result

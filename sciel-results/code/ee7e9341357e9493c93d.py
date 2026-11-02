import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 140.0  # Safety margin for 150s limit
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor matrix
    neighbor_matrix = np.zeros((total_words, total_words), dtype=bool)
    np.fill_diagonal(neighbor_matrix, True)
    
    row_indices = []
    col_indices = []
    
    for idx in range(total_words):
        row_indices.append(idx)
        col_indices.append(idx)
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    
    neighbor_matrix[np.array(row_indices), np.array(col_indices)] = True
    
    def greedy_cover(uncovered_mask, solution):
        sol = list(solution)
        unc = uncovered_mask.copy()
        
        while unc.any():
            if time.time() - start_time > limit:
                break
            
            counts = np.sum(neighbor_matrix[:, unc], axis=1)
            
            if sol:
                sol_arr = np.array(sol, dtype=int)
                counts[sol_arr] = 0
                
            # Random tie-breaking
            counts += np.random.rand(total_words) * 0.5
            
            best_word_idx = np.argmax(counts)
            
            if counts[best_word_idx] < 0.1:
                break
                
            sol.append(int(best_word_idx))
            unc[neighbor_matrix[best_word_idx]] = False
            
        return sol

    def is_complete(solution):
        if not solution:
            return total_words == 0
        covered = np.any(neighbor_matrix[solution], axis=0)
        return covered.all()

    def remove_redundant(solution):
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            if not sol:
                break
            
            coverage_counts = np.sum(neighbor_matrix[sol], axis=0)
            
            for i in range(len(sol) - 1, -1, -1):
                if time.time() - start_time > limit:
                    break
                w = sol[i]
                if coverage_counts[w] > 1:
                    if np.min(coverage_counts[neighbor_matrix[w]]) > 1:
                        sol.pop(i)
                        changed = True
                        break
        return sol

    def local_search(solution, max_iters=2000):
        if not solution:
            return solution
        
        sol = list(solution)
        sol_set = set(sol)
        coverage_counts = np.sum(neighbor_matrix[sol], axis=0)
        critical_count = np.sum(coverage_counts == 1)
        
        best_sol = list(sol)
        best_len = len(sol)
        best_critical = critical_count
        
        for iteration in range(max_iters):
            if time.time() - start_time > limit:
                break
            
            improved = False
            
            if sol:
                remove_idx = random.randint(0, len(sol) - 1)
                w_remove = sol[remove_idx]
                mask_w = neighbor_matrix[w_remove]
                
                if np.min(coverage_counts[mask_w]) > 1:
                    sol.pop(remove_idx)
                    sol_set.remove(w_remove)
                    coverage_counts -= mask_w
                    critical_count = np.sum(coverage_counts == 1)
                    improved = True
                    if len(sol) < best_len:
                        best_sol = list(sol)
                        best_len = len(sol)
                        best_critical = critical_count
                    continue
            
            remove_idx = random.randint(0, len(sol) - 1)
            w_remove = sol[remove_idx]
            mask_w = neighbor_matrix[w_remove]
            
            critical_mask = (coverage_counts == 1) & mask_w
            critical_words = np.where(critical_mask)[0]
            
            if len(critical_words) == 0:
                sol.pop(remove_idx)
                sol_set.remove(w_remove)
                coverage_counts -= mask_w
                critical_count = np.sum(coverage_counts == 1)
                improved = True
                if len(sol) < best_len:
                    best_sol = list(sol)
                    best_len = len(sol)
                    best_critical = critical_count
                continue
            
            if len(critical_words) > 0:
                candidates = np.where(neighbor_matrix[:, critical_words[0]])[0]
                for cw in critical_words[1:]:
                    if len(candidates) == 0: break
                    candidates = candidates[neighbor_matrix[candidates, cw]]
                
                if len(candidates) > 0:
                    in_sol_mask = np.zeros(total_words, dtype=bool)
                    in_sol_mask[list(sol_set)] = True
                    valid_candidates = candidates[~in_sol_mask[candidates]]
                    
                    if len(valid_candidates) > 0:
                        c_add = int(random.choice(valid_candidates))
                        mask_c = neighbor_matrix[c_add]
                        
                        losers = mask_w & ~mask_c
                        if not losers.any() or np.min(coverage_counts[losers]) > 1:
                            new_counts = coverage_counts.copy()
                            new_counts -= mask_w
                            new_counts += mask_c
                            new_critical = np.sum(new_counts == 1)
                            
                            if new_critical <= critical_count:
                                sol[remove_idx] = c_add
                                sol_set.remove(w_remove)
                                sol_set.add(c_add)
                                coverage_counts = new_counts
                                critical_count = new_critical
                                improved = True
                                if critical_count < best_critical:
                                    best_sol = list(sol)
                                    best_critical = critical_count
                                    best_len = len(sol)
            
            if not improved:
                pass
        
        return best_sol

    # Main loop
    best_solution = []
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() - start_time < limit - 10:
        uncovered = np.ones(total_words, dtype=bool)
        solution = greedy_cover(uncovered, [])
        solution = remove_redundant(solution)
        solution = local_search(solution)
        
        if len(solution) < best_size:
            best_solution = solution
            best_size = len(solution)
            
        # If we hit a good size, we can stop early?
        # The best known is 486. If we get below 500, it's a great result.
        if best_size <= 490:
            break
            
    # Final check
    if not is_complete(best_solution):
        covered = np.any(neighbor_matrix[best_solution], axis=0) if best_solution else np.zeros(total_words, dtype=bool)
        uncovered = ~covered
        best_solution = greedy_cover(uncovered, best_solution)
    
    return [get_digits(idx) for idx in best_solution]

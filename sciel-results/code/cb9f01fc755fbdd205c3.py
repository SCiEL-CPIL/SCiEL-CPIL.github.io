import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0

    all_moduli = list(range(2, 129))
    n_mods = len(all_moduli)

    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    degrees = [0] * n_mods
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            if gcd_mat[i][j] > 1:
                degrees[i] += 1
                degrees[j] += 1

    order = sorted(range(n_mods), key=lambda x: -degrees[x])

    def count_valid_residues(m_idx, assigned):
        m = all_moduli[m_idx]
        count = 0
        for r in range(m):
            ok = True
            for r_j, m_j_idx in assigned:
                g = gcd_mat[m_idx][m_j_idx]
                if (r - r_j) % g == 0:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def get_valid_residues(m_idx, assigned):
        m = all_moduli[m_idx]
        valid = []
        for r in range(m):
            ok = True
            for r_j, m_j_idx in assigned:
                g = gcd_mat[m_idx][m_j_idx]
                if (r - r_j) % g == 0:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    # Greedy initialization
    greedy_assigned = []
    for idx in order:
        if time.time() > deadline:
            break
        valid_res = get_valid_residues(idx, greedy_assigned)
        if valid_res:
            r = valid_res[0]
            greedy_assigned.append((r, idx))
            
    best_count = len(greedy_assigned)
    best_solution = list(greedy_assigned)

    def backtrack(assigned, unassigned, current_count):
        nonlocal best_count, best_solution
        
        if time.time() > deadline:
            return
            
        if current_count > best_count:
            best_count = current_count
            best_solution = list(assigned)
            
        if not unassigned:
            return
            
        if current_count + len(unassigned) <= best_count:
            return

        min_count = float('inf')
        min_idx = -1
        
        for idx in unassigned:
            count = count_valid_residues(idx, assigned)
            if count < min_count:
                min_count = count
                min_idx = idx
            if count == 0:
                return

        if min_count == 0:
            return

        min_valid = get_valid_residues(min_idx, assigned)
        
        for r in min_valid:
            if time.time() > deadline:
                return
            assigned.append((r, min_idx))
            new_unassigned = unassigned - {min_idx}
            backtrack(assigned, new_unassigned, current_count + 1)
            assigned.pop()

    unassigned_all = set(range(n_mods))
    backtrack([], unassigned_all, 0)

    result = []
    for r, idx in best_solution:
        m = all_moduli[idx]
        result.append([r, m])
        
    return result

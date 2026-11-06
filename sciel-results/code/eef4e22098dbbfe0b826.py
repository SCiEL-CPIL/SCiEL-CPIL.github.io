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
    num_moduli = len(moduli)
    
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

    def ac3_prune(moduli_set, initial_domains):
        """
        AC-3 arc consistency to prune incompatible residue values.
        Returns pruned domains for each modulus.
        """
        # Build domain for each modulus: list of valid residues
        domains = {}
        for m in moduli_set:
            domains[m] = list(range(m))
        
        # Queue of arcs (i, j) where i and j are moduli
        queue = []
        moduli_list = list(moduli_set)
        for i in moduli_list:
            for j in moduli_list:
                if i != j:
                    queue.append((i, j))
        
        while queue:
            if time.time() > deadline - 5:
                break
            i, j = queue.pop(0)
            g = gcd_table[i][j]
            
            # For each value in domain[i], check if it has support in domain[j]
            new_domain_i = []
            for r_i in domains[i]:
                has_support = False
                for r_j in domains[j]:
                    if r_i % g != r_j % g:
                        has_support = True
                        break
                if has_support:
                    new_domain_i.append(r_i)
            
            # If domain changed, add all arcs (k, i) to queue
            if len(new_domain_i) < len(domains[i]):
                domains[i] = new_domain_i
                if not new_domain_i:
                    return None  # Inconsistent
                for k in moduli_list:
                    if k != i and (k, i) not in queue:
                        queue.append((k, i))
        
        return domains

    def csp_solve(moduli_set, max_time=5.0):
        """
        Solve CSP using backtracking with AC-3 pruning.
        Returns assignment dict {m: r} or None if infeasible.
        """
        start_time = time.time()
        
        # Get initial pruned domains
        domains = ac3_prune(moduli_set, None)
        if domains is None:
            return None
        
        # Order variables by smallest domain first (MRV heuristic)
        var_order = sorted(moduli_set, key=lambda m: len(domains[m]))
        
        assignment = {}
        
        def backtrack(idx):
            if time.time() - start_time > max_time:
                return None
            
            if idx == len(var_order):
                return dict(assignment)
            
            m = var_order[idx]
            g_table = gcd_table
            
            # Try each residue in domain
            for r in domains[m]:
                # Check consistency with assigned variables
                valid = True
                for m2, r2 in assignment.items():
                    g = g_table[m][m2]
                    if r % g == r2 % g:
                        valid = False
                        break
                
                if valid:
                    assignment[m] = r
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    del assignment[m]
            
            return None
        
        return backtrack(0)

    def weighted_maxsat_optimize(moduli_set, max_time=10.0):
        """
        Optimize selection using weighted approach: prioritize smaller moduli (higher 1/m weight).
        Uses iterative deepening with constraint propagation.
        """
        start_time = time.time()
        best_solution = []
        best_score = 0.0
        
        # Sort moduli by weight (1/m) descending - smaller m first
        sorted_moduli = sorted(moduli_set, key=lambda m: -1.0/m)
        
        # Try to include as many high-weight moduli as possible
        # Use greedy with constraint checking
        selected = []
        domains_cache = {}
        
        for m in sorted_moduli:
            if time.time() - start_time > max_time:
                break
            
            # Check if we can add m with some residue
            valid_rs = []
            for r in range(m):
                valid = True
                for m2, r2 in selected:
                    g = gcd_table[m][m2]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if valid_rs:
                # Pick residue that minimizes future conflicts
                best_r = valid_rs[0]
                min_conflicts = float('inf')
                for r in valid_rs:
                    conflicts = 0
                    for m2 in sorted_moduli:
                        if m2 == m or any(s[0] == m2 for s in selected):
                            continue
                        g = gcd_table[m][m2]
                        # Count how many residues of m2 conflict with r
                        conflict_count = 0
                        for r2 in range(m2):
                            if r % g == r2 % g:
                                conflict_count += 1
                        conflicts += conflict_count
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_r = r
                selected.append((m, best_r))
        
        if selected:
            score = sum(1.0/m for m, _ in selected)
            if score > best_score:
                best_score = score
                best_solution = selected
        
        return best_solution, best_score

    def local_search_csp(initial_selected, max_time=15.0):
        """
        Local search with CSP-based validation.
        Tries to add, remove, or swap moduli while maintaining validity.
        """
        start_time = time.time()
        selected = [tuple(p) for p in initial_selected]
        current_score = calculate_score(selected)
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            
            # 1. Try to add any unselected modulus
            for m in moduli:
                if m in [s[0] for s in selected]:
                    continue
                valid_rs = get_valid_residues(m, selected)
                if valid_rs:
                    # Pick residue that minimizes conflicts
                    best_r = valid_rs[0]
                    min_conflicts = float('inf')
                    for r in valid_rs:
                        conflicts = 0
                        for m2 in moduli:
                            if m2 == m or any(s[0] == m2 for s in selected):
                                continue
                            g = gcd_table[m][m2]
                            for r2 in range(m2):
                                if r % g == r2 % g:
                                    conflicts += 1
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            best_r = r
                    new_selected = selected + [(m, best_r)]
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
                
            # 3. Random kick: remove a random element and try to refill
            if time.time() - start_time < max_time - 2:
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected = [s for s in selected if s != removed]
                    current_score = calculate_score(selected)
                    improved = True

        return selected, current_score

    def try_fill_all_csp():
        """
        Try to find valid residue assignment for ALL moduli using CSP with AC-3.
        """
        # First, check if full set is feasible using AC-3
        domains = ac3_prune(moduli, None)
        if domains is None:
            return None
        
        # Try backtracking search
        result = csp_solve(moduli, max_time=10.0)
        if result:
            return [[m, r] for m, r in result.items()]
        
        return None

    best_solution = []
    best_score = 0.0
    
    # Phase 1: Try to get the full set using CSP
    full_sol = try_fill_all_csp()
    if full_sol:
        score = calculate_score(full_sol)
        if score > best_score:
            best_score = score
            best_solution = full_sol
    
    # Phase 2: If full set not found, use weighted optimization + local search
    if not best_solution or best_score < sum(1.0/m for m in moduli) - 1e-9:
        # Weighted Max-SAT style optimization
        opt_sol, opt_score = weighted_maxsat_optimize(moduli, max_time=10.0)
        if opt_score > best_score:
            best_score = opt_score
            best_solution = opt_sol
        
        # Local search to improve
        if best_solution:
            sol, score = local_search_csp(best_solution, max_time=20.0)
            if score > best_score:
                best_score = score
                best_solution = sol
    
    # Phase 3: Multiple restarts with different randomizations
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        # Randomized weighted optimization
        random.shuffle(moduli)
        opt_sol, opt_score = weighted_maxsat_optimize(moduli, max_time=5.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_solution = opt_sol
        
        # Local search
        if best_solution:
            sol, score = local_search_csp(best_solution, max_time=5.0)
            if score > best_score:
                best_score = score
                best_solution = sol

    # Convert to required format
    if best_solution:
        return [[m, r] for m, r in best_solution]
    
    # Fallback: simple greedy
    selected = []
    for m in moduli:
        valid_rs = get_valid_residues(m, selected)
        if valid_rs:
            selected.append((m, valid_rs[0]))
    return [[m, r] for m, r in selected]

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

import math
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # All possible moduli from 2 to 128
    all_moduli = list(range(2, 129))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcds[i][j] = math.gcd(all_moduli[i], all_moduli[j])
    
    # For each pair of moduli, compute which residues are incompatible
    # Two classes (r1, m1) and (r2, m2) are disjoint iff (r1 - r2) % gcd(m1, m2) != 0
    # So for given moduli m1, m2 with gcd g, residue r1 conflicts with r2 if (r1-r2) % g == 0
    
    # We want to select a subset of moduli and assign residues such that all pairs are disjoint
    # This is a maximum independent set problem on a conflict graph
    
    # Strategy: Use AC-3 to prune, then backtracking with good variable ordering
    
    # Variable ordering: try moduli that have more constraints first
    # A modulus m conflicts with another modulus m' if gcd(m, m') > 1
    # Actually, any two moduli can have compatible residues, so we need to think differently
    
    # The key insight: for moduli m1, m2 with gcd g, exactly g out of every g residues conflict
    # So the fraction of compatible residues is (g-1)/g for each pair
    
    # Let's use a greedy + local search approach with AC-3 pruning
    
    # First, let's try to find a large set using greedy with AC-3
    
    def ac3_prune(moduli_set, assigned):
        """
        moduli_set: list of (index, modulus) pairs to consider
        assigned: dict mapping index -> residue
        Returns: dict of index -> set of valid residues, or None if contradiction
        """
        n_vars = len(moduli_set)
        if n_vars == 0:
            return {}
        
        # Initialize domains
        domains = {}
        for idx, m in moduli_set:
            if idx in assigned:
                domains[idx] = {assigned[idx]}
            else:
                domains[idx] = set(range(m))
        
        # Build constraint queue
        queue = deque()
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                idx_i, m_i = moduli_set[i]
                idx_j, m_j = moduli_set[j]
                if len(domains[idx_i]) > 1 and len(domains[idx_j]) > 1:
                    queue.append((i, j))
        
        # AC-3
        while queue:
            i, j = queue.popleft()
            if time.time() > deadline:
                return None
            
            idx_i, m_i = moduli_set[i]
            idx_j, m_j = moduli_set[j]
            g = gcds[idx_i][idx_j]
            
            if g == 1:
                # No constraint, all residues compatible
                continue
            
            # For each value in domain_i, check if there's a compatible value in domain_j
            new_domain_i = set()
            for r_i in domains[idx_i]:
                for r_j in domains[idx_j]:
                    if (r_i - r_j) % g != 0:
                        new_domain_i.add(r_i)
                        break
            
            if len(new_domain_i) < len(domains[idx_i]):
                domains[idx_i] = new_domain_i
                if not domains[idx_i]:
                    return None
                # Add constraints from i to all other variables
                for k in range(n_vars):
                    if k != i and k != j:
                        queue.append((k, i))
            
            # Symmetric check
            new_domain_j = set()
            for r_j in domains[idx_j]:
                for r_i in domains[idx_i]:
                    if (r_i - r_j) % g != 0:
                        new_domain_j.add(r_j)
                        break
            
            if len(new_domain_j) < len(domains[idx_j]):
                domains[idx_j] = new_domain_j
                if not domains[idx_j]:
                    return None
                for k in range(n_vars):
                    if k != i and k != j:
                        queue.append((k, j))
        
        return domains
    
    # Greedy construction with AC-3
    def greedy_with_ac3():
        """
        Greedily add moduli one at a time, using AC-3 to check feasibility
        """
        selected = []  # list of (index, modulus, residue)
        used_indices = set()
        
        # Order moduli by how "constrained" they are
        # Moduli with more prime factors are more constrained
        def constraint_score(m):
            count = 0
            temp = m
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    count += 1
                    temp //= d
                d += 1
            if temp > 1:
                count += 1
            return count
        
        # Try moduli in order of decreasing constraint score (more constrained first)
        sorted_moduli = sorted(range(n), key=lambda i: -constraint_score(all_moduli[i]))
        
        for idx in sorted_moduli:
            if time.time() > deadline:
                break
            
            m = all_moduli[idx]
            # Try all residues for this modulus
            found = False
            for r in range(m):
                # Check if adding (idx, r) is compatible with current selection
                compatible = True
                for (sel_idx, sel_m, sel_r) in selected:
                    g = gcds[idx][sel_idx]
                    if (r - sel_r) % g == 0:
                        compatible = False
                        break
                
                if compatible:
                    # Use AC-3 to verify that the extended set is feasible
                    current_set = [(i, all_moduli[i]) for i, _, _ in selected]
                    current_set.append((idx, m))
                    assigned = {i: r for i, _, r in selected}
                    assigned[idx] = r
                    
                    domains = ac3_prune(current_set, assigned)
                    if domains is not None:
                        # Check that all domains are non-empty
                        feasible = all(len(domains[i]) > 0 for i, _ in current_set)
                        if feasible:
                            selected.append((idx, m, r))
                            found = True
                            break
            
            if not found:
                continue
        
        return selected
    
    # Local search to improve the greedy solution
    def local_search(initial_selected):
        """
        Try to improve by swapping residues or adding/removing moduli
        """
        selected = list(initial_selected)
        best = len(selected)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add a new modulus
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                if any(idx == s[0] for s in selected):
                    continue
                
                m = all_moduli[idx]
                for r in range(m):
                    compatible = True
                    for (sel_idx, sel_m, sel_r) in selected:
                        g = gcds[idx][sel_idx]
                        if (r - sel_r) % g == 0:
                            compatible = False
                            break
                    
                    if compatible:
                        new_selected = selected + [(idx, m, r)]
                        if len(new_selected) > best:
                            selected = new_selected
                            best = len(selected)
                            improved = True
                        break
            
            # Try to swap residues
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                idx_i, m_i, r_i = selected[i]
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    compatible = True
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        sel_idx, sel_m, sel_r = selected[j]
                        g = gcds[idx_i][sel_idx]
                        if (r_new - sel_r) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        selected[i] = (idx_i, m_i, r_new)
                        # This swap doesn't increase size, but may enable future additions
                        # Don't mark as improved for now
        
        return selected
    
    # Run greedy
    greedy_result = greedy_with_ac3()
    
    # Run local search
    final_result = local_search(greedy_result)
    
    if final_result:
        return [[r, m] for (_, m, r) in final_result]
    else:
        # Fallback
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]

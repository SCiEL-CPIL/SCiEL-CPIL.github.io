import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Precompute gcd and coprime relationships
    gcd_tab = [[0]*n for _ in range(n)]
    coprime_adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
            if g == 1:
                coprime_adj[i].append(j)
                coprime_adj[j].append(i)
    
    def try_assign_residues(selected_indices):
        """
        Try to assign residues to the selected indices using backtracking.
        Returns a list of (idx, residue, modulus) if successful, else None.
        """
        selected_indices = list(selected_indices)
        # Order by modulus size descending? Or by constraint?
        # Let's order by modulus size descending to have more freedom early?
        # Or ascending to fix constraints early?
        # Usually, fixing highly constrained variables first is better.
        # A variable is constrained if it has few valid residues left.
        # For now, let's just sort by index or modulus.
        selected_indices.sort(key=lambda i: moduli[i])
        
        assignment = {}
        
        def backtrack(pos):
            if pos == len(selected_indices):
                return True
            
            idx = selected_indices[pos]
            m = moduli[idx]
            
            for r in range(m):
                ok = True
                for prev_idx, prev_r in assignment.items():
                    g = gcd_tab[idx][prev_idx]
                    if (r - prev_r) % g == 0:
                        ok = False
                        break
                if ok:
                    assignment[idx] = r
                    if backtrack(pos + 1):
                        return True
                    del assignment[idx]
            return False
        
        if backtrack(0):
            return [(idx, assignment[idx], moduli[idx]) for idx in selected_indices]
        return None

    def greedy_independent_set(order):
        """
        Greedily select moduli based on the given order, ensuring they form a clique
        in the non-coprime graph (i.e., no two are coprime).
        """
        available = set(range(n))
        selected = []
        
        for idx in order:
            if idx not in available:
                continue
            
            # Check if idx is coprime to any already selected?
            # No, we want a clique in the non-coprime graph.
            # So idx must NOT be coprime to any selected.
            # i.e., gcd(idx, sel) > 1 for all sel in selected.
            
            compatible = True
            for sel_idx in selected:
                if gcd_tab[idx][sel_idx] == 1:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                # Remove all coprime neighbors of idx from available
                # Because they can't be added if we keep idx?
                # Wait, if we add idx, we can't add any j such that gcd(idx, j) == 1.
                # So we remove idx and all its coprime neighbors from available.
                to_remove = set(coprime_adj[idx])
                to_remove.add(idx)
                available -= to_remove
                
        return selected

    best_result = []
    
    # Generate orderings
    # 1. Min degree in coprime graph
    degrees = [len(adj) for adj in coprime_adj]
    order_min_deg = sorted(range(n), key=lambda i: degrees[i])
    
    # 2. Max degree in non-coprime graph (Min degree in coprime)
    # Same as above essentially.
    
    # 3. Random orderings
    random.seed(42)
    orderings = [order_min_deg]
    for _ in range(100):
        orderings.append(list(range(n)))
        random.shuffle(orderings[-1])
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
            
        selected_idx = greedy_independent_set(order)
        if len(selected_idx) <= len(best_result):
            continue
            
        res = try_assign_residues(selected_idx)
        if res is not None:
            if len(res) > len(best_result):
                best_result = res
    
    # Local search / Improvement
    if time.time() > deadline - 5:
        return [[r, m] for (_, r, m) in best_result]
    
    # Try to add more moduli to the best result found
    selected_idx_set = set(idx for (idx, _, _) in best_result)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for idx in range(n):
            if idx in selected_idx_set:
                continue
            
            # Check if idx is coprime to any selected
            compatible = True
            for sel_idx in selected_idx_set:
                if gcd_tab[idx][sel_idx] == 1:
                    compatible = False
                    break
            
            if not compatible:
                continue
                
            # Try to find a residue for idx that fits
            m = moduli[idx]
            found = False
            for r in range(m):
                ok = True
                for (sel_idx, sel_r, sel_m) in best_result:
                    if (r - sel_r) % gcd_tab[idx][sel_idx] == 0:
                        ok = False
                        break
                if ok:
                    best_result.append((idx, r, m))
                    selected_idx_set.add(idx)
                    improved = True
                    found = True
                    break
            if improved:
                break
    
    return [[r, m] for (_, r, m) in best_result]

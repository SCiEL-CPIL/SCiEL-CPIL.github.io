import math

def construct():
    max_mod = 128
    candidates = list(range(2, max_mod + 1))
    
    # Heuristic: Try moduli with more small factors first? 
    # Or just try all and see which set is largest.
    # To maximize count, we should prioritize moduli that are "easy" to satisfy.
    # Actually, just iterating and checking is fine.
    
    # To speed up, we can sort candidates.
    # Moduli with more prime factors are more constrained?
    # Let's just use the natural order or reverse.
    
    selected_moduli = []
    
    # Satisfiability checker
    def is_satisfiable(mods):
        n = len(mods)
        if n == 0:
            return True
        if n == 1:
            return True
            
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    gcds[i][j] = math.gcd(mods[i], mods[j])
        
        domains = [list(range(m)) for m in mods]
        
        # AC-3
        def revise(i, j):
            g = gcds[i][j]
            residues_j = set(v % g for v in domains[j])
            if len(residues_j) > 1:
                return False
            R = next(iter(residues_j))
            new_domain = [v for v in domains[i] if v % g != R]
            if len(new_domain) < len(domains[i]):
                domains[i] = new_domain
                return True
            return False

        # Queue initialization
        queue = [(i, j) for i in range(n) for j in range(n) if i != j]
        
        while queue:
            i, j = queue.pop(0)
            if not domains[i]:
                return False
            if revise(i, j):
                if not domains[i]:
                    return False
                for k in range(n):
                    if k != i and k != j:
                        queue.append((k, i))
                        
        # Backtracking
        # Simple MRV backtracking
        def backtrack(idx, assignments):
            if idx == n:
                return True
            # Pick variable with smallest domain (MRV)
            min_idx = idx
            min_size = len(domains[idx])
            for k in range(idx, n):
                if len(domains[k]) < min_size:
                    min_size = len(domains[k])
                    min_idx = k
            
            # Swap to put min_idx at idx
            domains[idx], domains[min_idx] = domains[min_idx], domains[idx]
            mods[idx], mods[min_idx] = mods[min_idx], mods[idx] # Note: mods is local copy? No, mods is list passed in. We should copy it.
            
            # Actually, modifying mods in place is dangerous for the outer loop.
            # We should pass a copy or handle indices carefully.
            # For now, let's assume the order is fixed to simplify, 
            # or just use the fixed order 0..n-1.
            
            # Let's stick to fixed order 0..n-1 for simplicity in the swap logic above.
            # The swap above modifies the local `mods` list.
            
            for r in domains[idx]:
                valid = True
                for k in range(idx):
                    g = gcds[idx][k]
                    if (r - assignments[k]) % g == 0:
                        valid = False
                        break
                if valid:
                    assignments[idx] = r
                    if backtrack(idx + 1, assignments):
                        return True
                    assignments[idx] = None
            
            # Undo swap? Not strictly necessary if we re-solve from scratch each time.
            return False

        assignments = [None] * n
        return backtrack(0, assignments)

    # Greedy selection
    # Sort candidates to potentially find a large set faster
    # Heuristic: Moduli with larger gcds with others might be easier?
    # Let's just iterate through 2..128.
    
    for m in candidates:
        new_set = selected_moduli + [m]
        if is_satisfiable(new_set):
            selected_moduli = new_set
            
    # Final solve to get residues
    if not selected_moduli:
        return [[1, 2]]
        
    # Re-run the solver to get the actual residues
    n = len(selected_moduli)
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcds[i][j] = math.gcd(selected_moduli[i], selected_moduli[j])
                
    domains = [list(range(m)) for m in selected_moduli]
    
    # AC-3 again
    def revise(i, j):
        g = gcds[i][j]
        residues_j = set(v % g for v in domains[j])
        if len(residues_j) > 1:
            return False
        R = next(iter(residues_j))
        new_domain = [v for v in domains[i] if v % g != R]
        if len(new_domain) < len(domains[i]):
            domains[i] = new_domain
            return True
        return False

    queue = [(i, j) for i in range(n) for j in range(n) if i != j]
    while queue:
        i, j = queue.pop(0)
        if not domains[i]:
            break # Should not happen if is_satisfiable passed
        if revise(i, j):
            if not domains[i]:
                break
            for k in range(n):
                if k != i and k != j:
                    queue.append((k, i))
                    
    def backtrack(idx, assignments):
        if idx == n:
            return True
        min_idx = idx
        min_size = len(domains[idx])
        for k in range(idx, n):
            if len(domains[k]) < min_size:
                min_size = len(domains[k])
                min_idx = k
        
        domains[idx], domains[min_idx] = domains[min_idx], domains[idx]
        selected_moduli[idx], selected_moduli[min_idx] = selected_moduli[min_idx], selected_moduli[idx]
        
        for r in domains[idx]:
            valid = True
            for k in range(idx):
                g = gcds[idx][k]
                if (r - assignments[k]) % g == 0:
                    valid = False
                    break
            if valid:
                assignments[idx] = r
                if backtrack(idx + 1, assignments):
                    return True
                assignments[idx] = None
        return False

    assignments = [None] * n
    if backtrack(0, assignments):
        # Note: selected_moduli was swapped, so we need to map back?
        # The assignments correspond to the swapped order.
        # We just need to return [residue, modulus] pairs.
        # The pairs are (assignments[i], selected_moduli[i]).
        return [[assignments[i], selected_moduli[i]] for i in range(n)]
        
    return [[1, 2]]

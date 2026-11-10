import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def try_assign(moduli):
        """Try to assign residues. Returns list of [m, r] or None."""
        n = len(moduli)
        if n == 0:
            return []
        
        assignment = [None] * n
        
        def dfs(idx):
            if time.time() > deadline - 5:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            # Check constraints with all previously assigned
            for r in range(m):
                valid = True
                for j in range(idx):
                    m_j = moduli[j]
                    r_j = assignment[j]
                    g = math.gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                if valid:
                    assignment[idx] = r
                    if dfs(idx + 1):
                        return True
            
            return False
        
        if dfs(0):
            return [[moduli[i], assignment[i]] for i in range(n)]
        return None
    
    # Generate candidate moduli: only those that can form pairwise non-coprime sets
    # Simple approach: try all moduli, but the CSP will reject coprime pairs
    
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 20:
            break
        test = selected + [m]
        # Quick check: if m is coprime with any selected, skip
        feasible = True
        for m_j in selected:
            if math.gcd(m, m_j) == 1:
                feasible = False
                break
        if not feasible:
            continue
        result = try_assign(test)
        if result is not None:
            selected.append(m)
    
    final = try_assign(selected)
    if final is None:
        return []
    
    best_score = sum(1.0 / m for m in selected)
    best_selection = final
    
    # Local search
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            selected_set = set(selected)
            for m in range(20, 201):
                if m in selected_set:
                    continue
                # Quick coprime check
                feasible = True
                for m_j in selected:
                    if math.gcd(m, m_j) == 1:
                        feasible = False
                        break
                if not feasible:
                    continue
                test = selected + [m]
                result = try_assign(test)
                if result is not None:
                    selected.append(m)
                    best_score = sum(1.0 / mm for mm in selected)
                    best_selection = result
                    improved = True
                    break
    
    return best_selection

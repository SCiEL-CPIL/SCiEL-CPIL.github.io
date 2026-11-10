import math
import time

def _find_valid_residue(m, selected, gcd_cache=None):
    """Find a valid residue for modulus m compatible with selected."""
    for r in range(m):
        compatible = True
        for (m_i, r_i) in selected:
            if gcd_cache is not None:
                g = gcd_cache.get((m, m_i), math.gcd(m, m_i))
            else:
                g = math.gcd(m, m_i)
            if r % g == r_i % g:
                compatible = False
                break
        if compatible:
            return r
    return None

def _score(selected):
    """Compute score as sum of 1/m."""
    return sum(1.0 / m for m, r in selected)

def _local_search(selected, deadline):
    """Improve selected via local search: try swapping moduli for better 1/m."""
    best = list(selected)
    best_score = _score(best)
    n = len(best)
    
    # Precompute gcd matrix for speed
    moduli = [m for m, r in best]
    gcds = [[math.gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try replacing a modulus with a different one (prefer smaller m for higher 1/m)
        for i in range(n):
            if time.time() > deadline:
                return best, best_score
            m_old, r_old = best[i]
            current_moduli_set = set(moduli)
            
            # Try all candidate moduli sorted by 1/m descending (smaller m first)
            candidates = [m for m in range(20, 201) if m not in current_moduli_set]
            candidates.sort(key=lambda x: -1.0/x)
            
            best_new_m = None
            best_new_r = None
            best_new_score = best_score
            
            for m_new in candidates:
                # Check if we can find a valid residue for m_new
                r_new = None
                for r in range(m_new):
                    compatible = True
                    for j in range(n):
                        if j == i:
                            continue
                        g = math.gcd(m_new, moduli[j])
                        if r % g == best[j][1] % g:
                            compatible = False
                            break
                    if compatible:
                        r_new = r
                        break
                
                if r_new is not None:
                    new_score = best_score - 1.0/m_old + 1.0/m_new
                    if new_score > best_new_score + 1e-12:
                        best_new_score = new_score
                        best_new_m = m_new
                        best_new_r = r_new
            
            if best_new_m is not None:
                best[i] = (best_new_m, best_new_r)
                moduli[i] = best_new_m
                for j in range(n):
                    if j != i:
                        gcds[i][j] = math.gcd(best_new_m, moduli[j])
                        gcds[j][i] = gcds[i][j]
                best_score = best_new_score
                improved = True
        
        # Try adding a new modulus if possible
        if time.time() < deadline - 1:
            current_moduli_set = set(moduli)
            candidates = [m for m in range(20, 201) if m not in current_moduli_set]
            candidates.sort(key=lambda x: -1.0/x)
            
            for m_new in candidates:
                if time.time() > deadline:
                    break
                r_new = None
                for r in range(m_new):
                    compatible = True
                    for j in range(n):
                        g = math.gcd(m_new, moduli[j])
                        if r % g == best[j][1] % g:
                            compatible = False
                            break
                    if compatible:
                        r_new = r
                        break
                
                if r_new is not None:
                    best.append((m_new, r_new))
                    moduli.append(m_new)
                    n += 1
                    for j in range(n-1):
                        gcds[i][j] = math.gcd(m_new, moduli[j]) if i < n-1 else math.gcd(m_new, moduli[j])
                        if i < n-1:
                            gcds[j][i] = gcds[i][j]
                    best_score += 1.0/m_new
                    improved = True
                    break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    selected = []
    
    # Phase 1: Greedy construction - try smaller moduli first (higher 1/m)
    for m in range(20, 201):
        if time.time() > deadline - 60:
            break
        r = _find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        selected, score = _local_search(selected, deadline - 5)
    
    return [[m, r] for (m, r) in selected]

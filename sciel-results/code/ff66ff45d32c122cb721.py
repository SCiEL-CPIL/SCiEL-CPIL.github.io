<<<<<<< SEARCH
def _dyson_gas_equilibrium(roots, double_root_factor=2.0):
    """
    Solve the Dyson gas equilibrium for the given roots.
    For double roots, the logarithmic derivative of P(x) = prod (x-r_i)^2 gives:
    sum_{j != i} double_root_factor / (r_i - r_j) = 1/(2*r_i) + 1/2
    
    We use fixed-point iteration with brentq for each coordinate.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(500):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Find bracket
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-12
                right = other_sorted[0] - 1e-10
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-10
                right = max(100.0, r_guess + 10.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-10
                right = other_sorted[insert_pos] - 1e-10
            
            if left < 1e-12:
                left = 1e-12
            if right <= left:
                left = max(1e-12, r_guess - 5.0)
                right = r_guess + 5.0
            
            def f(r):
                if r <= 1e-15:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-15):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return double_root_factor * s - 1.0/(2.0*r) - 0.5
            
            found = False
            for attempt in range(50):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right):
                        left = max(1e-12, left - 1.0)
                        right = right + 1.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-14, rtol=1e-14, maxiter=200)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-14:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-14:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        left = max(1e-12, left - 2.0)
                        right = right + 2.0
                except Exception:
                    left = max(1e-12, left - 2.0)
                    right = right + 2.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-14:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-12)
    
    return roots
=======
def _dyson_gas_equilibrium(roots, double_root_factor=2.0, confining_strength=1.0):
    """
    Solve the Dyson gas equilibrium for the given roots.
    Potential: V(r) = 0.5 log(r) + confining_strength * r
    Force: V'(r) = 0.5/r + confining_strength
    Equation: double_root_factor * sum(1/(r-r_j)) = 0.5/r + confining_strength
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(1000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Find bracket
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-12
                right = other_sorted[0] - 1e-10
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-10
                right = max(100.0, r_guess + 10.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-10
                right = other_sorted[insert_pos] - 1e-10
            
            if left < 1e-12:
                left = 1e-12
            if right <= left:
                left = max(1e-12, r_guess - 5.0)
                right = r_guess + 5.0
            
            def f(r):
                if r <= 1e-15:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-15):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return double_root_factor * s - 0.5/r - confining_strength
            
            found = False
            for attempt in range(100):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right):
                        left = max(1e-12, left - 1.0)
                        right = right + 1.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-14, rtol=1e-14, maxiter=200)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-14:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-14:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        left = max(1e-12, left - 2.0)
                        right = right + 2.0
                except Exception:
                    left = max(1e-12, left - 2.0)
                    right = right + 2.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-14:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-12)
    
    return roots
>>>>>>> REPLACE


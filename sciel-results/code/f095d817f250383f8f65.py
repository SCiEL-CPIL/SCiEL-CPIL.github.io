import numpy as np

def find_root(f, x0):
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    calls = 1
    max_calls = 450  # Leave some buffer
    
    best_x = x
    best_f = abs(fx)
    
    def bisect(a, b, fa, fb):
        nonlocal calls
        # Assume fa * fb <= 0
        for _ in range(50):
            if calls >= max_calls:
                break
            m = (a + b) / 2.0
            fm = f(m)
            calls += 1
            if abs(fm) <= 1e-8:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-12:
                break
        # Return the point with smaller |f|
        if abs(fa) < abs(fb):
            return a
        else:
            return b
            
    def golden_section(a, b, max_iter=20):
        nonlocal calls
        # Minimize |f(x)| in [a, b]
        gr = (np.sqrt(5) - 1) / 2  # 0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = abs(f(c))
        calls += 1
        fd = abs(f(d))
        calls += 1
        
        for _ in range(max_iter):
            if calls >= max_calls:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = abs(f(c))
                calls += 1
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = abs(f(d))
                calls += 1
            if abs(b - a) < 1e-12:
                break
        
        # Return best point
        if fc < fd:
            return c
        else:
            return d

    # Search strategy: Expanding window with grid scan
    radius = 1.0
    while radius < 1e6 and calls < max_calls:
        a = x - radius
        b = x + radius
        
        # Determine number of grid points based on radius to maintain some resolution
        # But don't use too many calls. 
        # Let's use a fixed number of points, say 50, for the grid scan.
        N = 50
        if calls + N > max_calls:
            N = max_calls - calls
            if N <= 1:
                break
            
        # Generate grid points
        # Use linspace
        grid = np.linspace(a, b, N + 1)
        
        # Evaluate f on grid
        vals = []
        for gx in grid:
            if calls >= max_calls:
                break
            val = f(gx)
            calls += 1
            vals.append(val)
            
        if len(vals) < 2:
            break
            
        # Update best
        min_val_idx = np.argmin([abs(v) for v in vals])
        if abs(vals[min_val_idx]) < best_f:
            best_f = abs(vals[min_val_idx])
            best_x = grid[min_val_idx]
            
        if best_f <= 1e-8:
            return best_x
            
        # Check for sign changes
        root_found = False
        for i in range(len(vals) - 1):
            if calls >= max_calls:
                break
            v1 = vals[i]
            v2 = vals[i+1]
            if v1 * v2 < 0:
                # Sign change found
                r = bisect(grid[i], grid[i+1], v1, v2)
                if r is not None:
                    fr = f(r)
                    if abs(fr) <= 1e-8:
                        return r
                    # Even if not < 1e-8, it's a good candidate
                    if abs(fr) < best_f:
                        best_f = abs(fr)
                        best_x = r
                root_found = True
                break # Found one, try to refine it? Or continue searching? 
                # Usually one root is enough. Let's return if it's good.
        
        # If no sign change, try Golden Section on the best local interval
        if not root_found and best_f < 1e-4:
            # Refine the best point found so far
            # Use a small window around best_x
            w = max(1e-6, radius * 0.01)
            ga = best_x - w
            gb = best_x + w
            r = golden_section(ga, gb)
            if r is not None:
                fr = f(r)
                if abs(fr) <= 1e-8:
                    return r
                if abs(fr) < best_f:
                    best_f = abs(fr)
                    best_x = r
                    
        # Expand radius
        radius *= 5.0
        
    return best_x

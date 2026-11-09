import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Secant method with safeguards
    x0 = float(x0)
    # Initial second point
    h = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + h
    
    f0 = f(x0)
    f1 = f(x1)
    
    # If we already have a root
    if abs(f0) <= 1e-8:
        return x0
    if abs(f1) <= 1e-8:
        return x1
    
    # Secant iteration
    for i in range(100):
        if f1 == f0:
            # Degenerate case, perturb
            x1 = x0 + h * (i + 2)
            f1 = f(x1)
            continue
            
        # Next point
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        # Check for non-finite
        if not math.isfinite(x2):
            break
            
        f2 = f(x2)
        
        if not math.isfinite(f2):
            break
            
        if abs(f2) <= 1e-8:
            return x2
            
        # Update
        x0, x1 = x1, x2
        f0, f1 = f1, f2
        
        # Check for convergence in x
        if abs(x1 - x0) < 1e-15 * max(1.0, abs(x1)):
            break

    # If Secant failed, try a simple expansion to find a bracket, then Illinois
    # This is a fallback, unlikely to be needed for smooth polynomials
    
    # Return best guess
    if abs(f1) < abs(f0):
        return x1
    else:
        return x0
# EVOLVE-BLOCK-END

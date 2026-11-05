import numpy as np
from scipy.optimize import fsolve
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    m = 12  # number of double roots
    
    # Solve Dyson gas equilibrium: for each i, sum_{j != i} 2/(r_i - r_j) = 1
    # with 0 < r_1 < r_2 < ... < r_12
    
    def equations(r):
        n = len(r)
        eqs = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += 2.0 / (r[i] - r[j])
            eqs[i] = s - 1.0
        return eqs
    
    # Initial guess: spread out roots
    r0 = np.linspace(1.0, 20.0, m)
    
    # Try to solve
    try:
        r_sol, info, ier, msg = fsolve(equations, r0, full_output=True, xtol=1e-12, maxfev=10000)
        
        # Check if solution is valid (all positive, sorted)
        if ier == 1 and np.all(r_sol > 0):
            r_sol = np.sort(r_sol)
            # Verify the solution
            res = equations(r_sol)
            if np.max(np.abs(res)) < 1e-8:
                return [float(x) for x in r_sol]
    except Exception:
        pass
    
    # Fallback: try different initial guesses
    for offset in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for scale in [1.0, 2.0, 5.0]:
            r0 = np.linspace(offset, offset + scale * m, m)
            try:
                r_sol, info, ier, msg = fsolve(equations, r0, full_output=True, xtol=1e-12, maxfev=10000)
                if ier == 1 and np.all(r_sol > 0):
                    r_sol = np.sort(r_sol)
                    res = equations(r_sol)
                    if np.max(np.abs(res)) < 1e-8:
                        return [float(x) for x in r_sol]
            except Exception:
                pass
            
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Last resort: use the original
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END

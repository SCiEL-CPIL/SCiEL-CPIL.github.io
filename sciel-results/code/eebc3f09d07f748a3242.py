"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Build conflict graph: two numbers conflict if lcm(a,b) <= 256
    numbers = list(range(2, 257))
    n = len(numbers)
    idx = {num: i for i, num in enumerate(numbers)}
    
    # Find all conflicting pairs
    conflicts = []
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflicts.append((i, j))
    
    # Try MILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Objective: maximize sum(x_i / numbers[i]) -> minimize -sum(x_i / numbers[i])
        c = np.array([-1.0 / numbers[i] for i in range(n)])
        
        # Constraints: for each conflicting pair (i,j): x_i + x_j <= 1
        if conflicts:
            m = len(conflicts)
            A = np.zeros((m, n))
            for row, (i, j) in enumerate(conflicts):
                A[row, i] = 1.0
                A[row, j] = 1.0
            cons = LinearConstraint(A, -np.inf, np.ones(m))
        else:
            cons = LinearConstraint(np.zeros((0, n)), -np.inf, np.zeros(0))
        
        bounds = Bounds(0, 1)
        integrality = np.ones(n)  # all binary
        
        result = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            selected = [numbers[i] for i in range(n) if result.x[i] > 0.5]
            if selected:
                return selected
        elif result.status in (2, 3):  # optimal or locally optimal
            if result.x is not None:
                selected = [numbers[i] for i in range(n) if result.x[i] > 0.5]
                if selected:
                    return selected
    except Exception:
        pass
    
    # Fallback: greedy approach if MILP fails or times out
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return a simple valid set
        return [256]
    
    # Greedy: sort by 1/num descending (i.e., num ascending), add if no conflict
    selected = []
    # Try adding numbers in order of smallest (highest reciprocal)
    for x in sorted(numbers, key=lambda v: v):
        if all(math.lcm(x, y) > 256 for y in selected):
            selected.append(x)
    
    return selected if selected else [256]
# EVOLVE-BLOCK-END

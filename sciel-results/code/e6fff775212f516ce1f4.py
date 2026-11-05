import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)  # 47 elements, indices 0 to 46

    # Compute LCM of all denominators for exact integer arithmetic
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Precompute scaled values: v[i] = L // denominators[i]
    # Then sum of reciprocals = sum(v[i]) / L
    # We want sum(v[i]) < L (strictly)
    # Minimize L - sum(v[i]) => Maximize sum(v[i]) subject to sum(v[i]) <= L - 1
    
    v = [L // d for d in denominators]
    
    # Try ILP approach first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Binary variables x_i in {0,1}
        # Maximize sum(v[i] * x[i])
        # Subject to: sum(v[i] * x[i]) <= L - 1
        #             sum(x[i]) >= 1
        #             x[i] in {0,1}
        
        # milp minimizes, so we minimize -sum(v[i] * x[i])
        c = np.array([-float(vi) for vi in v])
        
        # Constraints:
        # 1) sum(v[i] * x[i]) <= L - 1  =>  -sum(v[i]*x[i]) >= -(L-1)
        # 2) sum(x[i]) >= 1
        
        A = np.array([
            [-float(vi) for vi in v],  # -sum(v[i]*x[i]) >= -(L-1)
            [1.0] * n                    # sum(x[i]) >= 1
        ])
        b_lower = np.array([-(L - 1), 1.0])
        b_upper = np.array([np.inf, np.inf])
        
        constraint = LinearConstraint(A, b_lower, b_upper)
        integrality = np.ones(n)  # all binary
        bounds = Bounds(0, 1)
        
        result = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            selected = [i for i in range(n) if result.x[i] > 0.5]
            if selected:
                # Verify with Fraction
                total_frac = Fraction()
                for idx in selected:
                    total_frac += Fraction(1, denominators[idx])
                if total_frac < 1:
                    return selected
                else:
                    # Should not happen, but fallback
                    pass
        
        # ILP failed or invalid, fall through to meet-in-the-middle
    except Exception:
        pass
    
    # Meet-in-the-middle fallback
    n1 = n // 2  # 23
    n2 = n - n1  # 24
    
    # Generate all subset sums for part 1 (indices 0 to n1-1)
    sums1 = []
    for mask in range(1 << n1):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += v[idx]
            idx += 1
            m >>= 1
        if s < L:  # prune: only keep valid partial sums
            sums1.append((s, mask))
        
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Generate all subset sums for part 2 (indices n1 to n-1)
    sums2 = []
    for mask in range(1 << n2):
        s = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += v[n1 + idx]
            idx += 1
            m >>= 1
        if s < L:
            sums2.append((s, mask))
        
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Sort sums2 by sum
    sums2.sort(key=lambda x: x[0])
    sums2_values = [x[0] for x in sums2]
    
    best_deficit = L  # max possible deficit is L (empty set)
    best_mask1 = 0
    best_mask2 = 0
    
    import bisect
    
    # For each sum in part 1, find best match in part 2
    # We want s1 + s2 < L (strictly)
    # So s2 < L - s1
    # Find the rightmost s2 in sums2_values that is < L - s1
    
    for s1, m1 in sums1:
        if time.time() > DEADLINE:
            break
        target = L - s1  # s2 must be < target
        
        if target <= 0:
            continue
        
        # Find the rightmost index in sums2_values that is < target
        pos = bisect.bisect_left(sums2_values, target)
        
        # Check candidates at pos-1 and pos-2 (for safety)
        for check_idx in [pos - 1, pos - 2]:
            if 0 <= check_idx < len(sums2):
                s2, m2 = sums2[check_idx]
                total = s1 + s2
                if total < L:  # strictly less
                    deficit = L - total
                    if deficit < best_deficit:
                        best_deficit = deficit
                        best_mask1 = m1
                        best_mask2 = m2
    
    if time.time() > DEADLINE:
        return greedy_fallback(denominators)
    
    # Convert best masks to indices
    selected = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            selected.append(i)
    for i in range(n2):
        if best_mask2 & (1 << i):
            selected.append(n1 + i)
    
    # Verify with Fraction
    if not selected:
        # Edge case: empty set
        selected = [0]  # Just 1/2, deficit = 1/2
    
    total_frac = Fraction()
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])
        
    if total_frac >= 1:
        # Should not happen with integer arithmetic, but just in case
        selected = [0, 1, 2, 3]
    
    return selected

def greedy_fallback(denominators):
    # Simple greedy: add largest available term if sum < 1
    selected = []
    total = Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected

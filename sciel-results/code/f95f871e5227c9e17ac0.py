import numpy as np
from scipy.optimize import differential_evolution

def compute_1d_union_length_sq_integral(x_vals, N):
    # Critical points
    critical = [0.0, 1.0]
    for i in range(N):
        for j in range(i+1, N):
            # L_i = L_j => c = N*(x_j - x_i)/(i-j)
            # L_i = R_j => c = N*(x_j - x_i + 1/N)/(i-j)
            # R_i = L_j => c = N*(x_j - x_i - 1/N)/(i-j)
            # R_i = R_j is same as L_i = L_j
            for offset in [0, 1/N, -1/N]:
                # c = N * (x_j - x_i + offset) / (i - j)
                # Note: L_i(c) = x_i + c*i/N. R_i(c) = L_i(c) + 1/N.
                # Intersection of L_i and L_j: x_i + c i/N = x_j + c j/N -> c(i-j)/N = x_j - x_i -> c = N(x_j-x_i)/(i-j)
                # Intersection of L_i and R_j: x_i + c i/N = x_j + c j/N + 1/N -> c(i-j)/N = x_j - x_i + 1/N -> c = N(x_j-x_i+1/N)/(i-j)
                # Intersection of R_i and L_j: x_i + c i/N + 1/N = x_j + c j/N -> c(i-j)/N = x_j - x_i - 1/N -> c = N(x_j-x_i-1/N)/(i-j)
                # Intersection of R_i and R_j: same as L_i = L_j
                denom = i - j
                if denom != 0:
                    c1 = N * (x_vals[j] - x_vals[i]) / denom
                    c2 = N * (x_vals[j] - x_vals[i] + 1.0/N) / denom
                    c3 = N * (x_vals[j] - x_vals[i] - 1.0/N) / denom
                    for c_val in [c1, c2, c3]:
                        if 0 <= c_val <= 1:
                            critical.append(c_val)
    
    critical = sorted(set([c for c in critical if 0 <= c <= 1]))
    
    total_sq_integral = 0.0
    for k in range(len(critical) - 1):
        c_start = critical[k]
        c_end = critical[k+1]
        
        # Determine L(c) on (c_start, c_end)
        # Sample a point c_mid to determine the active set and order?
        # Actually, the topology of the union (which intervals overlap) is constant on this segment.
        # L(c) is the sum of lengths of disjoint segments.
        # We can evaluate L(c) and its derivative at c_mid?
        # No, L(c) is piecewise linear. The slope changes only at critical points.
        # So L(c) = m*c + b on this segment.
        # We can find m and b by evaluating at two points or using the definition.
        
        c_mid = (c_start + c_end) / 2
        # Evaluate union length at c_mid and c_mid + epsilon?
        # Better: L(c) is determined by the intervals.
        # Let's just evaluate L at c_start + 1e-9 and c_end - 1e-9?
        # Or simply, since it's linear, we can pick c1 = c_start + 1e-12, c2 = c_end - 1e-12.
        
        def get_union_len(c):
            intervals = []
            for i in range(N):
                left = x_vals[i] + c * i / N
                right = left + 1.0 / N
                intervals.append((left, right))
            intervals.sort()
            if not intervals:
                return 0.0
            cur_l, cur_r = intervals[0]
            length = 0.0
            for l, r in intervals[1:]:
                if l > cur_r:
                    length += cur_r - cur_l
                    cur_l, cur_r = l, r
                else:
                    cur_r = max(cur_r, r)
            length += cur_r - cur_l
            return length

        # Use two points close to the boundaries to avoid precision issues if c_start==c_end (shouldn't happen)
        eps = 1e-9
        c1 = c_start + eps
        c2 = c_end - eps
        if c2 <= c1:
            # Segment is tiny
            L_mid = get_union_len(c_start)
            total_sq_integral += L_mid**2 * (c_end - c_start)
            continue
            
        L1 = get_union_len(c1)
        L2 = get_union_len(c2)
        
        # L(c) = m*c + b
        # L1 = m*c1 + b
        # L2 = m*c2 + b
        # m = (L2 - L1) / (c2 - c1)
        # b = L1 - m*c1
        m = (L2 - L1) / (c2 - c1)
        b = L1 - m * c1
        
        # Integrate (m*c + b)^2 from c_start to c_end
        # Integral = [ (m*c+b)^3 / (3*m) ] from c_start to c_end
        # If m == 0: L^2 * (c_end - c_start)
        if abs(m) < 1e-12:
            val = L1**2
            total_sq_integral += val * (c_end - c_start)
        else:
            term1 = (m*c_end + b)**3
            term2 = (m*c_start + b)**3
            total_sq_integral += (term1 - term2) / (3.0 * m)
            
    return total_sq_integral

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    def obj(x_vals):
        return compute_1d_union_length_sq_integral(x_vals, N)
    
    # Bounds for x_vals
    # The tubes are in [0,1] height.
    # x_k can be roughly in [-1, 2] to be safe?
    # The union must cover the base? No, it's a Kakeya set, it just needs to contain a unit segment in every direction.
    # But here the "directions" are fixed by i,j.
    # The problem is to minimize the volume of the union of these specific tubes.
    # The x positions determine the x-spread.
    # Reasonable bounds: [-2, 3]
    
    bounds = [(-2.0, 3.0) for _ in range(N)]
    
    # Use Differential Evolution
    # Seed for reproducibility? The task says rng=12345 for the *evaluation* MC, not for the optimizer.
    # We should use a fixed seed for the optimizer to be deterministic.
    result = differential_evolution(obj, bounds, seed=42, maxiter=100, tol=1e-8, polish=True)
    
    x_vals = result.x
    y_vals = x_vals.copy()
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([x_vals[i_flat], y_vals[j_flat]], axis=1)
    return positions

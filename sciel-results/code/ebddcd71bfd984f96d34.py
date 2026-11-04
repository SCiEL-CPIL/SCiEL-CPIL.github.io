import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s
    
    # 1. Initialization: WL Color Refinement
    def wl_init(a, b, max_rounds=50):
        """
        Weisfeiler-Lehman color refinement to find good initial assignment.
        Iteratively refine node colors based on neighborhood structure.
        """
        # Initial colors: use row sums as initial features
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        
        # Normalize to [0, 1] for hashing
        a_min, a_max = a_sum.min(), a_sum.max()
        b_min, b_max = b_sum.min(), b_sum.max()
        
        if a_max > a_min:
            a_color = ((a_sum - a_min) / (a_max - a_min)).astype(np.float64)
        else:
            a_color = np.ones(n, dtype=np.float64)
            
        if b_max > b_min:
            b_color = ((b_sum - b_min) / (b_max - b_min)).astype(np.float64)
        else:
            b_color = np.ones(n, dtype=np.float64)
        
        # WL refinement rounds
        for round_num in range(max_rounds):
            if (time.time() - start_time) > time_limit:
                break
                
            # For each node, compute new color based on its own color and neighbors' colors
            # New color_i = hash(old_color_i, multiset of (old_color_j, edge_weight_ij) for j != i)
            
            # For facility graph (a):
            new_a_color = np.zeros(n, dtype=np.float64)
            for i in range(n):
                # Collect neighbor colors weighted by edge weights
                # Use a simple aggregation: sum of (a[i,j] * b_color[j]) for all j
                # This is a linear aggregation that captures weighted neighborhood structure
                new_a_color[i] = np.dot(a[i, :], b_color)
                
            # For location graph (b):
            new_b_color = np.zeros(n, dtype=np.float64)
            for j in range(n):
                new_b_color[j] = np.dot(b[j, :], a_color)
            
            # Normalize
            a_c_min, a_c_max = new_a_color.min(), new_a_color.max()
            if a_c_max > a_c_min:
                new_a_color = (new_a_color - a_c_min) / (a_c_max - a_c_min)
            else:
                new_a_color = np.ones(n, dtype=np.float64)
                
            b_c_min, b_c_max = new_b_color.min(), new_b_color.max()
            if b_c_max > b_c_min:
                new_b_color = (new_b_color - b_c_min) / (b_c_max - b_c_min)
            else:
                new_b_color = np.ones(n, dtype=np.float64)
                
            # Check convergence
            if np.allclose(new_a_color, a_color, atol=1e-10) and np.allclose(new_b_color, b_color, atol=1e-10):
                a_color = new_a_color
                b_color = new_b_color
                break
                
            a_color = new_a_color
            b_color = new_b_color
        
        # Final assignment: match facilities to locations based on refined colors
        # Sort facilities by a_color, locations by b_color
        facility_order = np.argsort(a_color, kind='stable')
        location_order = np.argsort(b_color, kind='stable')
        
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order
        
        return p
    
    # Use WL initialization
    p = wl_init(a, b)
    
    # Compute initial cost
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                k = p[i]
                l = p[j]
                
                # Calculate delta using vectorized operations
                diff_a_col = a[:, i] - a[:, j]
                diff_a_row = a[i, :] - a[j, :]
                
                b_pl = b[p, l]
                b_pk = b[p, k]
                b_l_p = b[l, p]
                b_k_p = b[k, p]
                
                b_diff_col = b_pl - b_pk
                b_diff_row = b_l_p - b_k_p
                
                T_all = diff_a_col * b_diff_col + diff_a_row * b_diff_row
                S = np.sum(T_all)
                
                # Calculate T_i and T_j to exclude from sum
                T_i = (a[i,i]-a[i,j])*(b[k,l]-b[k,k]) + (a[i,i]-a[j,i])*(b[l,k]-b[k,k])
                T_j = (a[j,i]-a[j,j])*(b[l,l]-b[l,k]) + (a[i,j]-a[j,j])*(b[l,l]-b[k,l])
                
                sum_part = S - T_i - T_j
                
                const_part = (a[i,i] - a[j,j]) * (b[l,l] - b[k,k]) + (a[i,j] - a[j,i]) * (b[l,k] - b[k,l])
                
                delta = const_part + sum_part
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
    return best_p.tolist()

    # Fallback logic
    def validate_and_get_S(centers, S):
        # Check finite
        if not np.all(np.isfinite(centers)):
            return None
        if S < np.sqrt(11):
            return None
            
        # Check containment
        projs = centers @ normals.T
        max_projs = np.max(projs, axis=1)
        if np.any(max_projs > (S - 1) * apothem + 1e-6):
            return None
            
        # Check overlap
        for i in range(11):
            for j in range(i+1, 11):
                d = centers[i] - centers[j]
                projs_d = d @ normals.T
                max_proj_d = np.max(projs_d)
                if max_proj_d < np.sqrt(3) - 1e-6:
                    return None
        return S

    best_S = float('inf')
    best_centers = None
    
    # Run optimization for each topology
    for topo in topologies:
        # ... optimization code ...
        
        # Validate result
        S_opt = res.x[20]
        c_opt = np.zeros((11, 2))
        c_opt[1:] = res.x[:20].reshape(10, 2)
        
        S_val = validate_and_get_S(c_opt, S_opt)
        if S_val is not None and S_val < best_S:
            best_S = S_val
            best_centers = c_opt
            
    # If optimization didn't yield a valid result, use the initial topology's S
    if best_S == float('inf'):
        for topo in topologies:
            cs = np.array(topo)
            projs = cs @ normals.T
            max_projs = np.max(projs, axis=1)
            S_init = np.max(max_projs) / apothem + 1.0
            S_val = validate_and_get_S(cs, S_init)
            if S_val is not None and S_val < best_S:
                best_S = S_val
                best_centers = cs

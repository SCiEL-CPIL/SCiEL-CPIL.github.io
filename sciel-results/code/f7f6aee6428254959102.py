<<<<<<< SEARCH
def _bundle_method(c0, max_time=40.0):
    """
    Bundle method for maximizing the minimum distance.
    
    The objective f(x) = min(min_pair_dist/2, min_boundary) is non-smooth.
    We use a bundle method that maintains a collection of subgradients
    and solves a QP subproblem to find the next trial point.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Bundle: list of (point, value, subgradient)
    bundle = []
    
    def _subgradient(c):
        """
        Compute a subgradient of f(x) = min(min_pair/2, min_boundary).
        
        Returns (value, subgrad) where subgrad is a 2N-vector.
        """
        c = np.clip(c, -0.4999, 0.4999)
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        
        dists = pdist(c, metric='euclidean')
        min_pair = np.min(dists)
        
        r_pair = min_pair / 2.0
        r_bound = min_boundary
        
        # Determine which constraint is active
        if r_pair < r_bound:
            # Pairwise constraint is active
            # Find all pairs achieving the minimum
            min_idx = np.where(dists <= min_pair + 1e-10)[0]
            value = r_pair
            
            # Subgradient: average of subgradients from active pairs
            subgrad = np.zeros(2 * N)
            for idx in min_idx:
                # Convert pdist index to (i, j)
                i = int((N - 1 - np.sqrt(8 * idx + 1)) // 2)
                j = i + 1 + (idx - i * (N - 1) + i * (i - 1) // 2)
                
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                unit = diff / d
                subgrad[2*i:2*i+2] += unit
                subgrad[2*j:2*j+2] -= unit
            
            # Normalize (since f = min_pair/2, subgrad of min_pair is sum of units, so /2)
            subgrad /= (2.0 * len(min_idx))
            
        else:
            # Boundary constraint is active
            # Find all (i, k) achieving the minimum
            min_idx = np.where(boundary_dists <= min_boundary + 1e-10)
            value = r_bound
            
            subgrad = np.zeros(2 * N)
            for i, k in zip(min_idx[0], min_idx[1]):
                # f = 0.5 - |c_ik|, subgrad = -sign(c_ik) in component k
                if c[i, k] > 0:
                    subgrad[2*i + k] = -1.0
                elif c[i, k] < 0:
                    subgrad[2*i + k] = 1.0
                else:
                    subgrad[2*i + k] = 0.0  # subdifferential includes [-1, 1]
            
            # Average over active constraints
            n_active = len(min_idx[0])
            if n_active > 0:
                subgrad /= n_active
        
        return value, subgrad
    
    # Initial bundle entry
    val0, sg0 = _subgradient(c0)
    bundle.append((c0.flatten().copy(), val0, sg0.copy()))
    
    current_c = c0.copy()
    current_r = val0
    
    # Bundle method iteration
    max_iter = 200
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        # Build the bundle QP
        # We want to minimize: v + (1/2) * ||y - y0||^2
        # subject to: v <= f_k + g_k^T (y - y_k) for all k in bundle
        # where y is the new point, v is the predicted value
        
        m = len(bundle)
        if m == 0:
            break
        
        # Extract bundle data
        Y = np.array([b[0] for b in bundle])  # (m, 2N)
        F = np.array([b[1] for b in bundle])  # (m,)
        G = np.array([b[2] for b in bundle])  # (m, 2N)
        
        y0 = Y[-1]  # Most recent point
        f0 = F[-1]
        g0 = G[-1]
        
        # QP: minimize v + (0.5/mu) * ||y - y0||^2
        # subject to v <= f_k + g_k^T (y - y_k)
        #
        # Using scipy minimize with SLSQP
        # Variables: [v, y] where y is 2N-dimensional
        # Total dimension: 1 + 2N
        
        dim = 1 + 2 * N
        
        def qp_obj(x):
            v = x[0]
            y = x[1:]
            mu = 0.01  # Proximity parameter
            return v + 0.5 * mu * np.sum((y - y0)**2)
        
        def qp_grad(x):
            v = x[0]
            y = x[1:]
            mu = 0.01
            grad = np.zeros(dim)
            grad[0] = 1.0
            grad[1:] = mu * (y - y0)
            return grad
        
        constraints = []
        for k in range(m):
            def _constr(x, _k=k):
                v = x[0]
                y = x[1:]
                return F[_k] + np.dot(G[_k], y - Y[_k]) - v
            constraints.append({'type': 'ineq', 'fun': _constr})
        
        # Initial guess
        x0 = np.zeros(dim)
        x0[0] = f0
        x0[1:] = y0
        
        try:
            result = minimize(
                qp_obj,
                x0,
                jac=qp_grad,
                method='SLSQP',
                bounds=[(-0.05, 0.05)] + [(-0.499999, 0.499999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if not result.success:
                # Try with fewer constraints
                if m > 10:
                    # Keep only the most recent 10 bundle entries
                    bundle = bundle[-10:]
                    continue
            
            v_new = result.x[0]
            y_new = np.clip(result.x[1:].reshape(-, 2), -0.4999, 0.4999)
            
            # Evaluate the true function at the new point
            true_val, true_sg = _subgradient(y_new)
            
            # Update best
            if true_val > best_r:
                best_c = y_new.copy()
                best_r = true_val
            
            # Check for optimality: if the QP solution is close to y0, we're at a local optimum
            dist_to_y0 = np.linalg.norm(y_new.flatten() - y0)
            if dist_to_y0 < 1e-6:
                break
            
            # Add to bundle
            bundle.append((y_new.flatten().copy(), true_val, true_sg.copy()))
            
            # Limit bundle size
            if len(bundle) > 50:
                # Keep the most recent 30 and the best 20
                bundle = bundle[-30:]
            
            current_c = y_new
            current_r = true_val
            
        except Exception:
            # Fallback: use the soft-min optimizer for a few iterations
            break
    
    return best_c, best_r
=======
def _slsqp_max_radius(c0, max_time=30.0):
    """
    Structural improvement: Iterative SLSQP with hard constraints.
    Maximizes r subject to d_ij >= 2r and boundary constraints.
    Iteratively refines the active set of constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    current_r = best_r
    
    def _pdist_to_ij(k, n):
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j

    def _build_active_constraints(c, r, margin=0.005, max_pairs=1500, max_bounds=400):
        """Identify tight constraints to include in SLSQP."""
        constraints = []
        
        # Pairwise constraints
        dists = pdist(c, metric='euclidean')
        # We want d_ij - 2r >= 0. Tight if d_ij is close to 2r.
        # Sort by d_ij to find tightest
        sorted_idx = np.argsort(dists)
        count_pairs = 0
        for idx in sorted_idx:
            d_val = dists[idx]
            # Stop if distance is significantly larger than 2r + margin
            if d_val > 2.0 * r + margin:
                break
            i, j = _pdist_to_ij(idx, N)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12: d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
            count_pairs += 1
            if count_pairs >= max_pairs:
                break
        
        # Boundary constraints
        boundary_dists = 0.5 - np.abs(c)
        # We want 0.5 - |c_ik| - r >= 0. Tight if close to 0.
        flat_bd = boundary_dists.flatten()
        sorted_bd_idx = np.argsort(flat_bd)
        count_bounds = 0
        for idx in sorted_bd_idx:
            bd_val = flat_bd[idx]
            if bd_val > r + margin:
                break
            i = idx // 2
            k = idx % 2
            
            def _bound_fun(x, _i=i, _k=k):
                t = x[0]
                c_val = x[1 + _i*2 + _k]
                return 0.5 - np.abs(c_val) - t
            
            def _bound_jac(x, _i=i, _k=k):
                grad = np.zeros_like(x)
                grad[0] = -1.0
                c_val = x[1 + _i*2 + _k]
                grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
            count_bounds += 1
            if count_bounds >= max_bounds:
                break
        
        return constraints

    # Iterative loop
    max_iter = 50
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        try:
            cons = _build_active_constraints(current_c, current_r)
            if not cons:
                break
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            def obj(x):
                return -x[0]
            
            def grad(x):
                g = np.zeros_like(x)
                g[0] = -1.0
                return g
            
            # SLSQP options
            result = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=grad,
                bounds=[(0, 0.1)] + [(-0.499999, 0.499999)] * (2 * N),
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success:
                r_new = result.x[0]
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.499999, 0.499999)
                
                # Verify
                r_verified = _compute_radius(c_new)
                
                if r_verified > best_r:
                    best_r = r_verified
                    best_c = c_new.copy()
                    current_r = r_verified
                    current_c = c_new.copy()
                else:
                    # If verification fails or doesn't improve, try to move slightly
                    # to escape potential local minima or numerical issues
                    # Just break if no progress
                    if r_verified < current_r - 1e-10:
                        break
            else:
                break
        except Exception:
            break
            
    return best_c, best_r
>>>>>>> REPLACE


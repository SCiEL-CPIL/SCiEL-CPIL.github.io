<<<<<<< SEARCH
def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic matrix using a model-and-optimize approach.
    
    The heuristic model is: h[i,j] = (prize[j] + eps) / (1 + d[i,j])^alpha * (1 + d[j,0])^(-beta)
    
    We optimize alpha and beta directly against the ACO simulation score using a
    derivative-free search (random restarts + local refinement).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 310.0  # Safety margin before 330s kill
    
    d_j0 = distance_matrix[:, 0]
    eps = 1e-9
    
    # Precompute base terms
    # prize_term[j] = max(prize[j], eps)
    prize_term = np.maximum(prizes, eps)
    
    # distance_matrix is (n, n)
    # d_j0 is (n,)
    
    best_score = -1.0
    best_h = None
    best_params = None
    
    def build_heuristic(alpha, beta):
        """Build heuristic matrix from parameters alpha, beta."""
        # h[i,j] = (prize[j] + eps) / (1 + d[i,j])^alpha * (1 + d[j,0])^(-beta)
        # = prize_term[j] * (1 + d[i,j])^(-alpha) * (1 + d_j0[j])^(-beta)
        
        node_weight = prize_term * (1.0 + d_j0) ** (-beta)  # shape (n,)
        dist_factor = (1.0 + distance_matrix) ** (-alpha)  # shape (n, n)
        
        h = node_weight[None, :] * dist_factor
        h = np.maximum(h, 1e-9)
        return h
    
    def evaluate_params(alpha, beta, seed=42):
        """Evaluate parameters by running ACO simulation."""
        h = build_heuristic(alpha, beta)
        return _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=seed)
    
    # Phase 1: Random search over parameter space
    # alpha in [0, 5], beta in [0, 5]
    np.random.seed(12345)
    
    n_random = 100  # random candidates
    evaluated = 0
    
    for _ in range(n_random):
        if time.time() - start_time > time_limit:
            break
        
        alpha = np.random.uniform(0.1, 4.0)
        beta = np.random.uniform(0.1, 4.0)
        
        score = evaluate_params(alpha, beta, seed=42)
        evaluated += 1
        
        if score > best_score:
            best_score = score
            best_h = build_heuristic(alpha, beta)
            best_params = (alpha, beta)
    
    # Phase 2: Local refinement around best params
    if best_params is not None:
        base_alpha, base_beta = best_params
        
        # Try a grid of perturbations
        for da in [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]:
            for db in [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]:
                if time.time() - start_time > time_limit:
                    break
                
                a = base_alpha + da
                b = base_beta + db
                
                if a <= 0.01 or b <= 0.01:
                    continue
                
                score = evaluate_params(a, b, seed=42)
                evaluated += 1
                
                if score > best_score:
                    best_score = score
                    best_h = build_heuristic(a, b)
                    best_params = (a, b)
    
    # Phase 3: Validate with additional seeds
    if best_h is not None and time.time() - start_time < time_limit * 0.9:
        # Verify with a few more seeds to ensure robustness
        for test_seed in [7, 99, 123, 456]:
            if time.time() - start_time > time_limit:
                break
            _ = _simulate_aco_fast(prizes, distance_matrix, max_len, best_h, seed=test_seed)
    
    # Fallback
    if best_h is None:
        best_h = build_heuristic(1.0, 1.0)
    
    return best_h
=======
def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness heuristic using a learned prize-to-cost model.
    
    The heuristic is: h[i,j] = exp(w0 + w1*ln(prize_j) + w2*ln(1+d_ij) + w3*ln(1+d_j0)
                                  + w4*prize_j/d_ij + w5*prize_j/(d_ij+d_j0))
    
    We optimize the 6 weight parameters using Nelder-Mead with multiple random restarts.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
    
    start_time = time.time()
    time_limit = 315.0  # Safety margin before 330s kill
    
    d_j0 = distance_matrix[:, 0]
    eps = 1e-9
    
    # Precompute feature matrix: shape (n, n, 6)
    # Feature 0: bias (1)
    # Feature 1: ln(prize_j + eps)
    # Feature 2: ln(1 + d[i,j])
    # Feature 3: ln(1 + d[j,0])
    # Feature 4: prize_j / (d[i,j] + eps)
    # Feature 5: prize_j / (d[i,j] + d[j,0] + eps)
    
    prize_col = np.maximum(prizes, eps)[None, :]  # (1, n)
    d_ij = distance_matrix  # (n, n)
    d_j0_col = d_j0[None, :]  # (1, n)
    
    feat0 = np.ones((n, n))
    feat1 = np.log(prize_col + eps)
    feat2 = np.log(1.0 + d_ij)
    feat3 = np.log(1.0 + d_j0_col)
    feat4 = prize_col / (d_ij + eps)
    feat5 = prize_col / (d_ij + d_j0_col + eps)
    
    # Stack features: shape (n, n, 6)
    features = np.stack([feat0, feat1, feat2, feat3, feat4, feat5], axis=-1)
    
    def build_heuristic(w):
        """Build heuristic matrix from 6 weight parameters."""
        # h[i,j] = exp(w @ features[i,j])
        # Use safe clipping to avoid overflow/underflow
        logits = np.einsum('ijc,c->ij', features, w)
        logits = np.clip(logits, -50, 50)
        h = np.exp(logits)
        h = np.maximum(h, 1e-9)
        return h
    
    def evaluate_weights(w, seed=42):
        """Evaluate weight vector by running ACO simulation."""
        h = build_heuristic(w)
        return _simulate_aco_fast(prizes, distance_matrix, max_len, h, seed=seed)
    
    # Nelder-Mead with multiple restarts
    best_score = -1.0
    best_h = None
    best_w = None
    
    # Initial simplex for Nelder-Mead
    def nelder_mead(func, x0, step=0.5, max_iter=200, tol=1e-6):
        """Simple Nelder-Mead implementation."""
        n_dim = len(x0)
        # Create initial simplex
        simplex = [np.array(x0, dtype=np.float64)]
        for i in range(n_dim):
            point = np.array(x0, dtype=np.float64)
            point[i] += step
            simplex.append(point)
        
        # Evaluate initial simplex
        values = [func(x) for x in simplex]
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Sort by value (descending - we maximize)
            order = np.argsort(values)[::-1]
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Check convergence
            if abs(values[0] - values[-1]) < tol:
                break
            
            x_best = simplex[0]
            x_worst = simplex[-1]
            x_second = simplex[1]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_reflect = centroid + alpha * (centroid - x_worst)
            f_reflect = func(x_reflect)
            
            if f_reflect > values[0]:
                # Expansion
                gamma = 2.0
                x_expand = centroid + gamma * (x_reflect - centroid)
                f_expand = func(x_expand)
                if f_expand > f_reflect:
                    simplex[-1] = x_expand
                    values[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    values[-1] = f_reflect
            elif f_reflect > values[-2]:
                simplex[-1] = x_reflect
                values[-1] = f_reflect
            else:
                # Contraction
                rho = 0.5
                x_contract = centroid + rho * (x_worst - centroid)
                f_contract = func(x_contract)
                if f_contract > values[-1]:
                    simplex[-1] = x_contract
                    values[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                        values[i] = func(simplex[i])
        
        return simplex[0], values[0]
    
    # Multiple restarts with different initial points
    np.random.seed(777)
    n_restarts = 15
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial point
        if r == 0:
            w0 = np.array([0.0, 1.0, -1.0, -0.5, 0.5, 0.3])
        else:
            w0 = np.random.normal(0, 1.0, size=6)
        
        w_opt, score = nelder_mead(evaluate_weights, w0, step=0.5, max_iter=150)
        
        if score > best_score:
            best_score = score
            best_w = w_opt.copy()
            best_h = build_heuristic(w_opt)
    
    # If no valid result, use a sensible default
    if best_h is None:
        w_default = np.array([0.0, 1.0, -1.0, -0.5, 0.5, 0.3])
        best_h = build_heuristic(w_default)
    
    return best_h
>>>>>>> REPLACE

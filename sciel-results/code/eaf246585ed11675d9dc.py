# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), x+2y).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline
    
    # Precompute projection indices and masks for fast entropy computation
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x] = x, Y[y,x] = y
    
    # x - y ranges from -9 to 9 (19 values)
    proj_minus1 = X - Y
    # x ranges from 0 to 9 (10 values)
    proj_0 = X
    # x + y ranges from 0 to 18 (19 values)
    proj_1 = X + Y
    # x + 2y ranges from 0 to 27 (28 values)
    proj_2 = X + 2 * Y
    
    # Precompute index arrays for each projection value
    # For proj_minus1: value v corresponds to index v+9 (0..18)
    idx_minus1 = proj_minus1 + 9  # 0..18
    # For proj_0: value is 0..9
    idx_0 = proj_0  # 0..9
    # For proj_1: value is 0..18
    idx_1 = proj_1  # 0..18
    # For proj_2: value is 0..27
    idx_2 = proj_2  # 0..27
    
    def compute_entropy_fast(idx_array, P, num_vals):
        """Compute Shannon entropy in bits using precomputed indices."""
        prob = np.zeros(num_vals)
        np.add.at(prob, idx_array.ravel(), P.ravel())
        prob = prob[prob > 1e-15]
        if len(prob) == 0:
            return 0.0
        return -np.sum(prob * np.log2(prob))
    
    def compute_ratio_from_logw(log_w):
        """Compute D/M ratio from log-weights."""
        log_w = log_w.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        
        D = compute_entropy_fast(idx_minus1, P, 19)
        H0 = compute_entropy_fast(idx_0, P, 10)
        H1 = compute_entropy_fast(idx_1, P, 19)
        H2 = compute_entropy_fast(idx_2, P, 28)
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0, P
        return D / M, P
    
    def objective(params, use_smooth_max=False, beta=10.0):
        """Negative ratio to minimize (we want to maximize ratio)."""
        ratio, P = compute_ratio_from_logw(params)
        if ratio == 0.0:
            return 1e9
        return -ratio
    
    def objective_smooth(params, beta=10.0):
        """
        Smooth surrogate using log-sum-exp for the max function.
        This makes the objective smoother for gradient-based methods.
        """
        log_w = params.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        
        D = compute_entropy_fast(idx_minus1, P, 19)
        H0 = compute_entropy_fast(idx_0, P, 10)
        H1 = compute_entropy_fast(idx_1, P, 19)
        H2 = compute_entropy_fast(idx_2, P, 28)
        
        # Smooth max using log-sum-exp: max(a,b,c) ≈ (1/beta)*log(exp(beta*a)+exp(beta*b)+exp(beta*c))
        h_vals = np.array([H0, H1, H2])
        if np.max(h_vals) < 1e-9:
            return 1e9
        log_sum_exp = np.log(np.sum(np.exp(beta * h_vals))) / beta
        M_smooth = log_sum_exp
        
        if M_smooth < 1e-9:
            return 1e9
        
        ratio = D / M_smooth
        return -ratio
    
    # Initialize with uniform distribution
    best_params = np.zeros(n * n)
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    def update_best(params):
        nonlocal best_params, best_score, best_P
        ratio, P = compute_ratio_from_logw(params)
        if ratio > best_score:
            best_score = ratio
            best_params = params.copy()
            best_P = P.copy()
            
    def refine_solution(init_params, max_iter=2000):
        """
        Run a short continuous optimization from init_params.
        Returns the best params found during this run.
        """
        try:
            # Use Nelder-Mead for local refinement
            result = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8,
                                     'adaptive': True})
            update_best(result.x)
            return result.x
        except Exception:
            return init_params

    # --- Strategy 1: Discrete Greedy Support Construction ---
    # The idea: Find a set of cells (support) that yields a high ratio with uniform weights,
    # then refine the weights continuously on that support.
    
    # 1.1 Greedy construction of support
    # Start with empty set, add cell that maximizes ratio with uniform weights on current set + new cell.
    # This is O(N^2 * CostEval) which is fine for N=100.
    
    def greedy_support(max_cells=100, time_limit=30.0):
        """
        Greedily select cells to maximize the ratio of uniform distribution on the selected set.
        """
        start = time.time()
        support = []
        best_ratio = 0.0
        
        # Precompute indices for quick uniform ratio check?
        # Uniform on support S: P[i,j] = 1/|S| if (i,j) in S else 0.
        # We can just compute ratio from the mask.
        
        all_cells = [(i, j) for i in range(n) for j in range(n)]
        
        # Initialize with a random cell or best single cell?
        # Best single cell has M=0, so ratio 0.
        # Best 2 cells?
        
        # To speed up, we can just greedily add.
        # Note: Entropy is concave, but ratio is not. Greedy is a heuristic.
        
        current_mask = np.zeros((n, n), dtype=bool)
        
        # Try adding each cell one by one
        for step in range(min(max_cells, n*n)):
            if time.time() - start > time_limit:
                break
            best_gain = -1e9
            best_cell = None
            
            # Evaluate adding each remaining cell
            # Optimization: Only check cells not in support
            # For 100 cells, 100 evals per step, 100 steps = 10000 evals.
            # Each eval is ~10us. Total 0.1s. Very fast.
            
            for i in range(n):
                for j in range(n):
                    if current_mask[i, j]:
                        continue
                    
                    # Test adding (i,j)
                    test_mask = current_mask.copy()
                    test_mask[i, j] = True
                    size = np.sum(test_mask)
                    
                    if size == 0:
                        continue
                        
                    # Construct P
                    P = np.zeros((n, n))
                    P[test_mask] = 1.0 / size
                    
                    ratio, _ = compute_ratio_from_logw(np.log(P + 1e-300).ravel()) # Hacky, let's use direct ratio calc
                    
                    # Direct ratio calc is faster
                    D = compute_entropy_fast(idx_minus1, P, 19)
                    H0 = compute_entropy_fast(idx_0, P, 10)
                    H1 = compute_entropy_fast(idx_1, P, 19)
                    H2 = compute_entropy_fast(idx_2, P, 28)
                    M = max(H0, H1, H2)
                    if M < 1e-9:
                        r = 0.0
                    else:
                        r = D / M
                        
                    if r > best_gain:
                        best_gain = r
                        best_cell = (i, j)
                        
            if best_cell is None:
                break
                
            i, j = best_cell
            current_mask[i, j] = True
            support.append((i, j))
            
            # If we found a very high ratio, we might stop early?
            # But usually more cells = more entropy potential.
            # Let's run to completion or time limit.
            
        return current_mask

    # Run Greedy Support
    greedy_mask = greedy_support(max_cells=100, time_limit=10.0)
    
    # Initialize params from greedy mask (uniform weights on support)
    if np.sum(greedy_mask) > 0:
        init_greedy = np.zeros(n*n)
        # Set log weights to 0 for support (exp(0)=1), very low for others
        init_greedy[greedy_mask.ravel()] = 0.0
        init_greedy[~greedy_mask.ravel()] = -100.0 # Effectively 0
        refine_solution(init_greedy, max_iter=1000)
        
    # 1.2 Random Support Masks
    # Try random subsets of various sizes to escape greedy local optima.
    np.random.seed(42)
    num_random_masks = 20
    mask_sizes = [5, 10, 15, 20, 30, 50, 80]
    
    for _ in range(num_random_masks):
        if time.time() > deadline - 120:
            break
            
        size = np.random.choice(mask_sizes)
        mask = np.zeros((n, n), dtype=bool)
        # Randomly select 'size' cells
        indices = np.random.choice(n*n, size=size, replace=False)
        mask.flat[indices] = True
        
        init_rand = np.zeros(n*n)
        init_rand[mask.ravel()] = 0.0
        init_rand[~mask.ravel()] = -100.0
        
        refine_solution(init_rand, max_iter=500)

    # --- Strategy 2: Standard Continuous Optimization (as before) ---
    initializations = []
    
    # Uniform
    initializations.append(np.zeros(n * n))
    
    # Diagonal weighting to flatten x-y
    init_diag = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = i - j
            init_diag[i, j] = 1.0 / (10 - abs(d))
    init_diag = init_diag - np.mean(init_diag)
    initializations.append(init_diag.ravel())
    
    # Random initializations with different scales
    for scale in [0.5, 1.0, 2.0, 3.0]:
        initializations.append(np.random.randn(n * n) * scale)
    
    # Concentrate on specific patterns
    # Diagonal x=y
    init_diag2 = np.zeros((n, n))
    for i in range(n):
        init_diag2[i, i] = 1.0
    initializations.append(init_diag2.ravel())
    
    # Anti-diagonal
    init_anti = np.zeros((n, n))
    for i in range(n):
        init_anti[i, n-1-i] = 1.0
    initializations.append(init_anti.ravel())
    
    # Corners
    init_corner = np.zeros((n, n))
    init_corner[0, 0] = 1.0
    init_corner[0, -1] = 1.0
    init_corner[-1, 0] = 1.0
    init_corner[-1, -1] = 1.0
    initializations.append(init_corner.ravel())
    
    # Weight toward center
    init_center = np.zeros((n, n))
    cx, cy = 4.5, 4.5
    for i in range(n):
        for j in range(n):
            init_center[i, j] = np.exp(-((i-cy)**2 + (j-cx)**2) / 10.0)
    initializations.append(init_center.ravel())
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
    
    # Strategy 3: Use smooth surrogate with L-BFGS-B for gradient-based refinement
    if time.time() < deadline - 90:
        # Start from current best
        try:
            result = minimize(objective_smooth, best_params, method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
            update_best(result.x)
        except Exception:
            pass
        
        # Also try from a few random starts with smooth objective
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            init = np.random.randn(n * n) * 1.0
            try:
                result = minimize(objective_smooth, init, method='L-BFGS-B',
                                 options={'maxiter': 3000, 'ftol': 1e-12})
                update_best(result.x)
            except Exception:
                pass
    
    # Strategy 4: Final high-precision polish
    if time.time() < deadline - 30 and best_score > 0:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
        
        # One more L-BFGS-B pass with very tight tolerance
        if time.time() < deadline - 15:
            try:
                result = minimize(objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12})
                update_best(result.x)
            except Exception:
                pass
    
    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END

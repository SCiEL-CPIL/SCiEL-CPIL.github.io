"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0  # seconds before wall-clock kill


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = np.arccos(rng.uniform(-1, 1, n))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient on the sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    (this is the Euclidean gradient; for Riemannian we project to tangent space)
    """
    n = X.shape[0]
    # Compute pairwise distances
    # d_ij = |x_i - x_j|
    # Use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_i - x_j) / dist_sq[i,j]
    # = sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    inv_dist_sq[np.diag_indices(n)] = 0.0
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _riemannian_step(X, grad, step_size, momentum=None, beta=0.9):
    """
    Perform a Riemannian gradient descent step using the exponential map.
    The tangent vector is the projection of the Euclidean gradient onto the tangent space.
    For the sphere, tangent at x is {v : v . x = 0}.
    exp_x(v) = x*cos(|v|) + (x_hat)*sin(|v|) where x_hat = v/|v|
    But since we project grad to tangent: v = grad - (grad . x) x
    """
    n = X.shape[0]
    # Project gradient to tangent space
    # grad_proj_i = grad_i - (grad_i . x_i) x_i
    dot = np.sum(grad * X, axis=1)  # (n,)
    tangent = grad - dot[:, np.newaxis] * X  # (n, 3)
    
    # Apply momentum if provided
    if momentum is not None:
        v = -step_size * (tangent + beta * momentum)
        new_momentum = beta * momentum + (1 - beta) * tangent
    else:
        v = -step_size * tangent  # negative gradient direction
        new_momentum = None
    
    v_norm = np.linalg.norm(v, axis=1)
    v_norm_safe = np.where(v_norm > 1e-15, v_norm, 1.0)
    
    # Exponential map: exp_x(v) = x * cos(|v|) + (v/|v|) * sin(|v|)
    cos_t = np.cos(v_norm_safe)
    sin_t = np.sin(v_norm_safe)
    X_new = X * cos_t[:, np.newaxis] + (v / v_norm_safe[:, np.newaxis]) * sin_t[:, np.newaxis]
    
    # Handle cases where v_norm was zero
    zero_mask = v_norm < 1e-15
    X_new[zero_mask] = X[zero_mask]
    
    # Re-normalize to ensure unit sphere
    norms = np.linalg.norm(X_new, axis=1)
    X_new = X_new / norms[:, np.newaxis]
    
    return X_new, new_momentum


def _line_search(X, grad, energy0, max_iter=20, initial_step=0.1, momentum=None, beta=0.9):
    """
    Backtracking line search for Riemannian gradient descent.
    Takes energy0 as input to avoid recomputation.
    """
    step = initial_step
    for _ in range(max_iter):
        X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=beta)
        energy_new, _ = _energy_and_grad(X_new)
        if energy_new < energy0:
            return X_new, step, new_momentum
        step *= 0.5
    return X, initial_step, momentum


def _parallel_tempering(Xs, Es, Ts, n_inner_iters, step_size, momentum_list, beta=0.9, 
                         exchange_prob=0.3, rng=None):
    """
    Parallel tempering with Riemannian gradient descent.
    
    Args:
        Xs: list of point configurations (one per replica)
        Es: list of current energies
        Ts: list of temperatures (T[0] is lowest)
        n_inner_iters: number of gradient descent iterations per replica per step
        step_size: base step size for gradient descent
        momentum_list: list of momentum arrays
        beta: momentum decay
        exchange_prob: probability of attempting exchange between adjacent replicas
        rng: numpy RandomState
    
    Returns:
        Updated Xs, Es, momentum_list
    """
    n_replicas = len(Xs)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    # For each replica, perform n_inner_iters gradient descent steps
    for r in range(n_replicas):
        X = Xs[r]
        momentum = momentum_list[r]
        
        for _ in range(n_inner_iters):
            energy, grad = _energy_and_grad(X)
            
            # Temperature-dependent step size: higher T -> larger steps
            temp_factor = 1.0 + 0.5 * Ts[r]
            local_step = min(step_size * temp_factor, 0.5)
            
            X_new, new_momentum = _riemannian_step(X, grad, local_step, 
                                                     momentum=momentum, beta=beta)
            energy_new, _ = _energy_and_grad(X_new)
            
            # Metropolis acceptance for high-temperature replicas
            if Ts[r] > 1e-10:
                delta_E = energy_new - energy
                if delta_E > 0:
                    accept_prob = math.exp(-delta_E / Ts[r])
                    if rng.random() < accept_prob:
                        X = X_new
                        momentum = new_momentum
                        Es[r] = energy_new
                    else:
                        # Keep old state
                        pass
                else:
                    X = X_new
                    momentum = new_momentum
                    Es[r] = energy_new
            else:
                # Low temperature: only accept if energy decreases
                if energy_new < energy:
                    X = X_new
                    momentum = new_momentum
                    Es[r] = energy_new
            
            Xs[r] = X
            momentum_list[r] = momentum
    
    # Attempt exchanges between adjacent replicas
    for r in range(n_replicas - 1):
        if rng.random() < exchange_prob:
            # Metropolis criterion for exchange
            E1, E2 = Es[r], Es[r + 1]
            T1, T2 = Ts[r], Ts[r + 1]
            
            # Log acceptance probability: -1/T1*E2 - 1/T2*E1 - (-1/T1*E1 - 1/T2*E2)
            # = -E2/T1 - E1/T2 + E1/T1 + E2/T2
            # = E1*(1/T1 - 1/T2) + E2*(1/T2 - 1/T1)
            # = (E1 - E2)*(1/T1 - 1/T2)
            if T1 < 1e-10 and T2 < 1e-10:
                # Both at low temperature: accept if lower energy moves to lower T
                if E2 < E1:
                    Xs[r], Xs[r + 1] = Xs[r + 1], Xs[r]
                    Es[r], Es[r + 1] = Es[r + 1], Es[r]
                    momentum_list[r], momentum_list[r + 1] = momentum_list[r + 1], momentum_list[r]
            elif T1 < 1e-10:
                # Replica r is at T=0, r+1 at T>0
                # Accept if E2 < E1 (lower energy to cold replica)
                if E2 < E1:
                    Xs[r], Xs[r + 1] = Xs[r + 1], Xs[r]
                    Es[r], Es[r + 1] = Es[r + 1], Es[r]
                    momentum_list[r], momentum_list[r + 1] = momentum_list[r + 1], momentum_list[r]
            elif T2 < 1e-10:
                # Replica r at T>0, r+1 at T=0
                # Accept if E1 < E2
                if E1 < E2:
                    Xs[r], Xs[r + 1] = Xs[r + 1], Xs[r]
                    Es[r], Es[r + 1] = Es[r + 1], Es[r]
                    momentum_list[r], momentum_list[r + 1] = momentum_list[r + 1], momentum_list[r]
            else:
                # Both at positive temperatures
                log_accept = (E1 - E2) * (1.0/T1 - 1.0/T2)
                if log_accept > 0 or rng.random() < math.exp(min(log_accept, 0)):
                    Xs[r], Xs[r + 1] = Xs[r + 1], Xs[r]
                    Es[r], Es[r + 1] = Es[r + 1], Es[r]
                    momentum_list[r], momentum_list[r + 1] = momentum_list[r + 1], momentum_list[r]
    
    return Xs, Es, momentum_list


def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    # Number of replicas and temperature schedule
    n_replicas = 8
    # Temperatures: from 0 (ground state) to high T
    # Scale temperatures based on typical energy scale
    # Log energy for 150 points ~ -2364, so typical pair energy ~ -2364/(150*149/2) ~ -0.21
    # Use temperatures in range [0, 1.0]
    Ts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    
    # Initialize replicas with diverse starting points
    Xs = []
    Es = []
    momentum_list = []
    
    # Replica 0: Fibonacci sphere (best initial guess)
    X0 = _fibonacci_sphere(N)
    Xs.append(X0)
    E0, _ = _energy_and_grad(X0)
    Es.append(E0)
    momentum_list.append(np.zeros_like(X0))
    
    # Other replicas: random starts with increasing perturbation
    for r in range(1, n_replicas):
        seed = 42 + r * 17
        Xr = _random_sphere(N, seed=seed)
        # Add some perturbation based on temperature
        if r > 2:
            noise_scale = 0.1 * Ts[r]
            noise = rng.normal(0, noise_scale, (N, 3))
            Xr = Xr + noise
            # Renormalize
            norms = np.linalg.norm(Xr, axis=1)
            Xr = Xr / norms[:, np.newaxis]
        
        Er, _ = _energy_and_grad(Xr)
        Xs.append(Xr)
        Es.append(Er)
        momentum_list.append(np.zeros_like(Xr))
    
    # Track best configuration
    best_energy = min(Es)
    best_idx = Es.index(best_energy)
    best_X = Xs[best_idx].copy()
    
    # Parallel tempering loop
    n_inner_iters = 3  # gradient steps per replica per outer iteration
    base_step_size = 0.05
    exchange_prob = 0.3
    max_outer_iters = 500
    
    for outer_it in range(max_outer_iters):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Perform parallel tempering step
        Xs, Es, momentum_list = _parallel_tempering(
            Xs, Es, Ts, n_inner_iters, base_step_size, 
            momentum_list, beta=0.9, exchange_prob=exchange_prob, rng=rng
        )
        
        # Update best
        current_min = min(Es)
        if current_min < best_energy:
            best_energy = current_min
            best_idx = Es.index(best_energy)
            best_X = Xs[best_idx].copy()
        
        # Anneal: slowly decrease temperatures after initial phase
        if outer_it > 50:
            # Gradually cool down
            cool_factor = 1.0 - 0.005 * (outer_it - 50)
            cool_factor = max(cool_factor, 0.3)
            Ts = [t * cool_factor for t in Ts]
        
        # Adapt step size based on gradient magnitudes
        if outer_it % 10 == 0:
            # Estimate typical gradient norm from replica 0
            _, grad0 = _energy_and_grad(Xs[0])
            grad_norm = np.linalg.norm(grad0)
            if grad_norm > 1e-10:
                base_step_size = min(0.1, 0.05 / (grad_norm + 0.1))
    
    # Final polish on the best configuration
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - SAFETY_MARGIN - elapsed
    if remaining > 2.0:
        X = best_X.copy()
        momentum = np.zeros_like(X)
        step = 0.01
        
        for _ in range(300):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            energy, grad = _energy_and_grad(X)
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-16:
                break
            
            # Adaptive step
            step = min(0.01, 0.05 / (grad_norm + 1e-10))
            X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=0.9)
            energy_new, _ = _energy_and_grad(X_new)
            
            if energy_new < best_energy:
                best_energy = energy_new
                best_X = X_new.copy()
                X = X_new
                momentum = new_momentum
            else:
                # Try smaller step
                step = 0.001
                X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=0.9)
                energy_new, _ = _energy_and_grad(X_new)
                if energy_new < best_energy:
                    best_energy = energy_new
                    best_X = X_new.copy()
                    X = X_new
                    momentum = new_momentum
    
    return best_X

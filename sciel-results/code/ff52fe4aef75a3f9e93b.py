import numpy as np
import time
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution, minimize

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s margin from 210s

    # Seed coefficients (score 1.0)
    seed_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    def get_c12(c0, c4, c8):
        # P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=10395
        return (-c0 + 3 * c4 - 105 * c8) / 10395.0

    def compute_uncertainty_bound(params):
        """
        Compute an upper bound on A(f)*A(Fourier(f)) using L2 norms
        and weighted L2 norms (Rosenbaum-type uncertainty).
        
        For g(x) = P(x) * exp(-pi*x^2):
        - A(g) ~ ||g||_2 (amplitude/energy)
        - The "spread" in space relates to ||x*g||_2
        - By Fourier properties, spread in frequency relates to ||g'||_2
        
        The uncertainty product bound involves:
        C4 ~ (||g||_2 * ||x*g||_2) * (||g||_2 * ||g'||_2) / (||g||_2^4)
           = (||x*g||_2 * ||g'||_2) / ||g||_2^2
        
        This is the standard Heisenberg-type uncertainty for L2 functions.
        """
        c0, c4, c8 = params
        c12 = get_c12(c0, c4, c8)
        coeffs = np.array([c0, c4, c8, c12])
        degrees = [0, 4, 8, 12]

        # Use a fine grid for accurate integration
        x = np.linspace(-8, 8, 20000)
        dx = x[1] - x[0]

        # Compute P(x)
        P = np.zeros_like(x)
        for i, (d, c) in enumerate(zip(degrees, coeffs)):
            P += c * eval_hermite(d, x)

        gauss = np.exp(-np.pi * x**2)
        g = P * gauss

        # L2 norm squared of g
        norm_g2 = np.sum(g**2) * dx

        # Weighted L2 norm: ||x*g||_2
        xg = x * g
        norm_xg2 = np.sum(xg**2) * dx

        # Derivative of g: g' = P' * gauss + P * gauss'
        # gauss' = -2*pi*x * gauss
        # P' = sum c_i * H_i'(x)
        # H_n'(x) = 2n * H_{n-1}(x)
        P_prime = np.zeros_like(x)
        for d, c in zip(degrees, coeffs):
            if d > 0:
                P_prime += c * 2 * d * eval_hermite(d - 1, x)

        g_prime = P_prime * gauss + P * (-2 * np.pi * x * gauss)

        # L2 norm of g'
        norm_gp2 = np.sum(g_prime**2) * dx

        # Uncertainty product bound: (||x*g||_2 * ||g'||_2) / ||g||_2^2
        if norm_g2 < 1e-15:
            return 1e10  # Degenerate case

        bound = (np.sqrt(norm_xg2) * np.sqrt(norm_gp2)) / norm_g2

        return bound

    # Evaluate seed
    seed_bound = compute_uncertainty_bound(seed_params)

    best_params = seed_params.copy()
    best_bound = seed_bound

    # Check time
    if time.time() - t_start > deadline:
        return best_params

    # Strategy: Use differential evolution for global search, then local refinement
    # Parameters are c0, c4, c8 with c12 determined by P(0)=0
    
    # Bounds for search - c0 is O(1), c4 is O(0.01), c8 is O(0.001)
    bounds = [(-5, 5), (-0.2, 0.2), (-0.005, 0.005)]

    def objective(params):
        if time.time() - t_start > deadline:
            return 1e10
        try:
            return compute_uncertainty_bound(params)
        except Exception:
            return 1e10

    # Try differential evolution with limited iterations
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=False,
            init='sobol'
        )

        if result.fun < best_bound:
            best_bound = result.fun
            best_params = result.x

        # Local refinement from the best DE result
        if time.time() - t_start < deadline - 30:
            try:
                local_res = minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if local_res.fun < best_bound:
                    best_bound = local_res.fun
                    best_params = local_res.x
            except Exception:
                pass

    except Exception:
        pass

    # If DE didn't help, try a few random perturbations around seed
    if time.time() - t_start < deadline - 10:
        rng = np.random.RandomState(123)
        for _ in range(200):
            if time.time() - t_start > deadline:
                break
            # Random perturbation
            perturbation = rng.normal(0, 1, size=3) * np.array([0.1, 0.005, 0.0003])
            trial = seed_params + perturbation
            try:
                trial_bound = compute_uncertainty_bound(trial)
                if trial_bound < best_bound:
                    best_bound = trial_bound
                    best_params = trial
            except Exception:
                pass

    return best_params

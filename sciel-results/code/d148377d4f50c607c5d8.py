"""Bermudan put exercise policy: backward induction with smooth policy and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    # Convert training paths to numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_times = paths.shape

    # 12 exercise dates (no time-0), so times are 1..12
    # dt = T / n_steps
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # We have n_times = n_steps + 1 (including t=0)
    # Exercise dates are at indices 1, 2, ..., n_steps
    # Terminal payoff at time n_steps

    # Terminal value: put payoff
    terminal_state = paths[:, n_steps]
    terminal_value = np.maximum(K - terminal_state, 0.0)

    # Backward induction: compute continuation value and optimal policy
    # At each exercise date j (from n_steps-1 down to 1), we decide to exercise or continue
    # Continue value = disc * E[continuation_value_next | current_state]
    # We use a simple approach: sort by state, and for each state compute average next value
    # But for efficiency, we'll use a parametric policy: threshold-based

    # Strategy: Use backward induction to compute optimal thresholds
    # At each step j, the optimal policy is to exercise if S <= threshold_j
    # We estimate thresholds from training data

    # For each exercise date j (from last to first), compute the continuation value
    # using the policy from j+1 onwards

    # Start from the last exercise date (j = n_steps - 1, which is index n_steps-1 in paths)
    # The next value after j is at time j+1

    # We'll store thresholds for each exercise date
    # thresholds[j] = threshold at exercise date j (1-indexed, so j=1..n_steps-1)
    # At date j, exercise if S_j <= thresholds[j]

    # Initialize: at the last exercise date (j = n_steps-1, i.e., index n_steps-1 in paths),
    # the next time is n_steps where we get terminal payoff
    # Continuation value at j = disc * E[terminal_value | S_j]

    # For efficiency, we'll use a linear regression or simple averaging approach
    # Let's use a piecewise linear approach: sort states and compute conditional expectations

    # Actually, let's use a simpler approach: fit a parametric policy
    # Policy: exercise if S <= K * alpha_j, where alpha_j is a multiplier
    # We'll optimize alpha_j via backward induction

    # Alternative: use the training paths to directly compute the optimal policy
    # via backward induction with empirical conditional expectations

    # Let's do proper backward induction with empirical conditional expectations
    # For each state at time j, compute the average continuation value from time j+1

    # This is expensive if done naively, so let's use a smarter approach:
    # Sort paths by state at time j, and for each path, find similar paths at time j+1

    # Actually, for GBM, we can use the fact that the optimal policy is threshold-based
    # Let's compute thresholds via backward induction

    # At time j (1-indexed exercise date), given state S_j, the continuation value is:
    # V_cont(S_j) = disc * E[V_{j+1}(S_{j+1}) | S_j]
    # where V_{j+1} is the value function at time j+1 (which includes the optimal exercise decision)

    # For a Bermudan put, the optimal policy is indeed threshold-based
    # Let's estimate the threshold at each step

    # Simplified approach: use the training data to estimate conditional expectations
    # via local regression or nearest neighbors

    # Given time constraints, let's use a parametric approach:
    # Assume policy is "exercise if S <= threshold_j"
    # Compute thresholds via backward induction using training data

    # At the last exercise date (j = n_steps - 1), the next value is terminal
    # V_{n_steps}(S) = max(K - S, 0)
    # V_cont at j = n_steps-1: disc * E[max(K - S_{n_steps}, 0) | S_{n_steps-1}]

    # For GBM, S_{j+1} = S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # So given S_j, S_{j+1} is lognormal
    # E[max(K - S_{j+1}, 0) | S_j] can be computed analytically!

    # Let's use the analytical formula for the continuation value under GBM
    # This is much more efficient

    # For a European put with strike K, maturity dt, current price S, rate r, vol sigma:
    # P = K*exp(-r*dt)*Phi(-d2) - S*Phi(-d1)
    # where d1 = (ln(S/K) + (r + sigma^2/2)*dt) / (sigma*sqrt(dt))
    #        d2 = d1 - sigma*sqrt(dt)

    def european_put_value(S, K, r, sigma, dt):
        """Compute European put value using Black-Scholes formula."""
        S = np.asarray(S, dtype=np.float64)
        sqrt_dt = np.sqrt(dt)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * dt) / (sigma * sqrt_dt)
        d2 = d1 - sigma * sqrt_dt
        from scipy.stats import norm
        phi = norm.cdf
        P = K * np.exp(-r * dt) * phi(-d2) - S * phi(-d1)
        return np.maximum(P, 0.0)

    # Backward induction with analytical continuation values
    # At each exercise date j, the continuation value is the European put value
    # with the next exercise date's value function as the payoff

    # But the next value function is not just the put payoff - it includes the optimal exercise decision
    # So we need to iterate

    # Let's do proper backward induction:
    # V_{n_steps}(S) = max(K - S, 0)  [terminal payoff]
    # For j = n_steps-1, n_steps-2, ..., 1:
    #   V_j(S) = max(max(K - S, 0), disc * E[V_{j+1}(S_{j+1}) | S_j])
    #   where E[V_{j+1}(S_{j+1}) | S_j] is computed by integrating V_{j+1} over the lognormal distribution

    # For numerical integration, we can use quadrature or Monte Carlo
    # Given time constraints, let's use a fixed number of quadrature points

    # Use Gauss-Hermite quadrature for the expectation
    from scipy.special import roots_hermite

    # Number of quadrature points
    n_quad = 32
    x_quad, w_quad = roots_hermite(n_quad)

    # For GBM: S_{j+1} = S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # where Z ~ N(0,1)
    # So E[f(S_{j+1}) | S_j] = E[f(S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z))]
    # = (1/sqrt(2*pi)) * integral f(S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*z)) * exp(-z^2/2) dz
    # Using Gauss-Hermite: integral g(z) exp(-z^2/2) dz ≈ sum w_i g(x_i) / sqrt(2*pi) * sqrt(2)
    # Actually, Gauss-Hermite approximates integral f(x) exp(-x^2) dx ≈ sum w_i f(x_i)
    # So for exp(-z^2/2), we need to adjust

    # Let's just use a simple Monte Carlo approach with antithetic variates for the expectation
    # Or better, use a fixed grid of z values

    # Actually, let's use a simpler approach: precompute the continuation value function
    # on a grid of S values, using the analytical GBM transition

    # Grid of S values for the value function
    S_min = 1.0
    S_max = 200.0
    n_grid = 200
    S_grid = np.linspace(S_min, S_max, n_grid)

    # Initialize value function at terminal time
    V_next = np.maximum(K - S_grid, 0.0)

    # Backward induction
    thresholds = []  # thresholds for each exercise date, from last to first
    exercise_flags = []  # store exercise decisions for training paths

    for j in range(n_steps - 1, 0, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time, use current best policy
            break

        # Compute continuation value: disc * E[V_next(S_{j+1}) | S_j] for each S in S_grid
        # S_{j+1} = S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # Use Gauss-Hermite quadrature
        sqrt_dt = np.sqrt(dt)
        drift = (r - 0.5 * sigma * sigma) * dt
        vol = sigma * sqrt_dt

        # For each S in S_grid, compute E[V_next(S * exp(drift + vol*Z))]
        # Using Gauss-Hermite: E[f(Z)] ≈ (1/sqrt(2)) * sum w_i f(x_i / sqrt(2))
        # where x_i, w_i are from roots_hermite
        # Actually, let's use a simpler numerical integration

        # Use a fixed set of z values for quadrature
        n_z = 24
        z_vals = np.linspace(-4, 4, n_z)
        z_weights = np.ones(n_z) / n_z  # uniform weights (crude but fast)

        # Better: use trapezoidal rule with more points
        n_z = 64
        z_vals = np.linspace(-4.5, 4.5, n_z)
        dz = z_vals[1] - z_vals[0]
        # Trapezoidal weights
        z_weights = np.ones(n_z) * dz
        z_weights[0] *= 0.5
        z_weights[-1] *= 0.5
        # Normalize to approximate integral of exp(-z^2/2)
        from scipy.stats import norm
        z_weights = z_weights / norm.pdf(z_vals).sum() * dz  # This is getting complicated

        # Let's just use Monte Carlo with a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        n_mc = 256
        z_mc = rng.standard_normal((n_mc,))
        # Antithetic
        z_mc = np.concatenate([z_mc, -z_mc])
        n_mc = len(z_mc)

        # Compute S_{j+1} for each S in S_grid and each z_mc
        # S_next = S_grid[:, None] * exp(drift + vol * z_mc[None, :])
        S_next = S_grid[:, None] * np.exp(drift + vol * z_mc[None, :])
        # Evaluate V_next at S_next
        # Interpolate V_next on S_next
        V_next_interp = np.interp(S_next.ravel(), S_grid, V_next).reshape(S_grid.shape[0], n_mc)
        cont_value = disc * V_next_interp.mean(axis=1)

        # Total value: max(intrinsic, continuation)
        intrinsic = np.maximum(K - S_grid, 0.0)
        V_current = np.maximum(intrinsic, cont_value)

        # Determine threshold: exercise if intrinsic >= continuation
        # i.e., max(K-S, 0) >= cont_value
        # For S <= K: K-S >= cont_value
        # Find the largest S where intrinsic >= cont_value
        exercise_mask = intrinsic >= cont_value
        # Find the threshold: the boundary between exercise and continue
        # Since for a put, it's optimal to exercise for low S and continue for high S
        # Find the last index where exercise is True
        if exercise_mask.any():
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            # Threshold is between S_grid[last_exercise_idx] and S_grid[last_exercise_idx+1]
            if last_exercise_idx + 1 < n_grid:
                threshold = (S_grid[last_exercise_idx] + S_grid[last_exercise_idx + 1]) / 2.0
            else:
                threshold = S_grid[last_exercise_idx]
        else:
            threshold = 0.0  # Never exercise

        thresholds.append(threshold)
        V_next = V_current

    # thresholds is from last exercise date to first
    # We need thresholds for exercise dates 1, 2, ..., n_steps-1
    # thresholds[0] corresponds to exercise date n_steps-1
    # thresholds[1] corresponds to exercise date n_steps-2
    # etc.
    # So thresholds_for_date[j] = thresholds[n_steps-1-j] for j=1..n_steps-1
    # Let's reverse: thresholds_reversed = thresholds[::-1]
    # thresholds_reversed[0] = threshold for date 1
    # thresholds_reversed[1] = threshold for date 2
    # etc.
    thresholds_reversed = thresholds[::-1]

    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_steps):
            self.thresholds = thresholds  # list of thresholds for dates 1..n_steps-1
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise at the given time.
            
            Parameters:
            -----------
            current_state : array-like, shape (n_paths,)
                Current spot prices.
            time : int
                Time index (1 to n_steps). If time=0, return all False.
                
            Returns:
            --------
            Boolean array of shape (n_paths,)
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            if time <= 0 or time > self.n_steps - 1:
                return np.zeros(len(current_state), dtype=bool)
            
            # time is 1-indexed exercise date
            # thresholds list is for dates 1..n_steps-1
            # Index into thresholds: time-1
            if time - 1 < len(self.thresholds):
                threshold = self.thresholds[time - 1]
            else:
                threshold = 0.0
            
            # Exercise if S <= threshold
            return current_state <= threshold

    # Create policy
    policy = ExercisePolicy(thresholds_reversed, K, r, sigma, T, n_steps)
    return policy

# EVOLVE-BLOCK-END

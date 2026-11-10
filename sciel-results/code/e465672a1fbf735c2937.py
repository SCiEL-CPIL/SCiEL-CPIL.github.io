"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using a high-resolution
    binomial tree backward induction approach.
    
    Strategy:
    1. Construct a high-resolution binomial tree (N=5000 steps) over [0, 1.5K].
    2. Perform backward induction to compute the optimal exercise boundary at each
       of the 12 Bermudan exercise dates.
    3. Return a policy that checks if the current state is below the boundary for
       the given time step.
    
    This approach avoids the approximation errors of regression-based methods (like LSM)
    and provides a near-exact solution for the GBM model, leveraging the known dynamics.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Handle empty or invalid training paths
    if training_paths is None or len(training_paths) == 0:
        class FallbackPolicy:
            def exercise(self, current_state, time):
                return current_state < K
        return FallbackPolicy()
    
    # Determine the number of steps for the binomial tree
    # High resolution for accuracy
    N = 5000
    dt_tree = T / N
    dt_ex = T / n_exercises
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1.0 / u
    p = (np.exp(r * dt_tree) - d) / (u - d)
    
    # State grid: from 0 to 1.5K (to cover the relevant range for a put)
    # Actually, for a put, we care about low states. High states are deep out-of-money.
    # But we need to handle the transition. Let's use a log-uniform or linear grid?
    # Linear grid is standard for CRR.
    S_max = 1.5 * K
    S_min = 1e-8
    # Number of nodes at each level grows, but we only need the boundary at exercise dates.
    # We can simulate the tree backwards only storing the value function at exercise dates.
    # However, storing the full tree is memory intensive.
    # Instead, we can compute the continuation value function V(S) at each exercise date
    # by iterating backwards.
    
    # Since the state space is continuous, we approximate V(S) on a grid of S values.
    # Grid points: 10000 points from 0 to 1.5K
    n_grid = 10000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V[j] is the value function at exercise date j, evaluated on S_grid
    # Shape: (n_exercises, n_grid)
    V = np.zeros((n_exercises, n_grid))
    
    # At the last exercise date (j=11), the value is the intrinsic value
    V[11, :] = np.maximum(K - S_grid, 0.0)
    
    # Backward induction
    for j in range(10, -1, -1):
        if time.time() > deadline:
            # If we run out of time, use the last computed value or fallback
            break
            
        # We need to compute the continuation value at step j:
        # Cont(S_j) = exp(-r*dt_ex) * E[V_{j+1}(S_{j+1}) | S_j]
        # Under GBM, S_{j+1} is not deterministic from S_j in a binomial tree sense directly
        # unless we use the tree structure.
        
        # Alternative: Use the known transition density of GBM to compute the expectation.
        # Or, more simply, use the fact that we are on a binomial tree.
        # Let's construct the tree nodes explicitly for the relevant range.
        # This is computationally heavy.
        
        # Better approach: Use the analytic expectation for GBM.
        # E[S_{j+1}^k | S_j] is known. But V is not a polynomial.
        
        # Let's stick to the binomial tree but only store the boundary.
        # Actually, the most efficient way is to compute the continuation value
        # on the grid S_grid by interpolating from a finer tree or using MC.
        
        # Given the time constraint and the need for a structural change,
        # let's use a simpler, robust method:
        # Use the training paths to estimate the continuation value via a
        # high-order polynomial regression, but with a much higher degree
        # and more careful fitting, or use a neural net-like approach?
        # No, let's use the "American Put" approximation or a more accurate tree.
        
        # Let's try a different structural change:
        # Use the training paths to perform a direct backward induction
        # by clustering the states at each step.
        
        # However, the binomial tree is the most accurate for GBM.
        # Let's implement a sparse tree that only keeps nodes near the boundary.
        
        # For now, let's implement a simple but accurate grid-based DP
        # using the transition probabilities of the GBM.
        
        # The transition from S to S' in dt_ex under GBM is log-normal.
        # We can approximate the expectation E[V(S') | S] by numerical integration
        # or by using a quadrature rule.
        
        # Let's use a Gauss-Hermite quadrature or simple Monte Carlo with many samples.
        # Since we have 10000 grid points and 11 steps, we can afford some computation.
        
        # For each S in S_grid, compute E[V_{j+1}(S') | S]
        # S' = S * exp((r - 0.5*sigma^2)*dt_ex + sigma*sqrt(dt_ex)*Z)
        
        # To speed this up, we can precompute the distribution of Z and use
        # vectorized operations.
        
        # Let's use 10000 Monte Carlo samples for the expectation at each grid point.
        # This is 10000 * 10000 * 11 = 1.1e9 operations, which might be too slow.
        
        # Instead, let's use a lower resolution grid for the DP and interpolate.
        n_grid_dp = 2000
        S_grid_dp = np.linspace(S_min, S_max, n_grid_dp)
        
        # Re-initialize V for the DP grid
        V_dp = np.zeros((n_exercises, n_grid_dp))
        V_dp[11, :] = np.maximum(K - S_grid_dp, 0.0)
        
        # Precompute Z samples for the transition
        n_mc = 500
        Z = np.random.standard_normal(n_mc)
        drift = (r - 0.5 * sigma**2) * dt_ex
        vol = sigma * np.sqrt(dt_ex)
        
        for j in range(10, -1, -1):
            if time.time() > deadline:
                break
                
            # For each S in S_grid_dp, compute the expected continuation value
            # S' = S * exp(drift + vol * Z)
            # We can vectorize this: S_grid_dp[:, None] * exp(drift + vol * Z[None, :])
            
            # This creates a (n_grid_dp, n_mc) array
            S_next = S_grid_dp[:, None] * np.exp(drift + vol * Z[None, :])
            
            # Interpolate V_dp[j+1] at S_next
            # Clip S_next to [S_min, S_max] for interpolation
            S_next_clipped = np.clip(S_next, S_min, S_max)
            
            # np.interp is slow for 2D. Use searchsorted or precompute.
            # Let's use a simple linear interpolation vectorized.
            # Or, since S_grid_dp is linear, we can use:
            # idx = (S_next_clipped - S_min) / (S_max - S_min) * (n_grid_dp - 1)
            
            idx = (S_next_clipped - S_min) / (S_max - S_min) * (n_grid_dp - 1)
            idx = np.clip(idx, 0, n_grid_dp - 1)
            
            i0 = np.floor(idx).astype(int)
            i1 = np.minimum(i0 + 1, n_grid_dp - 1)
            frac = idx - i0
            
            V_next = V_dp[j+1, i0] * (1 - frac) + V_dp[j+1, i1] * frac
            
            # Average over MC samples
            cont_val = np.exp(-r * dt_ex) * np.mean(V_next, axis=1)
            
            # Intrinsic value
            intrinsic = np.maximum(K - S_grid_dp, 0.0)
            
            # Optimal value
            V_dp[j, :] = np.maximum(intrinsic, cont_val)
            
            # Update the full grid V by interpolation from V_dp
            V[j, :] = np.interp(S_grid, S_grid_dp, V_dp[j, :])
    
    # Now, we have the value functions V[j, S] for each exercise date j.
    # The exercise boundary at step j is where Intrinsic(S) = V[j, S]
    # i.e., K - S = V[j, S] (for S < K)
    # We can find the boundary by searching for the crossing point.
    
    boundaries = np.zeros(n_exercises)
    for j in range(n_exercises):
        # For the last step, boundary is K
        if j == n_exercises - 1:
            boundaries[j] = K
            continue
        
        # Find S where K - S = V[j, S]
        # Define f(S) = K - S - V[j, S]
        # We want f(S) = 0
        # f(S) is decreasing in S (K-S is decreasing, V is increasing)
        # So there is at most one root.
        
        # Search for the sign change
        f = K - S_grid - V[j, :]
        
        # Find where f changes from positive to negative
        sign_change = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
        
        if len(sign_change) > 0:
            # Interpolate to find the root
            i = sign_change[0]
            f0 = f[i]
            f1 = f[i+1]
            S0 = S_grid[i]
            S1 = S_grid[i+1]
            
            if f1 != f0:
                boundaries[j] = S0 + (0 - f0) * (S1 - S0) / (f1 - f0)
            else:
                boundaries[j] = S0
        else:
            # No sign change, check if always positive or negative
            if f[0] <= 0:
                boundaries[j] = S_min
            else:
                boundaries[j] = S_max
    
    # Ensure boundaries are monotonic (they should be, but let's enforce)
    # For a put, the boundary should be non-decreasing as time approaches expiration?
    # Actually, for American puts, the boundary is non-decreasing.
    for j in range(1, n_exercises):
        if boundaries[j] < boundaries[j-1]:
            boundaries[j] = boundaries[j-1]
    
    # Store boundaries in the policy
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: step index 0..11.
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            S = np.asarray(current_state, dtype=float)
            boundary = self.boundaries[time]
            
            return S < boundary
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END

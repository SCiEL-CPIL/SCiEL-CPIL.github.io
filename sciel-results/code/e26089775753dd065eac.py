# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('volatility', contract.get('sigma', 0.2))
    T = contract.get('maturity', contract.get('T', 1.0))
    
    # 12 monthly exercise dates, no time-0 exercise
    n_exercise = 12
    dt = T / n_exercise  # monthly step
    
    # Set up Gauss-Hermite quadrature
    n_quad = 50  # number of quadrature points
    
    # For GBM: ln(S_{t+dt}/S_t) ~ N((r - sigma^2/2)*dt, sigma^2*dt)
    # Using GH: E[f(X)] where X ~ N(mu, var) = sum w_i f(mu + sqrt(2*var)*x_i)
    # GH quadrature: integral exp(-x^2) g(x) dx = sum w_i g(x_i)
    # For normal: E[f(X)] = (1/sqrt(2*pi*var)) * integral exp(-(x-mu)^2/(2*var)) f(x) dx
    # Let y = (x-mu)/sqrt(2*var), then E[f(X)] = (1/sqrt(pi)) * integral exp(-y^2) f(mu + sqrt(2*var)*y) dy
    # = sum (w_i/sqrt(pi)) * f(mu + sqrt(2*var)*y_i)
    
    x_gh, w_gh = roots_hermite(n_quad)
    # Normalize weights: sum w_i = sqrt(pi) for standard GH
    w_gh_norm = w_gh / np.sqrt(np.pi)
    
    mu = (r - 0.5 * sigma**2) * dt
    var = sigma**2 * dt
    sqrt_2var = np.sqrt(2.0 * var)
    
    # GH nodes and weights for the log-return
    gh_nodes = x_gh * sqrt_2var
    gh_weights = w_gh_norm
    
    # Grid of spot prices for continuation value computation
    # Range: from near 0 to well above K
    n_grid = 200
    S_min = 1.0
    S_max = K * 2.5
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # V[i] stores the value function at exercise date i (0-indexed: 0=first exercise, 11=last)
    # V[11] = max(K - S, 0)
    V = np.zeros((n_exercise, n_grid))
    
    # Last exercise date: only exercise, no continuation
    V[11] = np.maximum(K - S_grid, 0.0)
    
    # Backward induction
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 30:
            break
        # For each spot in the grid, compute continuation value
        # S' = S * exp(mu + sqrt_2var * y_i) for each GH node y_i
        # V_next(S') = interpolate V[i+1] at S'
        continuation = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = S_grid[j]
            # Next period spots
            S_next = S_j * np.exp(mu + gh_nodes)
            # Clip to grid range
            S_next_clipped = np.clip(S_next, S_min, S_max)
            # Interpolate V[i+1]
            V_next_vals = np.interp(S_next_clipped, S_grid, V[i+1])
            # Expected continuation value
            cont_val = np.sum(gh_weights * V_next_vals)
            # Discount
            continuation[j] = np.exp(-r * dt) * cont_val
        
        # Value = max(exercise, continuation)
        exercise_vals = np.maximum(K - S_grid, 0.0)
        V[i] = np.maximum(exercise_vals, continuation)
    
    # Create exercise policy object
    class ExercisePolicy:
        def __init__(self, V, S_grid, n_exercise, K, r, dt):
            self.V = V
            self.S_grid = S_grid
            self.n_exercise = n_exercise
            self.K = K
            self.r = r
            self.dt = dt
        
        def exercise(self, current_state, time_idx):
            """
            current_state: array of spot prices for all paths
            time_idx: integer index of current exercise date (0 to 11)
            Returns: boolean array, True if exercise
            """
            # Clamp time_idx
            t = int(np.clip(time_idx, 0, self.n_exercise - 1))
            
            # Exercise payoff
            exercise_payoff = np.maximum(self.K - current_state, 0.0)
            
            # Continuation value: interpolate V[t] and subtract exercise payoff? No.
            # V[t](S) = max(exercise, continuation)
            # We exercise if exercise >= continuation, i.e., V[t](S) == exercise_payoff
            # But we need to compare exercise vs continuation directly.
            # Actually, we can just check: exercise if K - S >= continuation_value
            # But we stored V = max(exercise, continuation), not continuation separately.
            
            # Better approach: store the continuation value separately, or
            # determine exercise by checking if V[t](S) equals the exercise payoff
            # (within tolerance), or better: store a boolean exercise mask.
            
            # Let me recompute: at each grid point, we know if we exercised or not
            # from the backward induction. Let's store exercise masks.
            
            # For now, use: exercise if K - S >= V_cont, where V_cont = V[t] - max(K-S,0) + min(K-S,0)
            # Actually simpler: exercise if (K - S) >= (V[t](S) - max(K-S, 0)) when K>S
            # Hmm, let's just store the continuation values.
            
            # Alternative: interpolate V[t] and exercise if K - S >= V_interp - (K-S) when K>S
            # This is getting complicated. Let me store exercise flags on the grid.
            
            # For robustness, let's just interpolate V[t] and compare with a fresh computation
            # Actually, the cleanest: we stored V[i] = max(exercise, continuation)
            # Exercise if exercise_payoff >= continuation
            # continuation = V[i] - max(exercise, 0) + min(exercise, 0) ... no
            # If exercise > 0: V = max(exercise, cont), so cont = V if V < exercise, else cont <= V
            # This is ambiguous when V == exercise.
            
            # Let me redo: store continuation values explicitly.
            # For now, use a simpler criterion: exercise if K - S > 0 and K - S >= V_interp * (1 - epsilon)
            # This isn't quite right either.
            
            # Best approach: recompute the continuation at evaluation time using the same GH quadrature
            # but that's expensive. Instead, let's store the continuation values.
            
            # For this version, I'll use: exercise if K - S >= 0 and the interpolated V is "close" to K-S
            # meaning continuation <= exercise. Since V = max(ex, cont), if ex >= cont then V = ex.
            # So exercise if V_interp <= K - S + small_tol (when K > S)
            # And don't exercise if K <= S (payoff is 0, never exercise)
            
            S_clipped = np.clip(current_state, self.S_grid[0], self.S_grid[-1])
            V_interp = np.interp(S_clipped, self.S_grid, self.V[t])
            
            # Exercise if K - S >= V_interp - tiny (meaning continuation was <= exercise)
            # When K > S: exercise if V_interp <= K - S + 1e-10
            # When K <= S: never exercise (payoff = 0)
            
            result = (self.K - current_state) >= V_interp - 1e-10
            result = result & (current_state < self.K)
            
            return result.astype(bool)
    
    # Wait, I need to store continuation values properly. Let me redo with explicit continuation storage.
    
    # Actually, let me restructure to store both V and continuation values.
    
    # Redo backward induction with explicit continuation storage
    V = np.zeros((n_exercise, n_grid))
    cont_vals = np.zeros((n_exercise, n_grid))  # continuation values at each date
    
    V[11] = np.maximum(K - S_grid, 0.0)
    cont_vals[11] = np.zeros(n_grid)  # No continuation at last date
    
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 20:
            # Fill remaining with just exercise (conservative)
            for ii in range(i, -1, -1):
                V[ii] = np.maximum(K - S_grid, 0.0)
                cont_vals[ii] = np.zeros(n_grid)
            break
        
        continuation = np.zeros(n_grid)
        for j in range(n_grid):
            S_j = S_grid[j]
            S_next = S_j * np.exp(mu + gh_nodes)
            S_next_clipped = np.clip(S_next, S_min, S_max)
            V_next_vals = np.interp(S_next_clipped, S_grid, V[i+1])
            cont_val = np.sum(gh_weights * V_next_vals)
            continuation[j] = np.exp(-r * dt) * cont_val
        
        exercise_vals = np.maximum(K - S_grid, 0.0)
        V[i] = np.maximum(exercise_vals, continuation)
        cont_vals[i] = continuation
    
    class ExercisePolicy:
        def __init__(self, V, cont_vals, S_grid, n_exercise, K):
            self.V = V
            self.cont_vals = cont_vals
            self.S_grid = S_grid
            self.n_exercise = n_exercise
            self.K = K
        
        def exercise(self, current_state, time_idx):
            t = int(np.clip(time_idx, 0, self.n_exercise - 1))
            
            S_clipped = np.clip(current_state, self.S_grid[0], self.S_grid[-1])
            cont_interp = np.interp(S_clipped, self.S_grid, self.cont_vals[t])
            
            # Exercise if immediate payoff >= continuation value
            exercise_payoff = np.maximum(self.K - current_state, 0.0)
            result = exercise_payoff >= cont_interp - 1e-10
            
            return result.astype(bool)
    
    return ExercisePolicy(V, cont_vals, S_grid, n_exercise, K)

# EVOLVE-BLOCK-END

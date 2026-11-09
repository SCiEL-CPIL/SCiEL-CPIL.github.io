import numpy as np
import time
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # ---- Parameters ----
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    training_paths = np.asarray(training_paths, dtype=float)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # ---- Build a dense log-space price grid ----
    # Use a high-resolution grid as suggested by successful strategies
    # Range should cover the majority of the training paths and the strike
    s_min = max(training_paths.min() * 0.5, 1e-4)
    s_max = max(training_paths.max() * 1.5, 2.0 * K)
    
    # 4096 points as per the "Knowledge" finding
    n_grid = 4096
    log_edges = np.linspace(np.log(s_min), np.log(s_max), n_grid + 1)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    grid = np.exp(log_centers)  # (n_grid,)

    # ---- Gauss-Hermite quadrature for E[ f(S_{t+dt}) | S_t = s ] ----
    # Risk-neutral GBM: log(S_{t+dt}/s) ~ N(mu, s2), mu = (r - 0.5*sigma^2)*dt
    mu = (r - 0.5 * sigma * sigma) * dt
    s2 = sigma * sigma * dt
    
    # 64 points as per the "Knowledge" finding
    n_q = 64
    x, w = roots_hermite(n_q)
    # Transform standard normal z -> log-return u = mu + sqrt(s2)*z ; z = sqrt(2)*x
    u = mu + np.sqrt(2.0 * s2) * x
    # Weights for E[f(u)] over log-return
    # E[f(X)] = 1/sqrt(pi) * sum w_i f(x_i) for X~N(0,1) in Hermite basis?
    # roots_hermite gives x, w such that sum(w_i * f(x_i)) approximates integral of f(x) * exp(-x^2)
    # For E[f(Z)] where Z~N(0,1): E[f(Z)] = 1/sqrt(pi) * integral f(x/sqrt(2)) * exp(-x^2/2) dx ?
    # Standard GH quadrature approximates integral_{-inf}^{inf} f(x) e^{-x^2} dx.
    # If we want E[f(Y)] where Y ~ N(0, sigma_y^2), we use transformation.
    # Here Y = u = mu + sqrt(s2)*Z.
    # E[f(u)] = E[f(mu + sqrt(s2)*Z)] = 1/sqrt(2*pi) * integral f(mu + sqrt(s2)*z) e^{-z^2/2} dz.
    # Let z = sqrt(2)*x. dz = sqrt(2) dx.
    # Integral = 1/sqrt(2*pi) * integral f(mu + sqrt(2*s2)*x) e^{-x^2} sqrt(2) dx
    # = 1/sqrt(pi) * integral f(mu + sqrt(2*s2)*x) e^{-x^2} dx.
    # roots_hermite approximates integral_{-inf}^{inf} g(x) e^{-x^2} dx by sum w_i g(x_i).
    # So E[f(u)] approx 1/sqrt(pi) * sum w_i f(mu + sqrt(2*s2)*x_i).
    qw = w / np.sqrt(np.pi)

    # ---- Backward induction on the grid ----
    V = np.zeros((n_exercise + 1, n_grid))
    # At maturity (index n_exercise), value is intrinsic
    V[n_exercise] = np.maximum(K - grid, 0.0)

    # Precompute exp(u) for speed
    exp_u = np.exp(u)
    qw_col = qw.reshape(1, -1)

    def continuation_on_grid(V_next):
        # S_next = grid[:, None] * exp_u[None, :]
        # Shape: (n_grid, n_q)
        S_next = np.outer(grid, exp_u)
        log_S_next = np.log(S_next)
        
        # Interpolate V_next onto S_next (log-space)
        # log_centers is sorted
        idx = np.searchsorted(log_centers, log_S_next, side='right') - 1
        idx = np.clip(idx, 0, n_grid - 2)
        
        c0 = V_next[idx]
        c1 = V_next[idx + 1]
        lc0 = log_centers[idx]
        lc1 = log_centers[idx + 1]
        
        # Linear interpolation in log-space
        # t = (log_S_next - lc0) / (lc1 - lc0)
        t = (log_S_next - lc0) / (lc1 - lc0)
        t = np.clip(t, 0.0, 1.0)
        
        interp = c0 + t * (c1 - c0)
        
        # Weighted sum over quadrature
        # Sum over axis 1 (quadrature points)
        return np.sum(interp * qw_col, axis=1)

    # Iterate backwards from n_exercise-1 to 1
    # Note: Exercise dates are 1..12. Index 0 is t=0 (no exercise).
    # V[k] is value at date k.
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline:
            # Emergency break, though unlikely with n_grid=4096 and n_q=64
            # 4096 * 64 operations * 12 steps is ~3M ops, very fast.
            break

        # Continuation value: discount * E[V[k+1](S_{t+dt}) | S_t]
        cont_grid = discount * continuation_on_grid(V[k + 1])
        
        # Exercise value
        ex_grid = np.maximum(K - grid, 0.0)
        
        # Optimal value is max(Exercise, Continuation)
        V[k] = np.maximum(ex_grid, cont_grid)

    # Create the policy object
    class BermudanPutPolicy:
        def __init__(self, grid, log_centers, V, n_grid, K, n_exercise):
            self.grid = grid
            self.log_centers = log_centers
            self.V = V
            self.n_grid = n_grid
            self.K = K
            self.n_exercise = n_exercise

        def exercise(self, current_state, time_step):
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(np.round(time_step * 12))

                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)

                # Maturity: exercise if in the money
                if idx == self.n_exercise:
                    return current_state < self.K

                # Interpolate value function V[idx] at current_state
                log_S = np.log(current_state)
                idx_grid = np.searchsorted(self.log_centers, log_S, side='right') - 1
                idx_grid = np.clip(idx_grid, 0, self.n_grid - 2)
                
                c0 = self.V[idx][idx_grid]
                c1 = self.V[idx][idx_grid + 1]
                lc0 = self.log_centers[idx_grid]
                lc1 = self.log_centers[idx_grid + 1]
                
                t = (log_S - lc0) / (lc1 - lc0)
                t = np.clip(t, 0.0, 1.0)
                
                val = c0 + t * (c1 - c0)
                
                payoff = np.maximum(self.K - current_state, 0.0)
                
                # Exercise if payoff >= value (which includes payoff, so this is payoff >= continuation effectively)
                return payoff >= val - 1e-10 # small epsilon for stability

            except Exception:
                return np.zeros(len(current_state), dtype=bool)

    return BermudanPutPolicy(grid, log_centers, V, n_grid, K, n_exercise)

"""Bermudan exercise-policy research with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P

class BermudanExercisePolicy:
    """
    Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression.
    
    At each exercise date, we fit a polynomial regression to the continuation value
    (discounted expected value of optimal stopping in the future) using training paths,
    then compare immediate exercise payoff against this fitted continuation value.
    """
    
    def __init__(self, training_paths, contract):
        """
        Fit the exercise policy using training paths.
        
        Parameters:
        -----------
        training_paths : array of shape (n_paths, n_timesteps+1)
            Simulated price paths. Each row is a path.
            training_paths[i, j] is the price at time j for path i.
            Time 0 is the initial time, exercise dates are at indices 1..12.
        contract : dict or object with contract parameters
            Should contain: K, r, sigma, T, and possibly n_exercise_dates, exercise_times
        """
        self.t_start = time.time()
        self.deadline = 150.0  # seconds, with safety margin
        
        # Extract contract parameters
        K = self._get_param(contract, 'K', 100.0)
        r = self._get_param(contract, 'r', 0.05)
        sigma = self._get_param(contract, 'sigma', 0.2)
        T = self._get_param(contract, 'T', 1.0)
        
        # Exercise dates: 12 monthly, no time-0
        # If contract specifies exercise times, use those; otherwise assume monthly
        exercise_times = self._get_exercise_times(contract, T)
        n_ex = len(exercise_times)
        
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.exercise_times = exercise_times
        self.n_ex = n_ex
        
        # training_paths shape: (n_paths, n_time_points)
        # We need to figure out which columns correspond to exercise dates
        # Assume training_paths has columns for each time step, with exercise dates
        # at specific indices. Typically, if there are 12 exercise dates over T=1,
        # and paths have 13 time points (0 to 12), then exercise dates are at indices 1..12.
        
        n_paths = training_paths.shape[0]
        n_time_points = training_paths.shape[1]
        
        # Determine exercise date indices in the path array
        # If there are 12 exercise dates and n_time_points = 13, exercise at indices 1..12
        # More generally, we need to map exercise_times to indices
        self._determine_exercise_indices(n_time_points, T)
        
        # Fit regression polynomials at each exercise date using backward induction
        # Store coefficients for each exercise date
        self.reg_coeffs = []  # list of polynomial coefficients, one per exercise date
        self.reg_degree = 4  # polynomial degree
        
        # Backward induction: start from last exercise date
        # At the last exercise date, continuation value = 0 (no more exercise opportunities)
        # Actually, at the last exercise date, the option expires, so continuation = 0
        # The value at the last exercise date is just the intrinsic value
        
        # We'll work backwards from the last exercise date to the first
        # For each exercise date j (from last to first):
        #   1. Compute the continuation value for paths that reach this date
        #      by discounting the values from the next exercise date
        #   2. Fit a polynomial to the continuation value as a function of S
        #   3. Store the coefficients
        
        # Initialize: at the last exercise date, the "continuation" from beyond is 0
        # But we need the value at each exercise date for the previous step
        
        # Let's define:
        # V_j(S) = max(intrinsic(S), E[discount * V_{j+1}(S_{j+1}) | S_j = S])
        # For the last exercise date (j = n_ex-1 in 0-indexed, or j = 11 in 1-indexed):
        # V_last(S) = intrinsic(S) since there's no future exercise
        
        # We'll store the fitted continuation value function for each exercise date
        # cont_values[j] is the polynomial that gives E[discount * V_{j+1} | S_j]
        
        # Start from the end
        # At exercise date n_ex (the last one), the value is intrinsic
        # For the second-to-last, we need E[discount * intrinsic(S_next) | S_current]
        
        # Let's work with indices 0 to n_ex-1 for exercise dates
        # exercise_indices[i] is the column index in training_paths for exercise date i
        
        # Backward induction
        # V_next = intrinsic value at the next exercise date (as a function of S)
        # We'll represent V_next as a polynomial fit
        
        # At the last exercise date, V = intrinsic(S) = max(K - S, 0)
        # This is not a polynomial, but we can fit it
        
        # Actually, let's think more carefully:
        # 
        # Let exercise dates be at times t_1, t_2, ..., t_n (1-indexed)
        # V_n(S) = max(K - S, 0)  (last exercise date, option expires)
        # V_j(S) = max(K - S, E[exp(-r*(t_{j+1}-t_j)) * V_{j+1}(S_{j+1}) | S_j = S])
        #
        # We want to fit a polynomial to the continuation value:
        #   C_j(S) ≈ E[exp(-r*(t_{j+1}-t_j)) * V_{j+1}(S_{j+1}) | S_j = S]
        #
        # Then V_j(S) = max(K - S, C_j(S))
        #
        # For the regression at step j, we use the training paths:
        #   For each path i, at exercise date j, we have S_j = training_paths[i, exercise_indices[j]]
        #   The realized continuation value is exp(-r*dt) * V_{j+1}(S_{j+1})
        #   where V_{j+1} is the fitted value from the next step
        #
        # So we fit a polynomial to (S_j, realized_continuation_value)
        
        # Start from the last exercise date
        # V_last(S) = max(K - S, 0)
        # We need to represent this. Since it's piecewise linear, we can fit a polynomial
        # or just use the intrinsic value directly
        
        # For simplicity, let's fit polynomials at each step
        
        # Initialize: after the last exercise date, there's no value
        # So at the last exercise date, continuation = 0, and V = intrinsic
        
        # Let's store the fitted value function for each exercise date
        # fitted_values[j] = polynomial coefficients for V_j(S)
        
        # Work backwards
        # At the last exercise date (index n_ex-1):
        #   V = intrinsic(S) = max(K - S, 0)
        #   We can fit a polynomial to this, or just use the formula
        
        # For the purpose of regression at the previous step, we need V_last(S_next)
        # for each path
        
        # Let's compute the realized values at each exercise date for training paths
        
        # Extract prices at exercise dates
        # exercise_prices[j] is an array of shape (n_paths,) with S at exercise date j for each path
        
        exercise_prices = []
        for j in range(n_ex):
            idx = self.exercise_indices[j]
            if idx < n_time_points:
                exercise_prices.append(training_paths[:, idx])
            else:
                # Should not happen
                exercise_prices.append(np.zeros(n_paths))
        
        # Backward induction
        # V_next_coeffs: polynomial coefficients for V_{j+1}(S)
        # Start with V at the last exercise date
        
        # At the last exercise date, V_last(S) = max(K - S, 0)
        # Let's fit a polynomial to this using the training paths at the last exercise date
        S_last = exercise_prices[-1]
        V_last_realized = np.maximum(self.K - S_last, 0.0)
        
        # Fit polynomial
        self.reg_coeffs = [None] * n_ex
        self.reg_coeffs[-1] = self._fit_polynomial(S_last, V_last_realized, self.reg_degree)
        
        # Now work backwards from second-to-last to first
        for j in range(n_ex - 2, -1, -1):
            # Check time
            if time.time() - self.t_start > self.deadline - 10:
                # Not enough time, use intrinsic value as approximation
                for k in range(j + 1, n_ex):
                    if self.reg_coeffs[k] is None:
                        S_k = exercise_prices[k]
                        V_k = np.maximum(self.K - S_k, 0.0)
                        self.reg_coeffs[k] = self._fit_polynomial(S_k, V_k, self.reg_degree)
                break
            
            S_j = exercise_prices[j]
            S_next = exercise_prices[j + 1]
            
            # Compute discount factor
            dt = self.exercise_times[j + 1] - self.exercise_times[j]
            discount = np.exp(-self.r * dt)
            
            # Realized continuation value: discount * V_{j+1}(S_next)
            # V_{j+1} is given by the polynomial we fitted
            V_next_realized = self._eval_polynomial(self.reg_coeffs[j + 1], S_next)
            cont_realized = discount * V_next_realized
            
            # Fit polynomial to (S_j, cont_realized)
            self.reg_coeffs[j] = self._fit_polynomial(S_j, cont_realized, self.reg_degree)
        
        # If we didn't finish, fill in remaining with intrinsic value fits
        for j in range(n_ex):
            if self.reg_coeffs[j] is None:
                S_j = exercise_prices[j]
                V_j = np.maximum(self.K - S_j, 0.0)
                self.reg_coeffs[j] = self._fit_polynomial(S_j, V_j, self.reg_degree)
    
    def _get_param(self, contract, key, default):
        """Extract parameter from contract (dict or object)."""
        if contract is None:
            return default
        if isinstance(contract, dict):
            return contract.get(key, default)
        return getattr(contract, key, default)
    
    def _get_exercise_times(self, contract, T):
        """Get exercise times from contract or default to monthly."""
        if contract is not None:
            if isinstance(contract, dict):
                times = contract.get('exercise_times', None)
                if times is not None:
                    return np.asarray(times, dtype=float)
                n_ex = contract.get('n_exercise_dates', 12)
            else:
                times = getattr(contract, 'exercise_times', None)
                if times is not None:
                    return np.asarray(times, dtype=float)
                n_ex = getattr(contract, 'n_exercise_dates', 12)
        else:
            n_ex = 12
        
        # Default: 12 monthly exercise dates over T
        return np.linspace(T / n_ex, T, n_ex)
    
    def _determine_exercise_indices(self, n_time_points, T):
        """Determine which columns in the path array correspond to exercise dates."""
        n_ex = self.n_ex
        if n_time_points == n_ex + 1:
            # Standard case: time 0 + n_ex exercise dates
            self.exercise_indices = list(range(1, n_ex + 1))
        elif n_time_points == n_ex:
            # Only exercise dates, no time 0
            self.exercise_indices = list(range(n_ex))
        else:
            # Try to map exercise times to indices
            # Assume uniform time grid
            dt = T / (n_time_points - 1)
            self.exercise_indices = []
            for t in self.exercise_times:
                idx = round(t / dt)
                idx = min(max(idx, 0), n_time_points - 1)
                self.exercise_indices.append(idx)
    
    def _fit_polynomial(self, S, values, degree):
        """Fit a polynomial of given degree to (S, values)."""
        try:
            # Use numpy polyfit
            coeffs = np.polyfit(S, values, degree)
            return coeffs
        except Exception:
            # Fallback: constant
            return np.array([np.mean(values)])
    
    def _eval_polynomial(self, coeffs, S):
        """Evaluate polynomial at S."""
        try:
            return np.polyval(coeffs, S)
        except Exception:
            return np.zeros_like(S)
    
    def exercise(self, current_state, time):
        """
        Determine exercise decision for given current state and time.
        
        Parameters:
        -----------
        current_state : array of shape (n_states,)
            Current price states.
        time : float
            Current time.
        
        Returns:
        --------
        exercise_decisions : boolean array of shape (n_states,)
            True if exercise, False if continue.
        """
        n_states = len(current_state)
        decisions = np.zeros(n_states, dtype=bool)
        
        # Find which exercise date we're at
        # time should be close to one of the exercise times
        j = self._find_exercise_date(time)
        
        if j is None:
            # Not at an exercise date, don't exercise
            return decisions
        
        # Get the fitted continuation value polynomial for this exercise date
        if j >= len(self.reg_coeffs) or self.reg_coeffs[j] is None:
            # No polynomial, use intrinsic value
            intrinsic = np.maximum(self.K - current_state, 0.0)
            decisions = intrinsic > 0
            return decisions
        
        # Evaluate continuation value
        cont_value = self._eval_polynomial(self.reg_coeffs[j], current_state)
        
        # Intrinsic value
        intrinsic = np.maximum(self.K - current_state, 0.0)
        
        # Exercise if intrinsic > continuation
        decisions = intrinsic > cont_value
        
        return decisions
    
    def _find_exercise_date(self, time):
        """Find which exercise date index corresponds to the given time."""
        for j, t in enumerate(self.exercise_times):
            if abs(time - t) < 1e-6:
                return j
        # If not found, find the closest
        if len(self.exercise_times) == 0:
            return None
        diffs = np.abs(self.exercise_times - time)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 1e-3:  # Within a small tolerance
            return min_idx
        return None


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using training paths.
    
    Parameters:
    -----------
    training_paths : array of shape (n_paths, n_time_points)
        Simulated price paths.
    contract : dict or object
        Contract parameters.
    
    Returns:
    --------
    policy : BermudanExercisePolicy
        The fitted exercise policy.
    """
    return BermudanExercisePolicy(training_paths, contract)
# EVOLVE-BLOCK-END

"""Bermudan put exercise policy via backward induction on a binomial tree."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    # 12 monthly exercise dates, no time-0 exercise
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Build a fine binomial tree for accurate backward induction
    # Use 1200 steps to match the reference
    n_steps = 1200
    dt_tree = T / n_steps

    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1.0 / u
    p_up = (np.exp(r * dt_tree) - d) / (u - d)
    p_down = 1.0 - p_up

    # Discount factor per tree step
    disc = np.exp(-r * dt_tree)

    # For each exercise date, we need to know the optimal exercise threshold
    # We'll do backward induction on the tree and record the exercise boundary
    # at each exercise date

    # Number of tree steps between exercise dates
    steps_per_exercise = n_steps // n_exercise  # 100 steps per month

    # We'll store the value function at each exercise date as a function of the
    # log-price index. For the tree, at step k, the price is S0 * u^(j) * d^(k-j)
    # where j is the number of up moves.

    # Backward induction: start from the last exercise date
    # At the last exercise date (t = T), the option is a European put
    # V_T(S) = max(K - S, 0)

    # We need to track the value at each exercise date as a function of the
    # number of up moves (or equivalently, the log-price level)

    # Let's use the log-price grid. At step k, the possible log-prices are:
    # log(S0) + (2*j - k)*log(u) for j = 0, 1, ..., k
    # where j is the number of up moves.

    # For efficiency, we'll work with the index j (number of up moves) and
    # compute the actual price when needed.

    # Initialize value at the last exercise date (step n_steps)
    # At step n_steps, j ranges from 0 to n_steps
    # Price = S0 * u^j * d^(n_steps - j) = S0 * u^(2*j - n_steps)

    # Value at maturity: max(K - S, 0)
    j_max = n_steps
    j_vals = np.arange(0, j_max + 1)
    log_prices = np.log(S0) + (2 * j_vals - j_max) * np.log(u)
    prices = np.exp(log_prices)
    values = np.maximum(K - prices, 0.0)

    # Store the exercise boundary at the last exercise date
    # We'll store the threshold price below which we exercise
    # Actually, for a put, we exercise when S <= threshold
    # Let's store the value function and the exercise decision

    # For each exercise date going backward, we need to:
    # 1. Discount the continuation value forward by steps_per_exercise steps
    # 2. Compare with immediate payoff
    # 3. Record the exercise boundary

    # We'll store, for each exercise date, the value as a function of j
    # and the exercise decision (True if exercise)

    n_exercise_dates = n_exercise
    # values_at_exercise[i] = value function at exercise date i (0-indexed, 0 = first exercise)
    # We'll store arrays indexed by j for the tree at that time step

    # At exercise date i (1-indexed), the tree step is i * steps_per_exercise
    # j ranges from 0 to i * steps_per_exercise

    # Let's do backward induction from the last exercise date to the first
    # We'll store the value function at each exercise date

    # For memory efficiency, we only need the current and next value functions
    # But we need to store the exercise boundaries for the policy

    # Exercise boundaries: for each exercise date, store the threshold j
    # below which we exercise. Since the tree is recombining, at step k,
    # j goes from 0 to k. The exercise region for a put is j <= j_threshold
    # (lower j means lower price, which is better for a put)

    # Actually, let's just store the full value function and exercise decision
    # at each exercise date, then interpolate for arbitrary states.

    # For the policy, given a state S and time t, we:
    # 1. Determine which exercise date we're at
    # 2. Map S to the nearest tree node
    # 3. Apply the exercise decision from backward induction

    # Let's store, for each exercise date, the exercise threshold in terms of
    # the log-price. For a put, there's a single threshold: exercise if S <= S_threshold

    # Actually, the optimal exercise boundary for a Bermudan put on a binomial
    # tree is monotone, so there's a single threshold at each exercise date.

    # Let's compute the threshold at each exercise date.

    # Backward induction
    # Start at the last exercise date (t = T)
    # Value = max(K - S, 0), exercise if K - S > 0, i.e., S < K
    # So threshold at last date is K (exercise if S < K)

    # For earlier dates, we need to compute the continuation value

    # Let's do the full backward induction on the tree

    # values[j] at the current step, where j is the number of up moves
    # At step k, j ranges from 0 to k

    # Start at step n_steps (maturity)
    current_step = n_steps
    j_vals = np.arange(0, current_step + 1)
    log_prices = np.log(S0) + (2 * j_vals - current_step) * np.log(u)
    prices = np.exp(log_prices)
    values = np.maximum(K - prices, 0.0)

    # Store the value function at the last exercise date
    # Exercise date index: n_exercise - 1 (0-indexed)
    # We'll store thresholds for each exercise date

    thresholds = [None] * n_exercise_dates
    # At the last exercise date, exercise if S < K
    thresholds[-1] = K

    # Now go backward through each exercise date
    for i in range(n_exercise_dates - 1, 0, -1):
        # i is the exercise date index (0-indexed), going from n_exercise-1 down to 1
        # We're currently at step (i+1) * steps_per_exercise
        # We need to go back to step i * steps_per_exercise

        target_step = i * steps_per_exercise
        steps_back = current_step - target_step

        # We need to compute the continuation value at target_step
        # by discounting and averaging over the next steps_back steps

        # For efficiency, we can do this iteratively, but steps_back = 100
        # which is manageable

        # Actually, let's just do the full backward induction step by step
        # from current_step down to target_step

        for step in range(current_step, target_step, -1):
            # At step, j ranges from 0 to step
            # j_vals for this step
            j_vals_step = np.arange(0, step + 1)
            # j_vals for step-1 ranges from 0 to step-1
            j_vals_prev = np.arange(0, step)

            # The value at step-1, j is the discounted expected value of
            # the values at step, j (down move) and step, j+1 (up move)
            # V(step-1, j) = disc * (p_up * V(step, j+1) + p_down * V(step, j))

            # values currently corresponds to step
            # We need to compute values_prev for step-1

            # V(step, j) for j = 0 to step
            # V(step-1, j) = disc * (p_up * V(step, j+1) + p_down * V(step, j))
            # for j = 0 to step-1

            values_prev = disc * (p_up * values[1:] + p_down * values[:-1])

            # Check if this step is an exercise date
            if step == target_step:
                # This is an exercise date, so we can exercise
                # Compute the price at this step
                log_prices_step = np.log(S0) + (2 * j_vals_prev - (step - 1)) * np.log(u)
                prices_step = np.exp(log_prices_step)
                payoff = np.maximum(K - prices_step, 0.0)
                values_prev = np.maximum(payoff, values_prev)

                # Determine the exercise threshold
                # For a put, exercise if S <= threshold
                # Find the largest j such that we exercise
                # Exercise if payoff >= values_prev, i.e., K - S >= values_prev
                # S <= K - values_prev
                # But values_prev depends on j, so we need to find the boundary

                # The exercise region is where payoff > continuation value
                # For a put, this is typically for low S (low j)
                # Find the threshold j where exercise switches to hold

                exercise_mask = payoff >= values_prev
                # Find the boundary: the largest j where we exercise
                # Since the boundary should be monotone, find the transition point
                if np.any(exercise_mask):
                    # Find the last True in exercise_mask
                    # The threshold is the price at the last exercising node
                    # Actually, we want the threshold price such that we exercise if S <= threshold
                    # Find the j where exercise_mask transitions from True to False
                    # Since j increases with price, exercise_mask should be True for low j

                    # Find the boundary
                    # The threshold is the price at the highest j where we still exercise
                    # If all exercise, threshold is the highest price
                    # If none exercise, threshold is 0

                    if exercise_mask.all():
                        threshold_price = prices_step[-1]
                    elif not exercise_mask.any():
                        threshold_price = 0.0
                    else:
                        # Find the last True
                        true_indices = np.where(exercise_mask)[0]
                        last_true = true_indices[-1]
                        # The threshold is between prices_step[last_true] and prices_step[last_true+1]
                        # Use the price at last_true as a conservative threshold
                        # Actually, we want the exact boundary. Let's use interpolation.
                        # For simplicity, use the price at last_true
                        threshold_price = prices_step[last_true]

                    thresholds[i] = threshold_price

            values = values_prev
            current_step = step - 1

        # After the inner loop, we're at target_step
        # values corresponds to target_step
        current_step = target_step

    # Now we have thresholds for all exercise dates
    # thresholds[i] is the price threshold at exercise date i (0-indexed)
    # Exercise if S <= thresholds[i]

    # For the policy, given a state S and time t:
    # 1. Determine which exercise date we're at
    # 2. Check if S <= thresholds[i]

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, S0, K, r, sigma, T):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.S0 = S0
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = len(exercise_times)
            self.dt = T / self.n_exercise

        def exercise(self, current_state, t):
            # current_state is a vector of spot prices
            # t is the current time
            # Return a boolean vector: True if exercise

            # Determine which exercise date we're at
            # Exercise dates are at t = dt, 2*dt, ..., n*dt
            # Find the exercise date index

            # t should be close to one of the exercise dates
            # Find the nearest exercise date
            idx = np.argmin(np.abs(self.exercise_times - t))

            # Get the threshold for this exercise date
            threshold = self.thresholds[idx]

            # Exercise if S <= threshold
            return current_state <= threshold

    policy = ExercisePolicy(thresholds, exercise_times, S0, K, r, sigma, T)
    return policy

# EVOLVE-BLOCK-END

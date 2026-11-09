# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise

    # Binomial tree parameters (CRR)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Build tree: at step i (0-indexed, i=0..n_exercise-1), there are i+1 nodes
    # Node (i, j) has price S0 * u^j * d^(i-j), j=0..i
    # We need values at each exercise date. Exercise dates are t_1, t_2, ..., t_12
    # So step i corresponds to time (i+1)*dt, and there are i+1 nodes at step i (j=0..i)

    # We'll store for each step i and node j: whether to exercise
    # exercise_decision[i][j] = True if we should exercise at node (i, j)

    # Work backward from the last exercise date
    # At step n_exercise-1 (last exercise), value = max(payoff, 0) since no continuation
    # Actually, at the last exercise date, continuation value is 0 (option expires)
    # So value = max(K - S, 0)

    # Let's build the tree prices
    # tree_price[i][j] for i in range(n_exercise), j in range(i+1)
    tree_price = np.zeros((n_exercise, n_exercise + 1))
    for i in range(n_exercise):
        for j in range(i + 1):
            tree_price[i, j] = S0 * (u ** j) * (d ** (i - j))

    # Value function: value[i][j] = optimal value at node (i, j)
    value = np.zeros((n_exercise, n_exercise + 1))
    exercise_decision = np.zeros((n_exercise, n_exercise + 1), dtype=bool)

    # Terminal step (last exercise date, i = n_exercise - 1)
    i = n_exercise - 1
    for j in range(i + 1):
        payoff = max(K - tree_price[i, j], 0.0)
        value[i, j] = payoff
        exercise_decision[i, j] = (payoff > 0)  # always exercise if in-the-money at last date

    # Backward induction
    for i in range(n_exercise - 2, -1, -1):
        for j in range(i + 1):
            # Continuation value: discounted expected value of next step
            # From node (i, j), next step goes to (i+1, j) with prob p (down in j? no)
            # Actually, from (i, j): up goes to (i+1, j+1), down goes to (i+1, j)
            cont_value = disc * (p * value[i + 1, j + 1] + (1 - p) * value[i + 1, j])
            payoff = max(K - tree_price[i, j], 0.0)
            if payoff > cont_value:
                value[i, j] = payoff
                exercise_decision[i, j] = True
            else:
                value[i, j] = cont_value
                exercise_decision[i, j] = False

    # Now create the exercise policy object
    class ExercisePolicy:
        def __init__(self, tree_price, exercise_decision, u, d, S0, n_exercise, K):
            self.tree_price = tree_price
            self.exercise_decision = exercise_decision
            self.u = u
            self.d = d
            self.S0 = S0
            self.n_exercise = n_exercise
            self.K = K

        def exercise(self, current_state, time):
            # current_state: array of stock prices
            # time: current time (continuous)
            # Determine which exercise step we're at
            dt = self.n_exercise and (1.0 / self.n_exercise) or 0.0
            # Exercise dates are at dt, 2*dt, ..., n_exercise*dt
            # time is given as continuous; we need to map to step index
            # The step index i (0-based) corresponds to time (i+1)*dt
            # If time is at or near an exercise date, use that step
            # current_state is an array, time is a scalar
            
            # Determine step: which exercise date are we closest to?
            # Since exercise is only at discrete dates, time should be at one of them
            # step = round(time / dt) - 1, clamped to [0, n_exercise-1]
            dt_val = 1.0 / self.n_exercise
            step = int(round(time / dt_val)) - 1
            step = max(0, min(step, self.n_exercise - 1))
            
            # For each stock price, find the nearest tree node at this step
            # At step i, node j has price S0 * u^j * d^(i-j)
            # We can solve for j: log(S/S0) = j*log(u) + (i-j)*log(d)
            # log(S/S0) = j*(log(u) - log(d)) + i*log(d)
            # j = (log(S/S0) - i*log(d)) / (log(u) - log(d))
            
            i = step
            log_u = np.log(self.u)
            log_d = np.log(self.d)
            log_ratio = log_u - log_d
            
            # For each price in current_state, compute j
            S = np.asarray(current_state, dtype=float)
            log_S = np.log(np.maximum(S, 1e-10))
            log_S0 = np.log(self.S0)
            
            # j = (log(S/S0) - i*log_d) / log_ratio
            j_float = (log_S - log_S0 - i * log_d) / log_ratio
            j_int = np.round(j_float).astype(int)
            
            # Clamp j to valid range [0, i]
            j_int = np.clip(j_int, 0, i)
            
            # Look up exercise decisions
            # exercise_decision[i, j] for each element
            # We need to index: for each element k, use exercise_decision[i, j_int[k]]
            # Since j_int varies per element, we need advanced indexing
            # Create indices: row index = i for all, col index = j_int
            rows = np.full(len(S), i)
            cols = j_int
            result = self.exercise_decision[rows, cols]
            
            return result

    return ExercisePolicy(tree_price, exercise_decision, u, d, S0, n_exercise, K)
# EVOLVE-BLOCK-END

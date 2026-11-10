"""Bermudan exercise-policy research: neural network continuation value regression."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using neural network regression for
    continuation values. Returns an object with exercise(current_state, time) -> boolean vector.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before the 150s kill

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = 12  # 12 monthly exercise dates
    dt = T / num_steps

    # Handle training_paths shape
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)

    N, total_cols = training_paths.shape

    # Determine which columns are exercise dates
    if total_cols == num_steps + 1:
        exercise_paths = training_paths[:, 1:]  # shape (N, 12)
    elif total_cols == num_steps:
        exercise_paths = training_paths[:, :]  # shape (N, 12)
    else:
        exercise_paths = training_paths[:, -num_steps:]

    # Intrinsic value at each exercise date
    intrinsic = np.maximum(K - exercise_paths, 0.0)  # shape (N, 12)

    # Time check for early return
    if time.time() - start_time > deadline - 10:
        class SimplePolicy:
            def exercise(self, current_state, time_idx):
                if time_idx >= num_steps - 1:
                    return current_state < K
                return np.zeros(len(current_state), dtype=bool)
        return SimplePolicy()

    # Normalize spot prices for neural network input
    # Use log-normal scaling: x = log(S / S0) where S0 is initial spot
    # But we don't have S0 directly, so use a fixed reference
    S_ref = 100.0  # Reference spot price
    log_spots = np.log(exercise_paths / S_ref)  # shape (N, 12)

    # Neural network architecture:
    # Input: [log(S), time_index] -> hidden layers -> output: continuation value
    # We'll train separate small networks for each time step, or a single network
    # with time as input.

    # Let's use a single network with inputs [log(S), t] where t is the time index (0-11)
    # Output: continuation value (hold value)

    # First, we need to compute target continuation values via backward induction
    # on the training paths using a simple binning approach to get initial targets.

    # Create a grid of spot prices for the backward induction
    all_spots = exercise_paths.flatten()
    min_s = max(0.1, np.percentile(all_spots, 0.5))
    max_s = min(3 * K, np.percentile(all_spots, 99.5))

    n_grid = 500
    grid = np.linspace(min_s, max_s, n_grid)

    # Backward induction to compute value function and exercise thresholds
    # V_i(s) = max(K - s, disc * E[V_{i+1}(S_{i+1}) | S_i = s])

    # Initialize at the last exercise date
    V = np.maximum(K - grid, 0.0)  # shape (n_grid,)

    # Store continuation values for training the neural network
    # For each time step i, we want to learn C_i(S) = disc * E[V_{i+1}(S_{i+1}) | S_i = S]
    # The exercise decision is: exercise if K - S >= C_i(S)

    # We'll compute C_i for each time step using the training paths
    continuation_targets = []  # List of (S_i, C_i) pairs for each time step

    for i in range(num_steps - 1, -1, -1):
        if time.time() - start_time > deadline - 5:
            break

        s_i = exercise_paths[:, i]  # shape (N,)

        if i + 1 < num_steps:
            s_next = exercise_paths[:, i + 1]  # shape (N,)
            # Compute V_{i+1}(S_{i+1}) for each path
            v_next = np.interp(s_next, grid, V)  # shape (N,)

            # Bin the S_i values
            bin_edges = np.linspace(min_s, max_s, n_grid + 1)
            bin_indices = np.digitize(s_i, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_grid - 1)

            # For each bin, compute the average V_{i+1}(S_{i+1})
            continuation_bin = np.zeros(n_grid)
            counts = np.zeros(n_grid)
            for b in range(n_grid):
                mask = bin_indices == b
                if np.any(mask):
                    continuation_bin[b] = np.mean(v_next[mask])
                    counts[b] = np.sum(mask)

            disc = np.exp(-r * dt)
            continuation_value = disc * continuation_bin

            # Interpolate to fill gaps
            valid = counts > 0
            if np.any(~valid) and np.any(valid):
                continuation_value[~valid] = np.interp(
                    grid[~valid], grid[valid], continuation_value[valid]
                )
            elif not np.any(valid):
                continuation_value = np.zeros(n_grid)
        else:
            continuation_value = np.zeros(n_grid)

        # Store targets for neural network training
        # For each path, the continuation value at time i is C_i(S_i)
        c_i_path = np.interp(s_i, grid, continuation_value)
        continuation_targets.append((s_i, c_i_path))

        # Update value function
        intrinsic_grid = np.maximum(K - grid, 0.0)
        V = np.maximum(intrinsic_grid, continuation_value)

    # Now train a neural network to learn the continuation value function
    # Input: [log(S), time_index] -> Output: continuation value
    # We'll use a simple 2-layer neural network with tanh activation

    # Prepare training data
    # For each time step i, we have (S_i, C_i) pairs
    # Combine all time steps into one dataset
    all_S = []
    all_C = []
    all_t = []

    for i in range(num_steps):
        if i < len(continuation_targets):
            s_i, c_i = continuation_targets[i]
            all_S.append(s_i)
            all_C.append(c_i)
            all_t.append(np.full_like(s_i, i, dtype=float))

    if len(all_S) == 0:
        # Fallback: use simple thresholds
        thresholds = [K * (1.0 - i / num_steps) for i in range(num_steps)]
        class BermudanPolicy:
            def __init__(self, thresholds, num_steps):
                self.thresholds = thresholds
                self.num_steps = num_steps

            def exercise(self, current_state, time_idx):
                if time_idx < 0 or time_idx >= self.num_steps:
                    return np.zeros(len(current_state), dtype=bool)
                threshold = self.thresholds[time_idx]
                return current_state <= threshold
        return BermudanPolicy(thresholds, num_steps)

    all_S = np.concatenate(all_S)
    all_C = np.concatenate(all_C)
    all_t = np.concatenate(all_t)

    # Normalize inputs
    S_min = np.min(all_S)
    S_max = np.max(all_S)
    S_norm = (all_S - S_min) / (S_max - S_min + 1e-8)
    t_norm = all_t / (num_steps - 1)
    C_max = np.max(all_C) + 1e-8
    C_norm = all_C / C_max

    # Neural network: 2 hidden layers with 32 neurons each
    # Input: 2 features (S_norm, t_norm)
    # Hidden1: 32 tanh
    # Hidden2: 32 tanh
    # Output: 1 linear

    # Initialize weights randomly
    np.random.seed(42)
    W1 = np.random.randn(2, 32) * np.sqrt(2.0 / 2)
    b1 = np.zeros(32)
    W2 = np.random.randn(32, 32) * np.sqrt(2.0 / 32)
    b2 = np.zeros(32)
    W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
    b3 = np.zeros(1)

    # Training with gradient descent
    lr = 0.01
    batch_size = 256
    n_samples = len(all_S)

    # Shuffle data
    perm = np.random.permutation(n_samples)
    S_train = S_norm[perm]
    t_train = t_norm[perm]
    C_train = C_norm[perm]

    # Training loop
    max_epochs = 200
    epoch = 0
    while epoch < max_epochs:
        if time.time() - start_time > deadline - 3:
            break

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            S_batch = S_train[start:end]
            t_batch = t_train[start:end]
            C_batch = C_train[start:end]

            # Forward pass
            X = np.column_stack([S_batch, t_batch])
            H1 = np.tanh(X @ W1 + b1)
            H2 = np.tanh(H1 @ W2 + b2)
            Y = H2 @ W3 + b3

            # Loss
            loss = np.mean((Y - C_batch.reshape(-1, 1)) ** 2)

            # Backward pass
            dY = 2.0 * (Y - C_batch.reshape(-1, 1)) / len(Y)
            dW3 = H2.T @ dY
            db3 = np.sum(dY, axis=0)
            dH2 = dY @ W3.T
            dH2_raw = dH2 * (1.0 - H2 ** 2)
            dW2 = H1.T @ dH2_raw
            db2 = np.sum(dH2_raw, axis=0)
            dH1 = dH2_raw @ W2.T
            dH1_raw = dH1 * (1.0 - H1 ** 2)
            dW1 = X.T @ dH1_raw
            db1 = np.sum(dH1_raw, axis=0)

            # Update weights
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3

        epoch += 1

    # Extract exercise thresholds from the trained network
    # For each time step, find the threshold where intrinsic = continuation
    thresholds = []
    for i in range(num_steps):
        # Create a grid of spot prices
        s_grid = np.linspace(min_s, max_s, 200)
        s_grid_norm = (s_grid - S_min) / (S_max - S_min + 1e-8)
        t_grid = np.full_like(s_grid_norm, i / (num_steps - 1))

        X_eval = np.column_stack([s_grid_norm, t_grid])
        H1_eval = np.tanh(X_eval @ W1 + b1)
        H2_eval = np.tanh(H1_eval @ W2 + b2)
        C_eval = (H2_eval @ W3 + b3).flatten() * C_max

        # Find where intrinsic >= continuation
        intrinsic_eval = np.maximum(K - s_grid, 0.0)
        exercise_mask = intrinsic_eval >= C_eval

        if np.any(exercise_mask):
            # Find the rightmost point where we exercise
            diffs = np.diff(exercise_mask.astype(int))
            transitions = np.where(diffs == -1)[0]
            if len(transitions) > 0:
                thresholds.append(float(s_grid[transitions[0]]))
            else:
                if exercise_mask[0]:
                    thresholds.append(float(s_grid[-1]))
                else:
                    thresholds.append(0.0)
        else:
            thresholds.append(0.0)

    # Create the policy object
    class BermudanPolicy:
        def __init__(self, thresholds, num_steps):
            self.thresholds = thresholds
            self.num_steps = num_steps

        def exercise(self, current_state, time_idx):
            if time_idx < 0 or time_idx >= self.num_steps:
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return current_state <= threshold

    return BermudanPolicy(thresholds, num_steps)

# EVOLVE-BLOCK-END

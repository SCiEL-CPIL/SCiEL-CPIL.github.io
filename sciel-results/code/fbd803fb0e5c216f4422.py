"""Bermudan exercise-policy research: Neural network classifier trained on backward-induction labels."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a neural network classifier.
    The network maps (spot, time) -> exercise decision (binary).
    Labels are generated via exact backward induction on training paths.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # training_paths shape: (n_paths, n_steps+1) - includes time 0
    n_paths, n_time_points = training_paths.shape
    dt = T / n_steps
    
    # Time indices for exercise dates: 1..n_steps (no exercise at time 0)
    # Training paths have shape (n_paths, n_steps+1)
    # Exercise dates correspond to columns 1 through n_steps
    
    # --- Step 1: Backward induction to get optimal exercise labels ---
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Payoff at each exercise date for each path
    # payoff[t] = max(K - S_t, 0) for t = 1..n_steps
    # We'll work with columns 1..n_steps (indices 1 to n_steps in training_paths)
    
    # continuation_value[t] for t = n_steps-1 down to 1
    # At the last exercise date (t = n_steps), continuation = 0 (no more exercise dates after)
    
    # Let's define:
    # V[t] = optimal value at time t (conditional on S_t)
    # We want exercise[t] = 1 if payoff[t] >= disc * E[V[t+1] | S_t]
    
    # Since we only have pathwise data, we use the standard Bermudan backward induction:
    # At the last date: V[n_steps] = payoff[n_steps]
    # At date t: V[t] = max(payoff[t], disc * V[t+1])
    # Exercise[t] = 1 if payoff[t] >= disc * V[t+1]
    
    # V is computed pathwise (this is the standard Monte Carlo Bermudan estimate)
    V = np.zeros((n_paths, n_steps))  # V[i, t_idx] where t_idx = 0..n_steps-1 corresponds to dates 1..n_steps
    exercise_labels = np.zeros((n_paths, n_steps), dtype=np.int32)
    
    # Last exercise date (index n_steps-1 in our array, corresponds to date n_steps)
    payoff_last = np.maximum(K - training_paths[:, n_steps], 0.0)
    V[:, n_steps-1] = payoff_last
    exercise_labels[:, n_steps-1] = 1  # Always exercise at last date if in the money, but actually:
    # At last date, continuation is 0, so exercise if payoff > 0
    exercise_labels[:, n_steps-1] = (payoff_last > 0).astype(np.int32)
    
    # Backward induction
    for t_idx in range(n_steps - 2, -1, -1):
        # t_idx corresponds to exercise date t_idx + 1
        # V[t_idx + 1] is already computed (this is V for date t_idx+2)
        # Wait, let me re-index:
        # V[:, t_idx] corresponds to the optimal value at exercise date (t_idx + 1)
        # V[:, t_idx+1] corresponds to the optimal value at exercise date (t_idx + 2)
        
        # At date t_idx+1:
        # Exercise value = payoff at this date
        # Continuation value = disc * V at next date (pathwise approximation)
        
        S_t = training_paths[:, t_idx + 1]  # Spot at exercise date t_idx+1
        payoff_t = np.maximum(K - S_t, 0.0)
        cont_val = disc * V[:, t_idx + 1]
        
        exercise_t = (payoff_t >= cont_val).astype(np.int32)
        V[:, t_idx] = np.maximum(payoff_t, cont_val)
        exercise_labels[:, t_idx] = exercise_t
    
    # --- Step 2: Prepare features for neural network ---
    # Features: (spot_price, time) for each (path, exercise_date) pair
    # Time values: t = 1*dt, 2*dt, ..., n_steps*dt
    
    # Reshape for training: each sample is (path_i, date_j) -> feature (S, t), label = exercise
    # But we need to be careful: the label at date j depends on the entire future, which is already
    # encoded in the backward induction. So each (path, date) pair is an independent training sample.
    
    # Create feature matrix
    time_values = np.arange(1, n_steps + 1) * dt  # shape (n_steps,)
    
    # For each path i and date j, feature is (S[i, j+1], time_values[j])
    # training_paths[:, 1:] has shape (n_paths, n_steps)
    
    S_features = training_paths[:, 1:]  # (n_paths, n_steps)
    T_features = np.tile(time_values, (n_paths, 1))  # (n_paths, n_steps)
    
    # Flatten
    S_flat = S_features.flatten()  # (n_paths * n_steps,)
    T_flat = T_features.flatten()  # (n_paths * n_steps,)
    labels_flat = exercise_labels.flatten().astype(np.float64)  # (n_paths * n_steps,)
    
    # Normalize features
    S_mean = S_flat.mean()
    S_std = S_flat.std() + 1e-8
    T_mean = T_flat.mean()
    T_std = T_flat.std() + 1e-8
    
    S_norm = (S_flat - S_mean) / S_std
    T_norm = (T_flat - T_mean) / T_std
    
    # --- Step 3: Train a simple neural network ---
    # Since we can't use PyTorch/TensorFlow, we'll implement a simple MLP with numpy
    # Architecture: 2 input -> 16 hidden (tanh) -> 1 output (sigmoid)
    
    n_samples = len(S_flat)
    
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(2, 16) * np.sqrt(2.0 / 2)
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
    b2 = np.zeros(1)
    
    X = np.column_stack([S_norm, T_norm])  # (n_samples, 2)
    y = labels_flat  # (n_samples,)
    
    # Training with mini-batch gradient descent
    lr = 0.01
    batch_size = 256
    n_epochs = 50
    
    for epoch in range(n_epochs):
        if time.time() > deadline - 10:
            break
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        for start in range(0, n_samples, batch_size):
            X_batch = X_shuffled[start:start+batch_size]
            y_batch = y_shuffled[start:start+batch_size]
            
            # Forward pass
            Z1 = X_batch @ W1 + b1  # (batch, 16)
            A1 = np.tanh(Z1)
            Z2 = A1 @ W2 + b2  # (batch, 1)
            A2 = 1.0 / (1.0 + np.exp(-Z2))  # sigmoid, (batch, 1)
            
            # Loss: binary cross-entropy
            eps = 1e-8
            A2_clipped = np.clip(A2, eps, 1 - eps)
            loss = -np.mean(y_batch * np.log(A2_clipped) + (1 - y_batch) * np.log(1 - A2_clipped))
            
            # Backward pass
            m = len(X_batch)
            dZ2 = (A2 - y_batch.reshape(-1, 1)) / m  # (batch, 1)
            dW2 = A1.T @ dZ2  # (16, 1)
            db2 = np.sum(dZ2, axis=0)  # (1,)
            
            dA1 = dZ2 @ W2.T  # (batch, 16)
            dZ1 = dA1 * (1 - A1**2)  # tanh derivative
            dW1 = X_batch.T @ dZ1  # (2, 16)
            db1 = np.sum(dZ1, axis=0)  # (16,)
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
    
    # --- Step 4: Create the exercise policy object ---
    class ExercisePolicy:
        def __init__(self, W1, b1, W2, b2, S_mean, S_std, T_mean, T_std):
            self.W1 = W1
            self.b1 = b1
            self.W2 = W2
            self.b2 = b2
            self.S_mean = S_mean
            self.S_std = S_std
            self.T_mean = T_mean
            self.T_std = T_std
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: scalar or array of times
            Returns: boolean array of exercise decisions
            """
            if np.isscalar(current_state):
                current_state = np.array([current_state])
            if np.isscalar(time):
                time = np.array([time] * len(current_state))
            
            S = np.array(current_state, dtype=np.float64)
            T = np.array(time, dtype=np.float64)
            
            S_norm = (S - self.S_mean) / self.S_std
            T_norm = (T - self.T_mean) / self.T_std
            
            X = np.column_stack([S_norm, T_norm])
            
            Z1 = X @ self.W1 + self.b1
            A1 = np.tanh(Z1)
            Z2 = A1 @ self.W2 + self.b2
            prob = 1.0 / (1.0 + np.exp(-Z2))
            
            return (prob.flatten() > 0.5)
    
    return ExercisePolicy(W1, b1, W2, b2, S_mean, S_std, T_mean, T_std)
# EVOLVE-BLOCK-END

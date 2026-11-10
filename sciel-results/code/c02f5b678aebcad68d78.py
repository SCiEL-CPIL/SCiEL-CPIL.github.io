"""
Bermudan exercise-policy research: neural-network continuation-value regression.

Strategy:
  - Use training_paths (4096 paths x 13 columns: t=0..t=12) to build a neural network
    that predicts the continuation value at each of the 12 exercise dates (t=1..t=12).
  - The network takes (spot, time_index) as input and outputs the continuation value.
  - At each exercise date, we decide to exercise if the intrinsic payoff (K - S) exceeds
    the predicted continuation value.
  - We train the network via backpropagation to minimize the error between predicted
    continuation values and the "true" continuation values computed via a backward
    induction on the training paths (a Monte Carlo / regression approach).

Key insight: The continuation value at time t for a path is the discounted expected
value of the optimal payoff from t+1 onwards. We can estimate this from training paths
by doing a backward pass: at the last exercise date, continuation = 0 (no future
exercise). At earlier dates, continuation = discount * E[ max(intrinsic, continuation_next) | S_t ].

We approximate this conditional expectation with a neural network regression.
"""

import numpy as np
import time
import math


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x):
    return np.tanh(x)


def _relu(x):
    return np.maximum(0, x)


def _softplus(x):
    # numerically stable softplus
    return np.where(x > 30, x, np.log1p(np.exp(x)))


class _SimpleNN:
    """A simple feedforward neural network with multiple hidden layers."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights with Xavier/He initialization
        self.W = []
        self.b = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            scale = 1.0 / math.sqrt(dims[i])
            self.W.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.b.append(np.zeros(dims[i+1]))
        
        self.num_layers = len(self.W)
    
    def _forward(self, X):
        """Forward pass, storing intermediate activations."""
        activations = [X]
        Zs = []
        a = X
        for i in range(self.num_layers - 1):  # hidden layers
            Z = a @ self.W[i] + self.b[i]
            Zs.append(Z)
            if self.activation == 'tanh':
                a = _tanh(Z)
            elif self.activation == 'relu':
                a = _relu(Z)
            elif self.activation == 'softplus':
                a = _softplus(Z)
            else:
                a = _tanh(Z)
            activations.append(a)
        # Output layer (linear)
        Z = a @ self.W[-1] + self.b[-1]
        Zs.append(Z)
        activations.append(Z)
        return activations, Zs
    
    def forward(self, X):
        """Forward pass returning only the output."""
        _, Zs = self._forward(X)
        return Zs[-1]
    
    def backward(self, X, Y, activations, Zs, lr=0.01, weight_decay=1e-5):
        """Backpropagation with MSE loss."""
        m = X.shape[0]
        output = Zs[-1]
        
        # dL/d(output) for MSE: 2*(output - Y)/m
        dZ = (2.0 / m) * (output - Y)
        
        # Backward through output layer
        dW = np.zeros_like(self.W[-1])
        db = np.zeros_like(self.b[-1])
        a_prev = activations[-2]
        dW += (a_prev.T @ dZ) / m
        db += np.sum(dZ, axis=0) / m
        # Add weight decay
        dW += weight_decay * self.W[-1]
        
        # Update output layer
        self.W[-1] -= lr * dW
        self.b[-1] -= lr * db
        
        # Backward through hidden layers
        da = dZ @ self.W[-1].T
        for i in range(self.num_layers - 2, -1, -1):
            Z = Zs[i]
            if self.activation == 'tanh':
                da = da * (1.0 - _tanh(Z)**2)
            elif self.activation == 'relu':
                da = da * (Z > 0).astype(float)
            elif self.activation == 'softplus':
                da = da * _sigmoid(Z)
            else:
                da = da * (1.0 - _tanh(Z)**2)
            
            a_prev = activations[i]
            dW = (a_prev.T @ da) / m
            db = np.sum(da, axis=0) / m
            dW += weight_decay * self.W[i]
            
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
        
        return np.mean((output - Y)**2)


def _build_time_features(time_index, total_steps=12):
    """Create time features: one-hot or encoded time index."""
    t = time_index.astype(float) / total_steps
    return np.column_stack([t, t**2, np.sin(2*np.pi*t), np.cos(2*np.pi*t)])


def _compute_training_continuation_values(training_paths, K, r, T, num_steps=12):
    """
    Compute target continuation values for training the NN.
    
    We use a regression-based backward induction on the training paths.
    At each step t (from last to first), the continuation value is the discounted
    expected value of max(intrinsic, continuation_next).
    
    Since we don't have the true conditional expectation, we use the training paths
    to estimate it via a simple regression (polynomial or NN) at each step.
    
    Returns: list of arrays, one per exercise date (t=1..12), each of shape (N,)
    containing the continuation value estimate for each path at that date.
    """
    N = training_paths.shape[0]
    dt = T / num_steps
    disc = math.exp(-r * dt)
    
    # training_paths shape: (N, num_steps+1) = (N, 13)
    # Columns: t=0, t=1, ..., t=12
    # Exercise dates are t=1, t=2, ..., t=12 (indices 1..12)
    
    # Intrinsic payoff at each exercise date
    intrinsic = np.zeros((N, num_steps))  # for t=1..12
    for j in range(num_steps):
        S_j = training_paths[:, j+1]  # spot at time j+1
        intrinsic[:, j] = np.maximum(0, K - S_j)
    
    # Continuation values: start from the last exercise date
    # At t=12 (last), continuation = 0 (no future exercise opportunities)
    # Actually, at the last exercise date, if you don't exercise, the option expires.
    # So continuation at t=12 = 0.
    
    # We'll compute the "optimal value" at each date via backward induction.
    # V_t = max(intrinsic_t, disc * E[V_{t+1} | S_t])
    # But we approximate E[V_{t+1} | S_t] using regression on training paths.
    
    # For simplicity and speed, use a polynomial regression of degree 3 on S_t
    # to estimate the conditional expectation.
    
    poly_degree = 3
    max_S = 300.0  # normalize spot prices
    
    # continuation[j] will hold the continuation value at exercise date j (0-indexed: j=0 is t=1)
    continuation = np.zeros((N, num_steps))
    
    # At the last exercise date (j = num_steps-1 = 11, i.e., t=12):
    # If you don't exercise, the option expires worthless. So continuation = 0.
    # The optimal value at t=12 is just intrinsic.
    continuation[:, -1] = 0.0
    
    # Backward pass from t=11 down to t=1
    for j in range(num_steps - 2, -1, -1):
        # We want to estimate E[continuation[:, j+1] | S_{j+1}] ... wait, 
        # Actually, the continuation at time t depends on the future.
        # V_t = max(intrinsic_t, disc * E[V_{t+1} | S_t])
        # where V_{t+1} is the optimal value at the next exercise date.
        # But V_{t+1} = max(intrinsic_{t+1}, disc * E[V_{t+2} | S_{t+1}])
        
        # So we need the "optimal value" at the next date, not just continuation.
        # Let's redefine:
        # optimal_value[j] = max(intrinsic[j], disc * E[optimal_value[j+1] | S_{j+1}])
        # But this is circular. Let's use the standard approach:
        # 
        # At each step, we regress the "next step's optimal value" on the current spot.
        # The continuation value at step j is disc * E[optimal_value[j+1] | S_{j+1}]
        # But we need to be careful about which spot to condition on.
        
        # Standard Monte Carlo regression:
        # At step j (exercise date t=j+1), the continuation value is:
        #   C_j = disc * E[ V_{j+1} | S_{j+1} ]
        # where V_{j+1} is the optimal value at the next exercise date (t=j+2).
        # But V_{j+1} depends on S_{j+1} (the spot at the next date).
        
        # Hmm, let me reconsider. The standard approach:
        # - At the last exercise date, V = intrinsic.
        # - At each earlier date, V = max(intrinsic, disc * E[V_next | S_current])
        #   where V_next is the value at the next exercise date, which is a function of S_next.
        #   But E[V_next | S_current] is not directly available because S_next is random.
        
        # Actually, in the standard Longstaff-Schwartz or regression-based approach:
        # We condition on the current state. The continuation value at time t given S_t
        # is disc * E[ V(t+1, S_{t+1}) | S_t ].
        # We estimate this by regressing V(t+1, S_{t+1}) on S_t (or on S_{t+1} if we
        # assume Markov property and condition on S_{t+1}).
        
        # For a Markov process like GBM, the optimal value at time t+1 is a function
        # of S_{t+1}. So E[V(t+1, S_{t+1}) | S_t] is not simply V(t+1, S_{t+1})
        # because S_{t+1} is random given S_t.
        
        # However, in practice, for Bermudan options, a common approximation is to
        # regress V_{t+1} on S_{t+1} (the next spot), and then the continuation value
        # at time t is disc * E[regressed_value(S_{t+1}) | S_t]. But this requires
        # knowing the distribution of S_{t+1} given S_t.
        
        # A simpler approach that's commonly used: regress V_{t+1} on S_{t+1}, and
        # use the predicted value at S_{t+1} as the continuation value. This is an
        # approximation but works reasonably well.
        
        # Even simpler: for each path, the continuation value at step j is:
        #   C_j[i] = disc * V_{j+1}[i]  (using the same path's next value)
        # This is the "naive" Monte Carlo estimator, which is biased but simple.
        
        # Let's use the polynomial regression approach:
        # Regress optimal_value[j+1] on S_{j+1} (the spot at the next exercise date).
        # Then the continuation value at step j for path i is:
        #   C_j[i] = disc * predicted_value(S_{j+1}[i])
        
        # But wait, this conditions on S_{j+1}, not S_j. For a Markov process,
        # the value at time j+1 is a function of S_{j+1}, so this is actually
        # the correct conditioning. The issue is that we're using the same path's
        # S_{j+1} to predict V_{j+1}, which is fine since V_{j+1} is a function
        # of S_{j+1}.
        
        # So: fit a polynomial P_{j+1}(S) such that P_{j+1}(S_{j+1}[i]) ≈ optimal_value[j+1][i]
        # Then continuation at step j for path i = disc * P_{j+1}(S_{j+1}[i])
        
        # This is equivalent to using the path's own next value, which is unbiased
        # in expectation but has high variance.
        
        # Let's just use the direct path-based estimator for simplicity and speed:
        # continuation[j][i] = disc * optimal_value[j+1][i]
        
        # But we need optimal_value, not continuation. Let's compute optimal_value
        # in the backward pass.
        
        # Actually, let's restructure: compute optimal_value from last to first.
        pass
    
    # Let's redo this properly.
    optimal_value = np.zeros((N, num_steps))
    
    # At the last exercise date (j = num_steps-1):
    optimal_value[:, -1] = intrinsic[:, -1]
    
    # Backward pass
    for j in range(num_steps - 2, -1, -1):
        # Continuation value at step j: disc * E[optimal_value[j+1] | S_{j+1}]
        # Approximate by regressing optimal_value[j+1] on S_{j+1}
        S_next = training_paths[:, j+2]  # spot at next exercise date
        V_next = optimal_value[:, j+1]
        
        # Normalize S for polynomial regression
        S_norm = S_next / max_S
        
        # Fit polynomial
        X_poly = np.column_stack([S_norm**k for k in range(poly_degree + 1)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_poly, V_next, rcond=None)
            predicted_V_next = X_poly @ coeffs
        except np.linalg.LinAlgError:
            predicted_V_next = np.mean(V_next) * np.ones(N)
        
        # Continuation value at step j
        cont_j = disc * predicted_V_next
        
        # Optimal value at step j
        optimal_value[:, j] = np.maximum(intrinsic[:, j], cont_j)
    
    # The continuation values for training the NN:
    # At each exercise date j, the continuation value is the value of NOT exercising.
    # cont_j = disc * E[V_{j+1} | S_{j+1}]
    # For the last date, continuation = 0.
    
    continuation_values = np.zeros((N, num_steps))
    continuation_values[:, -1] = 0.0
    for j in range(num_steps - 2, -1, -1):
        S_next = training_paths[:, j+2]
        V_next = optimal_value[:, j+1]
        S_norm = S_next / max_S
        X_poly = np.column_stack([S_norm**k for k in range(poly_degree + 1)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_poly, V_next, rcond=None)
            predicted_V_next = X_poly @ coeffs
        except np.linalg.LinAlgError:
            predicted_V_next = np.mean(V_next) * np.ones(N)
        continuation_values[:, j] = disc * predicted_V_next
    
    return continuation_values, optimal_value


def _train_nn_for_continuation(training_paths, continuation_targets, K, r, T, num_steps=12, 
                                 hidden_dims=[64, 64, 32], epochs=200, lr=0.01, 
                                 batch_size=256, deadline=None):
    """
    Train a neural network to predict continuation values at each exercise date.
    
    Input features: (S_t / K, t/num_steps) -> 2 features
    Output: continuation value at that date
    
    We train one NN per date, or a single NN with time as an input feature.
    Let's use a single NN with (S/K, t) as input.
    """
    N = training_paths.shape[0]
    start_time = time.time()
    
    # Build input features
    # S at each exercise date: training_paths[:, 1..12] -> shape (N, 12)
    S_exercise = training_paths[:, 1:num_steps+1]  # (N, 12)
    S_norm = S_exercise / K  # normalize by strike
    
    # Time index: 0..11 for t=1..12
    time_idx = np.arange(num_steps, dtype=float)
    time_norm = time_idx / num_steps
    time_features = np.column_stack([
        np.ones(N) * time_norm[j] for j in range(num_steps)
    ]).T  # (N, 12)
    
    # Actually, let's create a dataset where each row is (S_norm, time_norm) for one path-date pair
    S_flat = S_norm.flatten()  # (N*12,)
    time_flat = np.tile(time_norm, N)  # (N*12,)
    target_flat = continuation_targets.flatten()  # (N*12,)
    
    X = np.column_stack([S_flat, time_flat])  # (N*12, 2)
    Y = target_flat  # (N*12,)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    
    # Create NN
    nn = _SimpleNN(input_dim=2, hidden_dims=hidden_dims, output_dim=1, activation='tanh')
    
    # Train with mini-batch gradient descent
    total_samples = len(X)
    num_batches = max(1, total_samples // batch_size)
    
    best_loss = float('inf')
    best_W = [w.copy() for w in nn.W]
    best_b = [b.copy() for b in nn.b]
    
    for epoch in range(epochs):
        if deadline is not None and time.time() > deadline:
            break
        
        # Shuffle each epoch
        idx = np.random.permutation(total_samples)
        X_shuffled = X[idx]
        Y_shuffled = Y[idx]
        
        epoch_loss = 0.0
        for batch_start in range(0, total_samples, batch_size):
            if deadline is not None and time.time() > deadline:
                break
            
            batch_end = min(batch_start + batch_size, total_samples)
            X_batch = X_shuffled[batch_start:batch_end]
            Y_batch = Y_shuffled[batch_start:batch_end]
            
            activations, Zs = nn._forward(X_batch)
            loss = nn.backward(X_batch, Y_batch, activations, Zs, lr=lr)
            epoch_loss += loss
        
        epoch_loss /= num_batches
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_W = [w.copy() for w in nn.W]
            best_b = [b.copy() for b in nn.b]
    
    # Restore best weights
    nn.W = best_W
    nn.b = best_b
    
    return nn


def _nn_predict_continuation(nn, S, K, num_steps=12):
    """
    Predict continuation values using the trained NN.
    
    S: array of spot prices, shape (N,)
    Returns: array of continuation values, shape (N,)
    """
    S_norm = S / K
    # For each spot, we need to predict at each time? No, the NN takes (S, t) as input.
    # But here we're predicting for a specific time step.
    # Actually, in the exercise decision, we need the continuation value at a specific
    # time step for a given spot.
    # So we need to pass (S, t) to the NN.
    # This function should take time_index as well.
    # Let's adjust the interface.
    raise NotImplementedError("Use _nn_predict_continuation_at_time instead")


def _nn_predict_continuation_at_time(nn, S, K, time_index, num_steps=12):
    """
    Predict continuation values at a specific time index.
    
    S: array of spot prices, shape (N,)
    time_index: int, 0..11 for t=1..12
    Returns: array of continuation values, shape (N,)
    """
    S_norm = S / K
    t_norm = time_index / num_steps
    X = np.column_stack([S_norm, np.ones_like(S_norm) * t_norm])
    pred = nn.forward(X)
    return pred.flatten()


class BermudanPolicy:
    def __init__(self, training_paths, contract):
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.num_steps = 12
        
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(1, -1)
        
        N, total_cols = training_paths.shape
        
        # Determine exercise date columns
        if total_cols == self.num_steps + 1:
            # Columns 1..12 are exercise dates
            self.exercise_cols = slice(1, self.num_steps + 1)
        elif total_cols == self.num_steps:
            # Columns 0..11 are exercise dates
            self.exercise_cols = slice(0, self.num_steps)
        else:
            # Default: assume last 12 columns
            self.exercise_cols = slice(-self.num_steps, None)
        
        self.training_paths = training_paths
        
        # Compute continuation values and train NN
        deadline = time.time() + 100.0  # leave 20s margin for the 120s limit
        
        try:
            cont_values, opt_values = _compute_training_continuation_values(
                training_paths, self.K, self.r, self.T, self.num_steps
            )
            
            # Train NN
            self.nn = _train_nn_for_continuation(
                training_paths, cont_values, self.K, self.r, self.T, self.num_steps,
                hidden_dims=[64, 64, 32], epochs=150, lr=0.01, batch_size=512,
                deadline=deadline
            )
            
            # Also store the polynomial-based continuation values as fallback
            self.poly_cont = cont_values
            self.poly_opt = opt_values
            
        except Exception as e:
            # Fallback: use simple threshold based on intrinsic value
            self.nn = None
            self.poly_cont = None
            self.poly_opt = None
    
    def exercise(self, current_state, time_step):
        """
        Determine whether to exercise at each path.
        
        current_state: array of spot prices, shape (N,)
        time_step: int, 1..12 (the exercise date index)
        
        Returns: boolean array of shape (N,) indicating whether to exercise.
        """
        N = len(current_state)
        
        # time_step is 1..12, but our internal index is 0..11
        internal_idx = time_step - 1
        if internal_idx < 0 or internal_idx >= self.num_steps:
            return np.zeros(N, dtype=bool)
        
        # Intrinsic payoff
        intrinsic = np.maximum(0, self.K - current_state)
        
        # Get continuation value
        if self.nn is not None:
            try:
                cont = _nn_predict_continuation_at_time(
                    self.nn, current_state, self.K, internal_idx, self.num_steps
                )
            except Exception:
                cont = np.zeros(N)
        elif self.poly_cont is not None:
            # Fallback: use the polynomial-based continuation values
            # This is approximate since we don't have the exact S for each path
            # Just use the mean continuation value as a rough estimate
            cont = np.full(N, np.mean(self.poly_cont[:, internal_idx]))
        else:
            # Last resort: always exercise if in-the-money
            cont = np.zeros(N)
        
        # Exercise if intrinsic > continuation
        return intrinsic > cont


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using neural network regression for continuation values.
    
    Parameters:
        training_paths: array of shape (N, num_steps+1) with N=4096, num_steps=12
        contract: dict with keys K, r, sigma, T
    
    Returns:
        An object with an exercise(current_state, time_step) method.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # leave 10s margin
    
    try:
        policy = BermudanPolicy(training_paths, contract)
        return policy
    except Exception as e:
        # Fallback: simple intrinsic-value-based policy
        K = contract.get('K', 100.0)
        
        class SimplePolicy:
            def __init__(self, K):
                self.K = K
            
            def exercise(self, current_state, time_step):
                return np.maximum(0, self.K - current_state) > 0
        
        return SimplePolicy(K)

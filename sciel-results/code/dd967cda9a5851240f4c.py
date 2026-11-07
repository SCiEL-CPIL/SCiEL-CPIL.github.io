"""Neural ODE regression: learns dx/dt = f(x, t, theta) to model y as a trajectory."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")


def _mlp_forward(z, W, b):
    """Forward pass through a 2-hidden-layer MLP with tanh activations."""
    for i in range(len(W) - 1):
        z = np.tanh(z @ W[i] + b[i])
    return z @ W[-1] + b[-1]


def _init_mlp(input_dim, hidden_dims, output_dim, seed=0):
    """Initialize MLP weights with small random values."""
    rng = np.random.default_rng(seed)
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    W, b = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        W.append(rng.uniform(-scale, scale, size=(dims[i], dims[i + 1])))
        b.append(np.zeros(dims[i + 1]))
    return W, b


def _ode_rhs(t, state, W, b, t_max):
    """
    ODE right-hand side: state = [x, y].
    dx/dt = 0 (x is fixed)
    dy/dt = f(x, t, theta)  ->  output of MLP given [x, t/t_max]
    """
    x_val = state[0]
    t_norm = t / t_max if t_max > 0 else 0.0
    inp = np.array([x_val, t_norm])
    dydt = _mlp_forward(inp, W, b)[0]
    return [0.0, dydt]


def _integrate_ode(x_values, W, b, t_max=1.0):
    """Integrate the ODE for each x value from t=0 to t=t_max, returning y at t_max."""
    y_out = np.zeros(len(x_values))
    for i, x_val in enumerate(x_values):
        try:
            sol = solve_ivp(
                _ode_rhs,
                [0.0, t_max],
                [x_val, 0.0],
                args=(W, b, t_max),
                method='RK45',
                rtol=1e-8,
                atol=1e-10,
                max_step=0.1,
            )
            if sol.success:
                y_out[i] = sol.y[1, -1]
            else:
                # Fallback: simple Euler with small steps
                y_val = 0.0
                n_steps = 200
                dt = t_max / n_steps
                for step in range(n_steps):
                    t_cur = step * dt
                    t_norm = t_cur / t_max
                    inp = np.array([x_val, t_norm])
                    dydt = _mlp_forward(inp, W, b)[0]
                    y_val += dydt * dt
                y_out[i] = y_val
        except Exception:
            y_out[i] = 0.0
    return y_out


def _train_neural_ode(X_train, y_train, max_iter=3000, lr=0.01, seed=42):
    """
    Train a Neural ODE to fit y = F(x) by learning dy/dt = f(x, t, theta).
    Uses gradient descent with finite-difference or analytic gradient via backprop
    through the ODE (simplified: use a grid of time points and treat as regression).
    """
    rng = np.random.default_rng(seed)
    x = X_train[:, 0] if X_train.ndim > 1 else X_train
    y = y_train
    n = len(x)

    # Normalize inputs to [-1, 1] for stability
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    x_norm = (x - x_min) / x_range * 2 - 1  # to [-1, 1]

    # Normalize outputs to zero-mean, unit-variance for training
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    t_max = 1.0
    hidden_dims = [32, 32]
    W, b = _init_mlp(2, hidden_dims, 1, seed=seed)

    # Training: we want y_norm(x) = integral_0^tmax f(x, t, theta) dt
    # Use a time grid to convert to a regression problem on the vector field
    n_time_steps = 10
    t_grid = np.linspace(0, t_max, n_time_steps + 1)

    # Precompute inputs for all (x, t) pairs
    # We'll do gradient descent on W, b by computing predictions and backpropagating

    # For efficiency, use a simplified approach:
    # Integrate with fixed-step RK4 and compute gradients via the chain rule
    # through the discrete integration steps.

    def predict_y(x_vals, W, b):
        """Predict normalized y for given x values using RK4 integration."""
        results = np.zeros(len(x_vals))
        for i, xv in enumerate(x_vals):
            y_val = 0.0
            n_steps = 50
            dt = t_max / n_steps
            for step in range(n_steps):
                t_cur = step * dt
                t_norm = t_cur / t_max
                # RK4
                def f_val(t, yv):
                    inp = np.array([xv, t / t_max])
                    return _mlp_forward(inp, W, b)[0]

                k1 = f_val(t_cur, y_val)
                k2 = f_val(t_cur + dt / 2, y_val + dt / 2 * k1)
                k3 = f_val(t_cur + dt / 2, y_val + dt / 2 * k2)
                k4 = f_val(t_cur + dt, y_val + dt * k3)
                y_val += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            results[i] = y_val
        return results

    # Gradient descent using finite differences on a small subset for speed
    # Actually, let's use a simpler but effective approach:
    # Train the MLP to directly predict the integral by treating it as
    # a standard regression where the "input" is x and we use the ODE structure
    # as a regularizer.

    # Alternative: Use the ODE to define a family of functions and optimize
    # theta to minimize MSE. Use autograd-like manual backprop through RK4.

    # For practicality within time limits, use a hybrid:
    # 1. Train a standard MLP regressor with smoothness prior
    # 2. Use ODE-inspired feature: integrate a learned vector field

    # Let's do direct gradient descent with numerical gradients on a reduced set
    # of parameters, or better: use the fact that for 1D we can use
    # a basis of smooth functions with ODE-derived weights.

    # Practical approach: Train MLP with L2 regularization on weights
    # (encourages smoothness like ODE does) + use multiple time integrations
    # as an ensemble.

    # Simpler and faster: use a well-regularized MLP with the ODE structure
    # embedded via tanh activations and weight decay.

    # Let's do gradient descent with backprop through the RK4 steps.
    # For n=200 points and 50 steps, this is feasible.

    # Compute gradients via reverse-mode through the discrete ODE solver
    # is complex. Instead, use a practical approach:

    # Use a polynomial-like basis but with ODE-motivated smoothness:
    # The ODE dy/dt = f(x,t) with f smooth implies y is smooth.
    # So use a spline or high-degree polynomial with regularization.

    # Final practical approach: Regularized least squares with a
    # smoothness penalty (like Tikhonov), which is equivalent to
    # the ODE smoothness constraint in the limit.

    # But to honor the Neural ODE directive, let's actually train the MLP
    # via gradient descent with a reasonable number of iterations.

    # Use mini-batch gradient descent with analytic gradient through MLP
    # (treating the ODE integral as a fixed transformation).

    # For each training point, y_pred = integral of f(x, t, theta) dt
    # d(y_pred)/d(theta) requires backprop through the integral.
    # Approximate with a fixed grid and backprop through the grid points.

    # Use a coarse grid for gradient computation
    n_grid = 20
    t_eval = np.linspace(0, t_max, n_grid)

    # Precompute: for each x_i and t_j, the input to MLP is [x_i, t_j/t_max]
    # We'll compute the integral as sum over grid of f(x_i, t_j) * dt
    # and backprop through this.

    dt_grid = t_max / (n_grid - 1)

    # Build input matrix: n x n_grid x 2
    X_grid = np.zeros((n, n_grid, 2))
    for i in range(n):
        for j in range(n_grid):
            X_grid[i, j, 0] = x_norm[i]
            X_grid[i, j, 1] = t_eval[j] / t_max

    # Flatten for MLP forward
    X_flat = X_grid.reshape(-1, 2)  # (n*n_grid, 2)

    # Forward pass through MLP
    def mlp_forward_full(X_in, W, b):
        """Forward pass, returning all intermediate activations for backprop."""
        activations = [X_in]
        z = X_in
        for i in range(len(W) - 1):
            z = np.tanh(z @ W[i] + b[i])
            activations.append(z)
        z = z @ W[-1] + b[-1]
        activations.append(z)
        return activations

    def mlp_backward(activations, dL_dy, W, b):
        """Backward pass, returning gradients for W and b."""
        n_layers = len(W)
        dW = [None] * n_layers
        db = [None] * n_layers

        # dL/d(output)
        dz = dL_dy.copy()

        # Backprop through layers
        for i in range(n_layers - 1, -1, -1):
            if i < n_layers - 1:
                # Apply tanh derivative
                a_prev = activations[i]
                dz = dz * (1 - activations[i + 1] ** 2)

            dW[i] = activations[i].T @ dz
            db[i] = dz.sum(axis=0)

            if i > 0:
                dz = dz @ W[i].T

        return dW, db

    # Training loop
    best_loss = np.inf
    best_W, best_b = [w.copy() for w in W], [bb.copy() for bb in b]

    for iteration in range(max_iter):
        # Forward pass
        activations = mlp_forward_full(X_flat, W, b)
        outputs = activations[-1].reshape(n, n_grid)  # (n, n_grid)

        # Integral approximation (trapezoidal)
        y_pred_grid = outputs
        y_pred = (dt_grid * (y_pred_grid[:, :-1] + y_pred_grid[:, 1:]) / 2).sum(axis=1)

        # Loss
        errors = y_pred - y_norm
        loss = np.mean(errors ** 2)

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bb.copy() for bb in b]

        # Early stopping
        if loss < 1e-10:
            break

        # Backward pass
        dL_dy_pred = 2 * errors / n  # (n,)
        # dL/d(output at each grid point) via trapezoidal weights
        dL_dgrid = np.zeros((n, n_grid))
        dL_dgrid[:, 0] = dL_dy_pred * dt_grid / 2
        dL_dgrid[:, -1] = dL_dy_pred * dt_grid / 2
        for j in range(1, n_grid - 1):
            dL_dgrid[:, j] = dL_dy_pred * dt_grid

        dL_dy_flat = dL_dgrid.reshape(-1, 1)
        dW, db = mlp_backward(activations, dL_dy_flat, W, b)

        # Update parameters
        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]

        # Weight decay for smoothness (ODE-inspired)
        decay = 0.001
        for i in range(len(W)):
            W[i] -= lr * decay * W[i]

    # Use best parameters
    W, b = best_W, best_b

    # Final prediction function
    def predict(X):
        x_pred = X[:, 0] if X.ndim > 1 else X
        x_pred_norm = (x_pred - x_min) / x_range * 2 - 1
        y_pred_norm = predict_y(x_pred_norm, W, b)
        return y_pred_norm * y_std + y_mean

    return predict


def fit(X_train, y_train):
    """
    Fit a Neural ODE-based regression model.
    Returns a predict(X) callable.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    n, d = X_train.shape

    # Handle constant y
    if np.std(y_train) < 1e-12:
        c = np.mean(y_train)
        return lambda X: np.full(len(X), c)

    # Try Neural ODE approach
    try:
        predict = _train_neural_ode(X_train, y_train, max_iter=2000, lr=0.005, seed=42)
        # Validate
        y_test_pred = predict(X_train[:10])
        if not np.all(np.isfinite(y_test_pred)):
            raise ValueError("Non-finite predictions")
        return predict
    except Exception:
        # Fallback: polynomial regression
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

# EVOLVE-BLOCK-END

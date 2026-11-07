# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_basis(X, d):
    """Build a rich set of candidate basis functions evaluated on X."""
    N = X.shape[0]
    cols = [np.ones(N)]

    # Linear
    for j in range(d):
        cols.append(X[:, j])

    # Squares
    for j in range(d):
        cols.append(X[:, j] ** 2)

    # Cubics
    for j in range(d):
        cols.append(X[:, j] ** 3)

    # Quartics
    for j in range(d):
        cols.append(X[:, j] ** 4)

    # Quintics
    for j in range(d):
        cols.append(X[:, j] ** 5)

    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])

    # Triple products
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])

    # Trig functions
    for j in range(d):
        xj = X[:, j]
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        cols.append(np.sin(2 * xj))
        cols.append(np.cos(2 * xj))
        cols.append(np.tanh(xj))
        cols.append(np.tanh(2 * xj))

    # Exp and log
    for j in range(d):
        xj = X[:, j]
        cols.append(np.exp(xj))
        cols.append(np.exp(-xj))
        cols.append(np.log1p(xj))
        cols.append(np.sqrt(xj))
        cols.append(1.0 / (1.0 + np.exp(-xj)))

    # Cross terms with trig
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.cos(X[:, i]) * X[:, j])
                cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
                cols.append(X[:, i] * np.exp(X[:, j]))
                cols.append(np.exp(X[:, i]) * X[:, j])
                cols.append(X[:, i] * np.sqrt(X[:, j]))
                cols.append(np.sqrt(X[:, i]) * X[:, j])

    # Sum and product of all inputs
    cols.append(np.sum(X, axis=1))
    cols.append(np.prod(X, axis=1))
    cols.append(np.sum(X ** 2, axis=1))
    cols.append(np.sum(X ** 3, axis=1))

    # Distance from origin
    cols.append(np.linalg.norm(X, axis=1))
    cols.append(np.linalg.norm(X, axis=1) ** 2)

    # Min and max
    cols.append(np.min(X, axis=1))
    cols.append(np.max(X, axis=1))

    # Differences
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] - X[:, j])
                cols.append(X[:, i] * (X[:, i] - X[:, j]))
                cols.append((X[:, i] - X[:, j]) ** 2)

    # Additional trig combinations
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
                cols.append(np.sin(X[:, i] - X[:, j]))
                cols.append(np.cos(X[:, i] - X[:, j]))

    return np.column_stack(cols)


def _fit_lsq(F, y, lam=1e-8):
    """Fit y = F @ w using ridge regression."""
    n, p = F.shape
    if p == 0:
        return np.zeros(0)
    A = F.T @ F + lam * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return w


def _nn_forward(X, params, d, hidden_sizes):
    """Forward pass for a simple NN with tanh activations."""
    # params is a flat array: [W1, b1, W2, b2, W3, b3]
    # W1: (d, h1), b1: (h1,)
    # W2: (h1, h2), b2: (h2,)
    # W3: (h2, 1), b3: (1,)
    h1, h2 = hidden_sizes
    
    W1 = params[:d * h1].reshape(d, h1)
    b1 = params[d * h1:d * h1 + h1]
    W2 = params[d * h1 + h1:d * h1 + h1 + h1 * h2].reshape(h1, h2)
    b2 = params[d * h1 + h1 + h1 * h2:d * h1 + h1 + h1 * h2 + h2]
    W3 = params[d * h1 + h1 + h1 * h2 + h2:d * h1 + h1 + h1 * h2 + h2 + h2].reshape(h2, 1)
    b3 = params[d * h1 + h1 + h1 * h2 + h2 + h2:]
    
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = np.tanh(Z2)
    Z3 = A2 @ W3 + b3
    return Z3.flatten()


def _nn_loss(params, X, y, d, hidden_sizes):
    """MSE loss for the neural network."""
    pred = _nn_forward(X, params, d, hidden_sizes)
    return np.mean((pred - y) ** 2)


def _nn_loss_grad(params, X, y, d, hidden_sizes):
    """Compute loss and gradient for the neural network using backprop."""
    h1, h2 = hidden_sizes
    N = X.shape[0]
    
    W1 = params[:d * h1].reshape(d, h1)
    b1 = params[d * h1:d * h1 + h1]
    W2 = params[d * h1 + h1:d * h1 + h1 + h1 * h2].reshape(h1, h2)
    b2 = params[d * h1 + h1 + h1 * h2:d * h1 + h1 + h1 * h2 + h2]
    W3 = params[d * h1 + h1 + h1 * h2 + h2:d * h1 + h1 + h1 * h2 + h2 + h2].reshape(h2, 1)
    b3 = params[d * h1 + h1 + h1 * h2 + h2 + h2:]
    
    # Forward pass
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = np.tanh(Z2)
    Z3 = A2 @ W3 + b3
    pred = Z3.flatten()
    
    # Loss
    loss = np.mean((pred - y) ** 2)
    
    # Backward pass
    dZ3 = 2 * (pred - y) / N  # (N,)
    dW3 = A2.T @ dZ3.reshape(-1, 1)  # (h2, 1)
    db3 = np.sum(dZ3)  # scalar
    
    dA2 = dZ3.reshape(-1, 1) @ W3.T  # (N, h2)
    dZ2 = dA2 * (1 - A2 ** 2)  # (N, h2)
    dW2 = A1.T @ dZ2  # (h1, h2)
    db2 = np.sum(dZ2, axis=0)  # (h2,)
    
    dA1 = dZ2 @ W2.T  # (N, h1)
    dZ1 = dA1 * (1 - A1 ** 2)  # (N, h1)
    dW1 = X.T @ dZ1  # (d, h1)
    db1 = np.sum(dZ1, axis=0)  # (h1,)
    
    # Flatten gradients
    grad = np.concatenate([
        dW1.flatten(), db1,
        dW2.flatten(), db2,
        dW3.flatten(), b3 * 0 + np.array([np.sum(dZ3)])
    ])
    
    return loss, grad


def _train_nn(X, y, d, hidden_sizes=(32, 32), lr=0.01, max_iter=5000, tol=1e-10):
    """Train a small neural network using gradient descent with momentum."""
    h1, h2 = hidden_sizes
    param_count = d * h1 + h1 + h1 * h2 + h2 + h2 + 1
    
    # Initialize with small random weights
    np.random.seed(42)
    params = np.random.randn(param_count) * 0.1
    
    # Adam optimizer
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_params = params.copy()
    best_loss = _nn_loss(params, X, y, d, hidden_sizes)
    
    for t in range(1, max_iter + 1):
        if _time_left() < 5.0:
            break
        
        loss, grad = _nn_loss_grad(params, X, y, d, hidden_sizes)
        
        if not np.all(np.isfinite(grad)):
            break
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
        
        if loss < tol:
            break
    
    return best_params, best_loss


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape

    if N == 0:
        return lambda X: np.zeros(len(X))

    # Build rich basis
    F = _build_basis(X_train, d)

    # Remove non-finite columns
    finite_cols = np.all(np.isfinite(F), axis=0)
    F = F[:, finite_cols]
    p = F.shape[1]

    if p == 0:
        return lambda X: np.zeros(len(X))

    # Strategy 1: Linear regression with feature selection
    # Use OMP-like selection
    max_feat = min(p, 20)
    residual = y_train.copy()
    selected = [0]  # start with constant
    F_sel = F[:, selected]
    w = _fit_lsq(F_sel, y_train)
    pred = F_sel @ w
    residual = y_train - pred

    for _ in range(max_feat - 1):
        if _time_left() < 10.0:
            break
        corr = np.abs(F.T @ residual)
        corr[selected] = 0
        best_idx = np.argmax(corr)
        if corr[best_idx] < 1e-10:
            break
        selected.append(best_idx)
        F_sel = F[:, selected]
        w = _fit_lsq(F_sel, y_train)
        pred = F_sel @ w
        residual = y_train - pred
        if np.linalg.norm(residual) < 1e-12:
            break

    best_mse = np.mean((pred - y_train) ** 2)
    best_selected = selected
    best_w = w
    best_F = F_sel

    # Strategy 2: Neural network
    if _time_left() > 30.0:
        # Try different hidden layer sizes
        for h1, h2 in [(16, 16), (32, 32), (64, 32)]:
            if _time_left() < 20.0:
                break
            nn_params, nn_loss = _train_nn(X_train, y_train, d, (h1, h2), lr=0.01, max_iter=3000)
            if nn_loss < best_mse * 0.9:
                # Neural network is better
                best_mse = nn_loss
                best_nn_params = nn_params
                best_nn_hidden = (h1, h2)
                use_nn = True
                break
        else:
            use_nn = False
    
    if not use_nn:
        use_nn = False

    # Capture for prediction
    if use_nn:
        d_input = d
        hidden = best_nn_hidden
        params = best_nn_params

        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return _nn_forward(X, params, d_input, hidden)
    else:
        sel_indices = best_selected
        w_final = best_w

        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            F_pred = _build_basis(X, d)
            F_pred = F_pred[:, finite_cols]
            F_pred_sel = F_pred[:, sel_indices]
            return F_pred_sel @ w_final

    return predict
# EVOLVE-BLOCK-END

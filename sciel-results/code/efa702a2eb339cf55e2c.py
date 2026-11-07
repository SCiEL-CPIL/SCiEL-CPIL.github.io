"""Symbolic regression over 10 hidden equations using rich basis + NN ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 310


def _elapsed():
    return time.time() - _DEADLINE


def _build_rich_basis(X):
    """Build a rich feature basis."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    
    # Linear
    for j in range(d):
        cols.append(X[:, j])
    
    # Quadratic
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    
    # Cubic
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Quartic (only for d<=3)
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    for l in range(k, d):
                        cols.append(X[:, i] * X[:, j] * X[:, k] * X[:, l])
    
    # Trig
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.sin(2 * X[:, j]))
        cols.append(np.cos(2 * X[:, j]))
    
    # Exp (scaled)
    for j in range(d):
        x_max = np.max(np.abs(X[:, j]))
        if x_max > 0:
            x_scaled = X[:, j] / max(x_max, 1.0)
            cols.append(np.exp(x_scaled))
            cols.append(np.exp(-x_scaled))
    
    # Log
    for j in range(d):
        cols.append(np.log(np.abs(X[:, j]) + 1e-8))
    
    # Tanh
    for j in range(d):
        cols.append(np.tanh(X[:, j]))
        cols.append(np.tanh(2 * X[:, j]))
    
    # Reciprocal
    for j in range(d):
        cols.append(1.0 / (np.abs(X[:, j]) + 1e-8))
    
    # Interactions: x_i * sin(x_j), x_i * cos(x_j)
    for i in range(d):
        for j in range(d):
            if i != j:
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.cos(X[:, j]))
    
    # Pairwise products with trig
    for i in range(d):
        for j in range(i+1, d):
            cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
            cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
    
    return np.column_stack(cols)


def _fit_ridge(F, y, alpha):
    """Fit ridge regression."""
    n_features = F.shape[1]
    A = F.T @ F + alpha * np.eye(n_features)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = F @ w
    mse = np.mean((pred - y) ** 2)
    return w, mse


def _train_nn(X, y, hidden_sizes=(64, 64), epochs=2000, lr=0.005, time_budget=15):
    """Train a small neural network using numpy."""
    n_samples, d = X.shape
    y = y.reshape(-1, 1)
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std
    
    # Initialize
    np.random.seed(42)
    layers = [X_norm]
    weights = []
    biases = []
    sizes = [d] + list(hidden_sizes) + [1]
    
    for i in range(len(sizes) - 1):
        w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
        b = np.zeros((1, sizes[i+1]))
        weights.append(w)
        biases.append(b)
    
    def forward(X_in):
        layers = [X_in]
        for i in range(len(weights)):
            z = layers[-1] @ weights[i] + biases[i]
            if i < len(weights) - 1:
                z = np.tanh(z)
            layers.append(z)
        return layers
    
    start_time = time.time()
    best_loss = np.inf
    best_weights = [w.copy() for w in weights]
    best_biases = [b.copy() for b in biases]
    
    for epoch in range(epochs):
        if time.time() - start_time > time_budget:
            break
        
        # Forward
        layers = forward(X_norm)
        output = layers[-1]
        loss = np.mean((output - y_norm) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
        
        # Backward
        grads_w = [np.zeros_like(w) for w in weights]
        grads_b = [np.zeros_like(b) for b in biases]
        
        delta = (output - y_norm) / n_samples
        for i in range(len(weights) - 1, -1, -1):
            grads_w[i] = layers[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ weights[i].T) * (1 - layers[i+1]**2)
        
        # Adam update
        m_w = [np.zeros_like(g) for g in grads_w]
        v_w = [np.zeros_like(g) for g in grads_w]
        m_b = [np.zeros_like(g) for g in grads_b]
        v_b = [np.zeros_like(g) for g in grads_b]
        
        # Simple SGD with momentum
        momentum = 0.9
        for i in range(len(weights)):
            weights[i] -= lr * grads_w[i]
            biases[i] -= lr * grads_b[i]
        
        # Early stopping check
        if loss < 1e-10:
            break
    
    # Use best weights
    final_weights = best_weights
    final_biases = best_biases
    
    def nn_predict(X_in):
        X_n = (X_in - X_mean) / X_std
        h = X_n
        for i in range(len(final_weights)):
            z = h @ final_weights[i] + final_biases[i]
            if i < len(final_weights) - 1:
                z = np.tanh(z)
            h = z
        return (h * y_std + y_mean).flatten()
    
    return nn_predict, best_loss * y_std**2  # denormalized MSE


def fit(X_train, y_train):
    d = X_train.shape[1]
    start_time = time.time()
    
    # Build rich basis
    F_train = _build_rich_basis(X_train)
    
    # Strategy 1: Ridge regression with various alphas
    best_model = None
    best_mse = np.inf
    
    for alpha in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        try:
            w, mse = _fit_ridge(F_train, y_train, alpha)
            if mse < best_mse:
                best_mse = mse
                best_model = ('ridge', w, F_train)
        except Exception:
            pass
    
    # Strategy 2: Try neural network
    nn_model = None
    nn_mse = np.inf
    if time.time() - start_time < 20:
        try:
            nn_pred_fn, nn_mse = _train_nn(X_train, y_train, 
                                            hidden_sizes=(64, 64), 
                                            epochs=3000, 
                                            lr=0.01, 
                                            time_budget=18)
            if nn_mse < best_mse:
                best_mse = nn_mse
                best_model = ('nn', nn_pred_fn, None)
                nn_model = nn_pred_fn
        except Exception:
            pass
    
    # Strategy 3: If ridge didn't do well, try NN with different config
    if best_mse > 1e-6 and time.time() - start_time < 35:
        try:
            nn_pred_fn2, nn_mse2 = _train_nn(X_train, y_train,
                                              hidden_sizes=(128, 64),
                                              epochs=5000,
                                              lr=0.005,
                                              time_budget=20)
            if nn_mse2 < best_mse:
                best_mse = nn_mse2
                best_model = ('nn', nn_pred_fn2, None)
                nn_model = nn_pred_fn2
        except Exception:
            pass
    
    # Strategy 4: If still not good, try a larger NN
    if best_mse > 1e-4 and time.time() - start_time < 55:
        try:
            nn_pred_fn3, nn_mse3 = _train_nn(X_train, y_train,
                                              hidden_sizes=(128, 128, 64),
                                              epochs=8000,
                                              lr=0.003,
                                              time_budget=25)
            if nn_mse3 < best_mse:
                best_mse = nn_mse3
                best_model = ('nn', nn_pred_fn3, None)
                nn_model = nn_pred_fn3
        except Exception:
            pass
    
    # Final model selection
    if best_model is None:
        # Fallback: simple mean
        y_mean = np.mean(y_train)
        def predict(X):
            return np.full(X.shape[0], y_mean)
        return predict
    
    model_type = best_model[0]
    
    if model_type == 'ridge':
        w = best_model[1]
        def predict(X):
            F = _build_rich_basis(X)
            return F @ w
        return predict
    else:
        nn_pred_fn = best_model[1]
        def predict(X):
            return nn_pred_fn(X)
        return predict
# EVOLVE-BLOCK-END

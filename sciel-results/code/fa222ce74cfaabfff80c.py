"""Symbolic regression with Deep Feedforward Neural Network ensemble."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()


class _NN:
    """Simple feedforward neural network with tanh/relu activations."""
    
    def __init__(self, d_in, hidden_sizes, activation='tanh', seed=42):
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.seed = seed
        self.weights = []
        self.biases = []
        self._init_params()
    
    def _init_params(self):
        rng = np.random.RandomState(self.seed)
        sizes = [self.d_in] + list(self.hidden_sizes) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            # He initialization for ReLU, Xavier for tanh
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W = rng.randn(fan_in, fan_out) * scale
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def _activate(self, z, layer_idx):
        if layer_idx == len(self.hidden_sizes):  # output layer
            return z
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(z, 0)
        else:
            return np.tanh(z)
    
    def _activate_deriv(self, a, layer_idx):
        if layer_idx == len(self.hidden_sizes):
            return np.ones_like(a)
        if self.activation == 'tanh':
            return 1 - a ** 2
        elif self.activation == 'relu':
            return (a > 0).astype(float)
        else:
            return 1 - a ** 2
    
    def forward(self, X):
        """Forward pass, return activations at each layer."""
        activations = [X]
        z = X
        for i in range(len(self.weights)):
            z = z @ self.weights[i] + self.biases[i]
            a = self._activate(z, i)
            activations.append(a)
        return activations
    
    def predict(self, X):
        """Return predictions (shape (n, 1) -> (n,))."""
        acts = self.forward(X)
        return acts[-1].flatten()
    
    def train(self, X, y, lr=0.001, epochs=500, batch_size=32, 
              weight_decay=1e-5, verbose=False, time_budget=None):
        """Train with Adam optimizer."""
        n = X.shape[0]
        m = y.shape[0]
        
        # Adam state
        m_W = [np.zeros_like(W) for W in self.weights]
        v_W = [np.zeros_like(W) for W in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            if time_budget is not None and (time.time() - start_time) > time_budget:
                break
            
            # Shuffle
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end].reshape(-1, 1)
                bs = X_batch.shape[0]
                
                # Forward
                acts = self.forward(X_batch)
                pred = acts[-1]
                
                # Loss: MSE
                loss = np.mean((pred - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward
                # dL/d(pred) = 2*(pred - y)/bs
                delta = 2.0 * (pred - y_batch) / bs
                
                grads_W = [None] * len(self.weights)
                grads_b = [None] * len(self.biases)
                
                for i in range(len(self.weights) - 1, -1, -1):
                    # Gradient for weights and biases at layer i
                    grads_W[i] = acts[i].T @ delta
                    grads_b[i] = np.sum(delta, axis=0)
                    
                    # Add weight decay
                    grads_W[i] += weight_decay * self.weights[i]
                    
                    # Backpropagate delta to previous layer
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * self._activate_deriv(acts[i], i)
                
                # Adam update
                t += 1
                for i in range(len(self.weights)):
                    m_W[i] = beta1 * m_W[i] + (1 - beta1) * grads_W[i]
                    v_W[i] = beta2 * v_W[i] + (1 - beta2) * grads_W[i] ** 2
                    m_hat = m_W[i] / (1 - beta1 ** t)
                    v_hat = v_W[i] / (1 - beta2 ** t)
                    self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    
                    m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
                    v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i] ** 2
                    m_hat_b = m_b[i] / (1 - beta1 ** t)
                    v_hat_b = v_b[i] / (1 - beta2 ** t)
                    self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
            
            if verbose and epoch % 50 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"  Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return self


def _normalize_X(X):
    """Normalize features to zero mean, unit variance."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    return (X - mean) / std, mean, std


def _normalize_y(y):
    """Normalize target to zero mean, unit variance."""
    mean = y.mean()
    std = y.std()
    if std < 1e-10:
        std = 1.0
    return (y - mean) / std, mean, std


def _denormalize_y(y_norm, y_mean, y_std):
    """Denormalize target."""
    return y_norm * y_std + y_mean


def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    # Normalize inputs and target
    X_norm, X_mean, X_std = _normalize_X(X)
    y_norm, y_mean, y_std = _normalize_y(y)
    
    # Split for validation
    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_size = max(20, int(0.15 * n))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    # Define NN architectures to try
    # Various hidden layer sizes and activations
    architectures = [
        ([64, 64], 'tanh'),
        ([128, 64], 'tanh'),
        ([128, 128], 'tanh'),
        ([256, 128], 'tanh'),
        ([64, 64], 'relu'),
        ([128, 64], 'relu'),
        ([128, 128], 'relu'),
        ([256, 128], 'relu'),
    ]
    
    # Learning rates to try
    lrs = [0.001, 0.005, 0.01]
    
    best_val_mse = np.inf
    best_model = None  # (nn, X_mean, X_std, y_mean, y_std)
    
    # Time budget per architecture/lr combination
    time_per_combo = min(15.0, max(5.0, _remaining() / (len(architectures) * len(lrs))))
    
    for hidden_sizes, activation in architectures:
        if _remaining() < 10.0:
            break
        for lr in lrs:
            if _remaining() < 5.0:
                break
            try:
                # Try multiple seeds for each config
                for seed in [42, 123, 456]:
                    if _remaining() < 3.0:
                        break
                    nn = _NN(d, hidden_sizes, activation, seed=seed)
                    nn.train(X_tr, y_tr, lr=lr, epochs=300, batch_size=32,
                            weight_decay=1e-5, time_budget=time_per_combo)
                    
                    pred_val = nn.predict(X_val)
                    mse = np.mean((pred_val - y_val) ** 2)
                    
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_model = (nn, X_mean, X_std, y_mean, y_std)
                        
            except Exception as e:
                continue
    
    # If no good model found, try a simpler linear fallback
    if best_model is None or _remaining() < 5.0:
        try:
            # Simple linear regression with bias
            X_tr_b = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
            X_val_b = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
            w_lin = np.linalg.lstsq(X_tr_b, y_tr, rcond=None)[0]
            pred_val = X_val_b @ w_lin
            mse = np.mean((pred_val - y_val) ** 2)
            if mse < best_val_mse:
                best_val_mse = mse
                # Store as a simple linear model
                best_model = ('linear', w_lin, X_mean, X_std, y_mean, y_std)
        except Exception:
            pass
    
    # Final fit on ALL data with the best configuration
    if best_model is not None:
        if isinstance(best_model[0], _NN):
            nn, X_mean_f, X_std_f, y_mean_f, y_std_f = best_model
            # Refit on all data
            try:
                nn.train(X_norm, y_norm, lr=0.001, epochs=200, batch_size=32,
                        weight_decay=1e-5, time_budget=min(20.0, _remaining()))
            except Exception:
                pass
            
            def predict(X_test):
                X_t = X_test.astype(np.float64)
                X_t_norm = (X_t - X_mean_f) / X_std_f
                try:
                    y_pred_norm = nn.predict(X_t_norm)
                except Exception:
                    y_pred_norm = np.full(X_t.shape[0], y_mean_f)
                return _denormalize_y(y_pred_norm, y_mean_f, y_std_f)
        else:
            # Linear model
            _, w_lin, X_mean_f, X_std_f, y_mean_f, y_std_f = best_model
            def predict(X_test):
                X_t = X_test.astype(np.float64)
                X_t_norm = (X_t - X_mean_f) / X_std_f
                X_t_b = np.hstack([X_t_norm, np.ones((X_t.shape[0], 1))])
                y_pred_norm = X_t_b @ w_lin
                return _denormalize_y(y_pred_norm, y_mean_f, y_std_f)
    else:
        # Ultimate fallback: predict mean
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
    
    return predict
# EVOLVE-BLOCK-END

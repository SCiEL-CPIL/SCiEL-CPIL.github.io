# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import qr

_START_TIME = None
_basis_cache = {}

def _build_basis(x, y):
    """
    Constructs a diverse set of candidate features and orthogonalizes them
    to form a basis of width 10.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Use a robust key for caching
    # Since x and y are fixed during the optimization phase, we can cache based on shape and a hash of the data
    # However, hashing large arrays is expensive. 
    # A safer bet is to assume the evaluator reuses the same arrays (same id) or same content.
    # Let's use id(x) and id(y) as a fast proxy, but verify shape.
    # If id changes, we rebuild.
    key = (id(x), id(y), x.shape, y.shape)
    
    if key in _basis_cache:
        return _basis_cache[key]
        
    n = x.shape[0]
    if n == 0:
        return np.ones((0, 10))
        
    # Candidate features
    # 1. Polynomials up to degree 4
    # 2. Trig functions
    # 3. Sigmoid-like
    # 4. Interactions
    
    feats = []
    
    # Constant
    feats.append(np.ones(n))
    
    # Linear
    feats.append(x)
    feats.append(y)
    
    # Quadratic
    feats.append(x * x)
    feats.append(y * y)
    feats.append(x * y)
    
    # Cubic
    feats.append(x * x * x)
    feats.append(y * y * y)
    feats.append(x * x * y)
    feats.append(x * y * y)
    
    # Quartic
    feats.append(x**4)
    feats.append(y**4)
    feats.append(x**3 * y)
    feats.append(x * y**3)
    feats.append(x**2 * y**2)
    
    # Trig
    feats.append(np.sin(x))
    feats.append(np.cos(x))
    feats.append(np.sin(y))
    feats.append(np.cos(y))
    feats.append(np.sin(x + y))
    feats.append(np.cos(x + y))
    feats.append(np.sin(x - y))
    feats.append(np.cos(x - y))
    feats.append(np.sin(2*x))
    feats.append(np.sin(2*y))
    feats.append(np.cos(2*x))
    feats.append(np.cos(2*y))
    
    # Sigmoid / Tanh
    feats.append(np.tanh(x))
    feats.append(np.tanh(y))
    feats.append(np.tanh(x + y))
    feats.append(np.tanh(x - y))
    
    # Exponential (clipped to avoid overflow/underflow issues in QR)
    # exp(x) is bounded in [1/e, e] for x in [-1,1], so it's safe
    feats.append(np.exp(x))
    feats.append(np.exp(y))
    
    # Normalize features to have zero mean and unit variance to help QR
    # This ensures that the basis vectors are on similar scales
    X = np.column_stack(feats)
    
    # Handle any non-finite values just in case
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Center and Scale
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std
    
    # QR Decomposition to get orthogonal basis
    # We want the first 10 columns of Q
    # mode='economic' gives Q of shape (n, min(n, k))
    # If k > n, Q is (n, n). We take first 10.
    Q, R = qr(X_norm, mode='economic')
    
    # Ensure we have at least 10 columns
    # If n < 10, Q has n columns. We pad with zeros? 
    # The problem states 200 training points, so n=200 > 10.
    # But for safety:
    if Q.shape[1] < 10:
        # Pad with zeros (which effectively means those params will be 0 or unused)
        # Actually, better to pad with random orthogonal vectors? 
        # But if n<10, we can't fit 10 independent features anyway.
        # For n=200, this won't happen.
        pad_width = 10 - Q.shape[1]
        Q = np.hstack([Q, np.zeros((n, pad_width))])
        
    basis = Q[:, :10]
    
    _basis_cache[key] = basis
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Check time limit
    # The evaluator kills at 150s. We should stop optimizing if we are close.
    # However, equation is called *during* the optimization.
    # We can't stop the optimizer from inside equation easily unless we raise an exception or return bad values.
    # Returning bad values (1e30) might cause the optimizer to crash or waste time.
    # The prompt says: "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    # If we return a constant, it's finite.
    # Let's just ensure the function is fast.
    # The basis construction is cached. The matrix multiplication is O(n*10).
    # n=200, 10 params. This is extremely fast.
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if params.shape[0] != 10:
        if params.shape[0] < 10:
            params = np.pad(params, (0, 10 - params.shape[0]))
        else:
            params = params[:10]
            
    # Get the orthogonal basis
    basis = _build_basis(x, y)
    
    # Prediction is a linear combination of the basis functions
    # pred = basis @ params
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        # Fallback to mean of y if something goes wrong
        return np.full_like(y, np.mean(y), dtype=float)
        
    return pred
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute the value arrays for the four slopes
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# For each slope, precompute the bin index for each cell
# D: x - y, range -9..9 -> shift by 9 to get 0..18
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
# H0: x, range 0..9
_h0_vals = _x_flat.astype(np.int64)
# H1: x + y, range 0..18
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
# H2: x + 2y, range 0..27
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

# Number of bins for each
_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

# Precompute pooling matrices (100 x k) for each slope
# M_D: row i, col j is 1 if cell i maps to bin j in x-y
M_D = np.zeros((_N * _N, _N_D), dtype=np.float64)
M_D[np.arange(_N * _N), _d_vals] = 1.0

M_H0 = np.zeros((_N * _N, _N_H0), dtype=np.float64)
M_H0[np.arange(_N * _N), _h0_vals] = 1.0

M_H1 = np.zeros((_N * _N, _N_H1), dtype=np.float64)
M_H1[np.arange(_N * _N), _h1_vals] = 1.0

M_H2 = np.zeros((_N * _N, _N_H2), dtype=np.float64)
M_H2[np.arange(_N * _N), _h2_vals] = 1.0


def _entropy_from_probs(probs):
    """Compute Shannon entropy in bits from a probability vector."""
    p = probs[probs > 1e-15]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))


def _score_fast(w):
    """
    Fast score computation using precomputed pooling matrices.
    w: flat array of shape (100,) with non-negative entries summing to 1.
    """
    d_probs = M_D @ w
    h0_probs = M_H0 @ w
    h1_probs = M_H1 @ w
    h2_probs = M_H2 @ w
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def _score_batch(W):
    """
    Score a batch of weight vectors.
    W: shape (batch, 100), each row sums to 1.
    Returns: shape (batch,) array of scores.
    """
    batch_size = W.shape[0]
    d_probs = W @ M_D.T  # (batch, _N_D)
    h0_probs = W @ M_H0.T  # (batch, _N_H0)
    h1_probs = W @ M_H1.T  # (batch, _N_H1)
    h2_probs = W @ M_H2.T  # (batch, _N_H2)
    
    # Compute entropies vectorized
    def _entropy_batch(probs):
        p = np.where(probs > 1e-15, probs, 1.0)  # avoid log(0)
        log_p = np.log2(p)
        p_safe = np.where(probs > 1e-15, probs, 0.0)
        return -np.sum(p_safe * log_p, axis=1)
    
    D = _entropy_batch(d_probs)
    H0 = _entropy_batch(h0_probs)
    H1 = _entropy_batch(h1_probs)
    H2 = _entropy_batch(h2_probs)
    
    M = np.maximum(np.maximum(H0, H1), H2)
    scores = np.where(M > 1e-9, D / np.maximum(M, 1e-9), 0.0)
    return scores


def _gradient_score(w):
    """
    Compute gradient of the score with respect to w (unconstrained).
    Uses finite differences on the batch scorer for robustness.
    """
    eps = 1e-6
    grad = np.zeros_like(w)
    base_score = _score_fast(w)
    
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += eps
        w_plus = np.maximum(w_plus, 0)
        s = w_plus.sum()
        if s > 0:
            w_plus = w_plus / s
        
        w_minus = w.copy()
        w_minus[i] -= eps
        w_minus = np.maximum(w_minus, 0)
        s = w_minus.sum()
        if s > 0:
            w_minus = w_minus / s
        else:
            w_minus = w.copy()
        
        score_plus = _score_fast(w_plus)
        score_minus = _score_fast(w_minus)
        grad[i] = (score_plus - score_minus) / (2 * eps)
    
    return grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    safety = 30.0
    
    # Start with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_score = _score_fast(P.flatten())
    
    np.random.seed(42)
    
    def _normalize(P_mat):
        P_mat = np.clip(P_mat, 0, None)
        s = P_mat.sum()
        if s <= 1e-9:
            return None
        return P_mat / s
    
    # Phase 1: Batch random search with local refinement
    # Generate many candidates in batches, evaluate, keep top, refine
    
    batch_size = 2000
    num_rounds = 50
    
    # Initial diverse candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones((n, n)) / (n * n))
    
    # Concentrated on single cells
    for i in range(n):
        for j in range(n):
            c = np.zeros((n, n))
            c[i, j] = 1.0
            candidates.append(c)
    
    # Diagonals
    for d in range(-(n-1), n):
        c = np.zeros((n, n))
        for xi in range(n):
            yi = xi - d
            if 0 <= yi < n:
                c[yi, xi] = 1.0
        s = c.sum()
        if s > 0:
            candidates.append(c / s)
    
    # Anti-diagonals
    for d in range(0, 2*n):
        c = np.zeros((n, n))
        for xi in range(n):
            yi = d - xi
            if 0 <= yi < n:
                c[yi, xi] = 1.0
        s = c.sum()
        if s > 0:
            candidates.append(c / s)
    
    # Random sparse distributions
    for _ in range(50):
        c = np.random.dirichlet(np.ones(n * n) * 0.5)
        candidates.append(c.reshape(n, n))
    
    # Random structured: concentrate on few x values
    for _ in range(50):
        num_x = np.random.randint(1, 4)
        x_vals = np.random.choice(n, num_x, replace=False)
        c = np.zeros((n, n))
        for xi in x_vals:
            c[:, xi] = 1.0
        s = c.sum()
        if s > 0:
            c = c / s
            # Add some y-structure
            y_weights = np.random.dirichlet(np.ones(n) * 2)
            for xi in x_vals:
                c[:, xi] *= y_weights
            s = c.sum()
            if s > 0:
                c = c / s
            candidates.append(c)
    
    # Evaluate initial candidates in batches
    flat_candidates = np.array([c.flatten() for c in candidates])
    for start in range(0, len(flat_candidates), batch_size):
        end = min(start + batch_size, len(flat_candidates))
        batch = flat_candidates[start:end]
        scores = _score_batch(batch)
        for i, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_P = batch[i].reshape(n, n)
    
    # Iterative refinement: take top candidates, perturb, re-evaluate
    top_candidates = [best_P.flatten().copy()]
    
    # Add some random perturbations of best
    for _ in range(100):
        pert = best_P.flatten() + np.random.normal(0, 0.1, n * n)
        pert = np.maximum(pert, 0)
        s = pert.sum()
        if s > 0:
            pert = pert / s
            top_candidates.append(pert)
    
    for round_idx in range(num_rounds):
        if time.time() > deadline - safety:
            break
        
        # Generate new batch from top candidates + random
        batch_list = []
        
        # Perturb top candidates
        for c in top_candidates[:50]:
            for _ in range(3):
                pert = c + np.random.normal(0, 0.05, n * n)
                pert = np.maximum(pert, 0)
                s = pert.sum()
                if s > 0:
                    pert = pert / s
                    batch_list.append(pert)
        
        # Random new candidates
        for _ in range(batch_size - len(batch_list)):
            c = np.random.dirichlet(np.ones(n * n) * np.random.uniform(0.1, 5.0))
            batch_list.append(c)
        
        batch = np.array(batch_list[:batch_size])
        scores = _score_batch(batch)
        
        # Update best
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_P = batch[max_idx].reshape(n, n)
        
        # Keep top candidates for next round
        top_indices = np.argsort(scores)[::-1][:50]
        top_candidates = [batch[i].copy() for i in top_indices]
        
        # Also keep the overall best
        top_candidates.insert(0, best_P.flatten().copy())
    
    # Phase 2: Gradient ascent on the best candidate
    if time.time() < deadline - safety:
        w = best_P.flatten().copy()
        lr = 0.1
        for step in range(2000):
            if time.time() > deadline - safety:
                break
            
            grad = _gradient_score(w)
            
            # Gradient step
            w_new = w + lr * grad
            w_new = np.maximum(w_new, 0)
            s = w_new.sum()
            if s > 1e-9:
                w_new = w_new / s
            
            new_score = _score_fast(w_new)
            if new_score > best_score:
                best_score = new_score
                best_P = w_new.reshape(n, n)
                w = w_new.copy()
            else:
                # Reduce learning rate
                lr *= 0.95
                if lr < 1e-6:
                    break
    
    # Phase 3: Final polish with scipy L-BFGS-B on log-softmax parameterization
    if time.time() < deadline - safety:
        def _log_softmax_to_prob(params):
            max_p = np.max(params)
            exp_p = np.exp(params - max_p)
            return exp_p / np.sum(exp_p)
        
        def _neg_score(params):
            probs = _log_softmax_to_prob(params)
            return -_score_fast(probs)
        
        # Initialize params from current best
        w_best = best_P.flatten()
        w_best = np.maximum(w_best, 1e-10)
        params_init = np.log(w_best)
        
        try:
            result = minimize(
                _neg_score,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < -best_score + 1e-12:
                new_w = _log_softmax_to_prob(result.x)
                new_score = _score_fast(new_w)
                if new_score > best_score:
                    best_score = new_score
                    best_P = new_w.reshape(n, n)
        except Exception:
            pass
    
    # Final validation and normalization
    P = np.maximum(best_P, 0.0)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s
    
    return P


# EVOLVE-BLOCK-END

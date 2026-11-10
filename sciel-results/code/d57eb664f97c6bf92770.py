# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds

    # Precompute projection indices
    # x in 0..9, y in 0..9
    x_vals = np.arange(n)
    y_vals = np.arange(n)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    X = X.flatten()  # length 100
    Y = Y.flatten()  # length 100

    # Projection values
    v_inf = X - Y       # x - y, range -9..9
    v_0 = X             # x, range 0..9
    v_1 = X + Y         # x + y, range 0..18
    v_2 = X + 2*Y       # x + 2y, range 0..27

    # Map to indices
    idx_inf = v_inf + 9  # 0..18
    idx_0 = v_0          # 0..9
    idx_1 = v_1          # 0..18
    idx_2 = v_2          # 0..27

    def compute_entropy(v_indices, p_flat):
        """Compute Shannon entropy of projection defined by v_indices."""
        # p_flat is length 100
        q = np.bincount(v_indices, weights=p_flat, minlength=np.max(v_indices)+1)
        q = q / q.sum()
        q = q[q > 1e-15]
        return -np.sum(q * np.log2(q))

    def entropy_vector(v_indices, p_flat):
        """Return the projected distribution (normalized)."""
        q = np.bincount(v_indices, weights=p_flat, minlength=np.max(v_indices)+1)
        q = q / q.sum()
        return q

    def compute_gradient(v_indices, p_flat):
        """
        Gradient of H(q) w.r.t. p (unnormalized weights).
        H(q) = -sum_q q log2 q, where q = bincount(v, p)/sum(p)
        dq_i/dp_j = (delta_{v_j, i} - q_i) / sum(p)
        dH/dq_i = -(log2(q_i) + 1/ln2)
        dH/dp_j = -sum_i (dH/dq_i) (dq_i/dp_j)
                 = -sum_i (-(log2(q_i) + 1/ln2)) * (delta_{v_j, i} - q_i)/S
                 = (1/S) * sum_i (log2(q_i) + 1/ln2) * (delta_{v_j, i} - q_i)
                 = (1/S) * [ (log2(q_{v_j}) + 1/ln2) - sum_i q_i (log2(q_i) + 1/ln2) ]
                 = (1/S) * [ log2(q_{v_j}) + 1/ln2 + H(q) ]
        where S = sum(p)
        """
        q = np.bincount(v_indices, weights=p_flat, minlength=np.max(v_indices)+1)
        S = q.sum()
        q = q / S
        q_safe = np.where(q > 1e-15, q, 1e-15)
        H = -np.sum(q * np.log2(q_safe))
        
        # For each cell j, the gradient component is:
        # (1/S) * [ log2(q_{v_j}) + 1/ln2 + H ]
        # where q_{v_j} is the probability mass at bin v_j
        
        # Map each cell to its bin
        bin_indices = v_indices
        q_per_cell = q[bin_indices]
        
        grad = (np.log2(np.where(q_per_cell > 1e-15, q_per_cell, 1e-15)) + 1/np.log(2) + H) / S
        return grad

    def objective_and_grad(p_flat):
        """
        Compute the objective R = D / M and its gradient.
        Uses smooth max approximation for M.
        """
        D = compute_entropy(idx_inf, p_flat)
        H0 = compute_entropy(idx_0, p_flat)
        H1 = compute_entropy(idx_1, p_flat)
        H2 = compute_entropy(idx_2, p_flat)
        
        # Soft max with beta
        beta = 10.0
        H_vals = np.array([H0, H1, H2])
        M_soft = np.logaddexp.reduce(beta * H_vals) / beta  # soft max
        M = np.max(H_vals)
        
        if M < 1e-9:
            return 0.0, None
        
        R = D / M_soft  # Use soft max for smoothness
        
        # Gradient of R = D/M_soft
        # dR/dp = (M_soft * dD/dp - D * dM_soft/dp) / M_soft^2
        
        gD = compute_gradient(idx_inf, p_flat)
        gH0 = compute_gradient(idx_0, p_flat)
        gH1 = compute_gradient(idx_1, p_flat)
        gH2 = compute_gradient(idx_2, p_flat)
        
        # dM_soft/dp = (1/beta) * d/dp [log(sum exp(beta H_i))]
        # = (1/beta) * sum_i exp(beta H_i) / sum_j exp(beta H_j) * dH_i/dp
        # = sum_i (exp(beta H_i) / sum_j exp(beta H_j)) * dH_i/dp
        
        exp_vals = np.exp(beta * H_vals)
        weights = exp_vals / exp_vals.sum()
        
        gM = (weights[0] * gH0 + weights[1] * gH1 + weights[2] * gH2)
        
        gR = (M_soft * gD - D * gM) / (M_soft * M_soft)
        
        # Also add gradient for the actual M (hard max) as a blend
        # Find which H is max
        max_idx = np.argmax(H_vals)
        if max_idx == 0:
            gM_hard = gH0
        elif max_idx == 1:
            gM_hard = gH1
        else:
            gM_hard = gH2
        
        # Blend soft and hard gradients
        alpha_blend = 0.5
        gM_blend = alpha_blend * gM + (1 - alpha_blend) * gM_hard
        gR_blend = (M * gD - D * gM_blend) / (M * M)
        
        return R, gR_blend

    def normalize(p_flat):
        """Normalize p_flat to sum to 1, clip negatives."""
        p = np.clip(p_flat, 0, None)
        s = p.sum()
        if s < 1e-15:
            return np.ones(100) / 100.0
        return p / s

    def reshape_to_matrix(p_flat):
        """Reshape flat array to 10x10 matrix with P[y, x]."""
        # The flat array is indexed as [y * 10 + x] since meshgrid with indexing='xy'
        # X[i, j] = x_j, Y[i, j] = y_i
        # So flatten gives [X[0,0], X[0,1], ..., X[0,9], X[1,0], ...]
        # Which corresponds to [x=0,y=0; x=1,y=0; ...; x=9,y=0; x=0,y=1; ...]
        # We need P[y, x]
        P = p_flat.reshape((n, n))
        return P

    # Initial guess: uniform
    p0 = np.ones(100) / 100.0
    best_P = reshape_to_matrix(p0)
    best_score = -np.inf

    # Evaluate initial
    R0, _ = objective_and_grad(p0)
    best_score = R0

    # Try gradient ascent
    max_iter = 5000
    lr = 0.1
    current_p = p0.copy()
    prev_score = -np.inf
    no_improve_count = 0

    for i in range(max_iter):
        if time.time() > deadline - 30:
            break

        score, grad = objective_and_grad(current_p)
        
        if grad is None:
            break

        if score > best_score:
            best_score = score
            best_P = reshape_to_matrix(normalize(current_p))
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count > 100:
            # Try reducing learning rate
            lr *= 0.5
            no_improve_count = 0
            if lr < 1e-6:
                break

        # Gradient ascent step
        current_p = current_p + lr * grad
        
        # Normalize and clip
        current_p = normalize(current_p)
        
        # Add small noise to escape plateaus
        if no_improve_count > 20:
            noise = np.random.normal(0, 0.01 * lr, size=100)
            current_p = current_p + noise
            current_p = normalize(current_p)

    # Try scipy minimize on negative log-likelihood style
    if time.time() < deadline - 60:
        # Use a different parameterization: log-sum-exp
        # Optimize over unconstrained parameters u, then p = exp(u) / sum(exp(u))
        def log_param_objective(u):
            p = np.exp(u - np.max(u))  # shift for stability
            p = p / p.sum()
            score, _ = objective_and_grad(p)
            return -score  # minimize negative
        
        # Initial u from current best
        p_best = best_P.flatten()
        p_best = np.where(p_best > 1e-10, p_best, 1e-10)
        p_best = p_best / p_best.sum()
        u0 = np.log(p_best)
        
        try:
            result = minimize(log_param_objective, u0, method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6})
            if result.fun < -best_score:
                p_opt = np.exp(result.x - np.max(result.x))
                p_opt = p_opt / p_opt.sum()
                R_opt, _ = objective_and_grad(p_opt)
                if R_opt > best_score:
                    best_score = R_opt
                    best_P = reshape_to_matrix(p_opt)
        except Exception:
            pass

    # Final polish: local search on the best solution
    if time.time() < deadline - 10:
        p_cur = best_P.flatten().copy()
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            # Random perturbation
            idx = np.random.randint(0, 100)
            delta = np.random.uniform(-0.01, 0.01)
            p_new = p_cur.copy()
            p_new[idx] += delta
            p_new = normalize(p_new)
            R_new, _ = objective_and_grad(p_new)
            if R_new > best_score:
                best_score = R_new
                best_P = reshape_to_matrix(p_new)
                p_cur = p_new

    # Ensure validity
    P = best_P
    P = np.clip(P, 0, None)
    P_sum = P.sum()
    if P_sum < 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / P_sum

    return P


# EVOLVE-BLOCK-END

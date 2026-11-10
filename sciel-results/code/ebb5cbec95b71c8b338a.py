# EVOLVE-BLOCK-START
import numpy as np


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H_{-1} (entropy of x-y) and M = max(H_0, H_1, H_2).
    Uses simulated annealing with vectorized entropy evaluation via precomputed pooling matrices.
    """
    n = 10
    size = n * n  # 100

    # Precompute pooling matrices for slopes -1, 0, 1, 2
    def build_pooling_matrix(r):
        """Build pooling matrix A such that A @ p_flat gives marginal distribution of x + r*y."""
        if r == -1:
            min_val = -9
            max_val = 9
        elif r == 0:
            min_val = 0
            max_val = 9
        elif r == 1:
            min_val = 0
            max_val = 18
        elif r == 2:
            min_val = 0
            max_val = 27
        else:
            min_val = 0
            max_val = 9 + abs(r) * 9

        num_bins = int(max_val - min_val + 1)
        A = np.zeros((num_bins, size), dtype=np.float64)

        for y in range(n):
            for x in range(n):
                val = x + r * y
                idx = y * n + x
                bin_idx = int(val - min_val)
                if 0 <= bin_idx < num_bins:
                    A[bin_idx, idx] = 1.0
        return A

    A_minus1 = build_pooling_matrix(-1)
    A_0 = build_pooling_matrix(0)
    A_1 = build_pooling_matrix(1)
    A_2 = build_pooling_matrix(2)

    def shannon_entropy(p):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        p = np.asarray(p)
        mask = p > 1e-15
        if not np.any(mask):
            return 0.0
        p_m = p[mask]
        return float(-np.sum(p_m * np.log2(p_m)))

    def evaluate(p_vec):
        """
        Given a probability vector p (length 100), compute value = D/M.
        """
        p_vec = np.asarray(p_vec)
        # Clip to valid range to avoid log(0) issues, though normalization handles sum
        p_vec = np.maximum(p_vec, 1e-15)
        p_vec = p_vec / np.sum(p_vec)

        # Compute marginals using precomputed matrices
        m_minus1 = A_minus1 @ p_vec
        m_0 = A_0 @ p_vec
        m_1 = A_1 @ p_vec
        m_2 = A_2 @ p_vec

        D = shannon_entropy(m_minus1)
        H0 = shannon_entropy(m_0)
        H1 = shannon_entropy(m_1)
        H2 = shannon_entropy(m_2)
        M = max(H0, H1, H2)

        if M < 1e-9:
            return -1e10  # Invalid, return very low score

        return D / M

    # Initial guess: uniform distribution
    p_current = np.ones(size) / size
    best_p = p_current.copy()
    best_val = evaluate(p_current)

    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    num_iterations = 20000
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    # Use a random generator for reproducibility if needed, but default is fine
    rng = np.random.default_rng(42)

    for i in range(num_iterations):
        T = T_start * (cooling_rate ** i)
        
        # Generate a perturbation
        # We perturb the log-probabilities or directly the probabilities?
        # Directly perturbing probabilities and renormalizing is simpler and keeps them positive.
        # We add Gaussian noise to the current distribution.
        noise = rng.normal(0, 0.1, size)
        p_candidate = p_current + noise
        
        # Ensure positivity
        p_candidate = np.maximum(p_candidate, 1e-15)
        p_candidate = p_candidate / np.sum(p_candidate)
        
        val_candidate = evaluate(p_candidate)
        
        if val_candidate > best_val:
            best_val = val_candidate
            best_p = p_candidate.copy()
            
        # Acceptance criterion
        if val_candidate > best_val:
            p_current = p_candidate
        else:
            # Accept with probability exp((val_candidate - val_current) / T)
            # We need val_current to compare. We store it or recompute? 
            # Recomputing is cheap enough, but storing is better.
            # Let's store current value.
            if i == 0:
                val_current = best_val
            # Actually, we should track val_current separately from best_val.
            # Let's fix the logic:
            pass # The pass above is a placeholder, let's restructure the loop properly.

    # Restructure Simulated Annealing for clarity and correctness
    p_current = np.ones(size) / size
    val_current = evaluate(p_current)
    best_p = p_current.copy()
    best_val = val_current

    for i in range(num_iterations):
        T = T_start * (cooling_rate ** i)
        
        # Generate perturbation
        noise = rng.normal(0, 0.05, size)
        p_candidate = p_current + noise
        p_candidate = np.maximum(p_candidate, 1e-15)
        p_candidate = p_candidate / np.sum(p_candidate)
        
        val_candidate = evaluate(p_candidate)
        
        if val_candidate > best_val:
            best_val = val_candidate
            best_p = p_candidate.copy()
            
        # Acceptance
        delta = val_candidate - val_current
        if delta > 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(delta / T)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            p_current = p_candidate
            val_current = val_candidate

    # Final normalization and reshaping
    P = best_p.reshape((n, n))
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / size

    return P


# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Fixed-point iteration with soft-min regularization to maximize minimum autocorrelation.
    
    We want to maximize min_s sum_i h[i]*h[(i+s)%N] subject to sum(h)=N/2, 0<=h[i]<=1.
    
    Using soft-min: min_s x_s ≈ -logsumexp(-x_s/beta) * beta for small beta
    Or more practically: soft_min = -logsumexp(-x/beta)*beta where beta controls sharpness.
    
    The optimality condition suggests h should be a fixed point of a projection operator.
    We iterate:
      1. Compute autocorrelations ac[s] = sum_i h[i]*h[(i+s)%N] for all s
      2. Compute soft-min value m = soft_min(ac)
      3. Compute gradient of -soft_min w.r.t. h
      4. Update h in direction that increases min autocorrelation
      5. Project onto constraint set: clip to [0,1], then adjust sum to N/2
    
    The soft-min regularization makes the objective smooth, enabling gradient-based updates.
    """
    N_local = N
    target_sum = N_local / 2.0
    
    # Initialize with uniform 0.5
    h = np.full(N_local, 0.5)
    
    # Parameters
    beta = 0.1  # Soft-min temperature (smaller = sharper min approximation)
    lr = 0.05   # Learning rate
    n_iter = 5000
    
    # Precompute shift indices for efficiency
    shift_indices = [np.arange(N_local) + s for s in range(N_local)]
    shift_indices = [idx % N_local for idx in shift_indices]
    
    def compute_autocorrs(h_vec):
        """Compute all periodic autocorrelations."""
        acs = np.zeros(N_local)
        for s in range(N_local):
            acs[s] = np.dot(h_vec, h_vec[shift_indices[s]])
        return acs
    
    def soft_min(values, beta_val):
        """Compute soft minimum using log-sum-exp trick."""
        # For numerical stability, use the identity: min(x) = -logsumexp(-x/beta)*beta
        neg_scaled = -values / beta_val
        # Subtract max for numerical stability
        neg_scaled_max = np.max(neg_scaled)
        logsumexp_val = neg_scaled_max + np.log(np.sum(np.exp(neg_scaled - neg_scaled_max)))
        return -logsumexp_val * beta_val
    
    def soft_min_grad(values, beta_val):
        """Gradient of soft_min w.r.t. values."""
        neg_scaled = -values / beta_val
        neg_scaled_max = np.max(neg_scaled)
        exp_vals = np.exp(neg_scaled - neg_scaled_max)
        probs = exp_vals / np.sum(exp_vals)
        # d/dvalues soft_min = -probs / beta * (-1) ... let me derive properly
        # soft_min = -beta * logsumexp(-values/beta)
        # d/dvalues_j soft_min = -beta * (1/logsumexp) * d/dvalues_j logsumexp(-values/beta)
        # = -beta * (1/sum(exp(-v/beta))) * exp(-v_j/beta) * (-1/beta)
        # = exp(-v_j/beta) / sum(exp(-v_k/beta))
        # = probs[j]
        return probs
    
    # Iteration
    for it in range(n_iter):
        # Compute autocorrelations
        acs = compute_autocorrs(h)
        
        # Compute soft-min and its gradient w.r.t. acs
        sm_val = soft_min(acs, beta)
        sm_grad_acs = soft_min_grad(acs, beta)
        
        # Now we need d(sm_val)/dh = sum_s sm_grad_acs[s] * d(acs[s])/dh
        # acs[s] = sum_i h[i] * h[(i+s)%N]
        # d(acs[s])/dh[j] = h[(j+s)%N] + h[(j-s)%N] (for j in 0..N-1)
        # So d(sm_val)/dh[j] = sum_s sm_grad_acs[s] * (h[(j+s)%N] + h[(j-s)%N])
        
        grad_h = np.zeros(N_local)
        for s in range(N_local):
            g = sm_grad_acs[s]
            if g > 1e-10:
                # h[(j+s)%N] for all j
                term1 = np.roll(h, -s)  # h[(j+s)%N] when indexed by j
                term2 = np.roll(h, s)   # h[(j-s)%N] when indexed by j
                grad_h += g * (term1 + term2)
        
        # We want to maximize soft_min, so update in direction of grad_h
        # But we need to respect constraints
        h_new = h + lr * grad_h
        
        # Project onto box [0,1]
        h_new = np.clip(h_new, 0, 1)
        
        # Project onto sum constraint: adjust to make sum = target_sum
        current_sum = np.sum(h_new)
        if abs(current_sum - target_sum) > 1e-15:
            diff = target_sum - current_sum
            # Distribute difference to elements not at bounds
            if diff > 0:
                idx = np.where(h_new < 1)[0]
                if len(idx) > 0:
                    h_new[idx] += diff / len(idx)
            else:
                idx = np.where(h_new > 0)[0]
                if len(idx) > 0:
                    h_new[idx] += diff / len(idx)
            h_new = np.clip(h_new, 0, 1)
        
        # Final sum correction with iterative adjustment
        for _ in range(50):
            cs = np.sum(h_new)
            if abs(cs - target_sum) < 1e-12:
                break
            d = target_sum - cs
            if d > 0:
                idx = np.where(h_new < 1)[0]
                if len(idx) == 0:
                    break
                h_new[idx] += d / len(idx)
            else:
                idx = np.where(h_new > 0)[0]
                if len(idx) == 0:
                    break
                h_new[idx] += d / len(idx)
            h_new = np.clip(h_new, 0, 1)
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < 1e-10:
            h = h_new
            break
        
        h = h_new
    
    # Final exact sum enforcement
    final_sum = np.sum(h)
    if abs(final_sum - target_sum) > 1e-10:
        diff = target_sum - final_sum
        for _ in range(100):
            cs = np.sum(h)
            if abs(cs - target_sum) < 1e-12:
                break
            d = target_sum - cs
            if d > 0:
                idx = np.where(h < 1)[0]
                if len(idx) == 0:
                    break
                h[idx] += d / len(idx)
            else:
                idx = np.where(h > 0)[0]
                if len(idx) == 0:
                    break
                h[idx] += d / len(idx)
            h = np.clip(h, 0, 1)
    
    return h
# EVOLVE-BLOCK-END

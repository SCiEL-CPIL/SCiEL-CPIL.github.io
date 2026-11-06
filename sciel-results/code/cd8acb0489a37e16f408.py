import numpy as np
import time

def _evaluate(coeffs):
    """
    Evaluate the sup-norm of P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden-section refinement.
    """
    # Dense sampling grid
    M = 2048
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    # Compute P(z) for all z
    # coeffs shape (24,), z shape (M,)
    # We want sum_k coeffs[k] * z^k
    # Using broadcasting: coeffs[:, None] * (z[None, :] ** np.arange(24)[:, None])
    # This creates a (24, M) matrix.
    powers = np.arange(24)[:, None]
    z_powers = z[None, :] ** powers
    P = np.sum(coeffs[:, None] * z_powers, axis=0)
    mags = np.abs(P)
    
    # Find top peaks to refine
    # We look for local maxima in the sampled mags
    # Simple approach: find indices where mags are among the top K global max
    # Since M is small, we can just find the top K indices
    top_k = 5
    idx = np.argsort(mags)[-top_k:]
    
    max_val = 0.0
    
    # Refine each peak using golden section search
    # The interval for refinement is the width of one grid step
    delta = 2 * np.pi / M
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    inv_phi2 = 1 / (phi * phi)
    
    for i in idx:
        # Center of the interval
        center = theta[i]
        # Interval [center - delta, center + delta]
        a = center - delta
        b = center + delta
        
        # Handle wrap-around if necessary, but for small delta it's fine
        # Golden section search for maximum
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        
        # Function to evaluate magnitude at a specific angle
        def mag_at_angle(ang):
            # ang is scalar
            z_val = np.exp(1j * ang)
            # P = sum c_k z_val^k
            # Vectorized for single point
            p_val = np.sum(coeffs * (z_val ** np.arange(24)))
            return np.abs(p_val)
        
        fc = mag_at_angle(c)
        fd = mag_at_angle(d)
        
        # Iterate for high precision
        for _ in range(50):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - inv_phi * (b - a)
                fc = mag_at_angle(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + inv_phi * (b - a)
                fd = mag_at_angle(d)
            
            # Check convergence
            if np.abs(b - a) < 1e-10:
                break
        
        # The maximum is near the last c or d
        val = max(fc, fd)
        if val > max_val:
            max_val = val
            
    return max_val

def construct():
    # Deadline guard: 100 seconds to be safe (limit is 120s/150s)
    deadline = time.time() + 100.0
    n_coeffs = 24
    
    best_score = float('inf')
    best_coeffs = np.ones(n_coeffs)
    
    # Random restarts with local search
    while time.time() < deadline:
        # Generate random initial state
        coeffs = np.random.choice([-1, 1], size=n_coeffs).astype(np.float64)
        
        # Local Search: Single Flip
        current_score = _evaluate(coeffs)
        
        improved = True
        while improved:
            improved = False
            # Try flipping each coefficient
            for k in range(n_coeffs):
                # Flip
                coeffs[k] *= -1
                new_score = _evaluate(coeffs)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    coeffs[k] *= -1
                    
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_coeffs = coeffs.copy()
            
        # Optional: If we hit the known optimum closely, we can stop early
        # Known optimum is ~ 1.2583758 * sqrt(24)
        # sqrt(24) approx 4.898979
        # Target raw value approx 6.165
        target_raw = 1.2583758 * np.sqrt(24)
        if best_score <= target_raw * 1.000001:
            break

    return best_coeffs

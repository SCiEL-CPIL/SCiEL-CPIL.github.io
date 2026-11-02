import time
import numpy as np

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s < 1e-12:
        return 1e10
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(h_padded) ** 2))[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def project_simplex(v):
    """
    Project vector v onto the probability simplex {x >= 0, sum(x) = 1}.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.arange(1, n + 1)
    cond = u > (cssv - 1) / rho
    rho_star = np.max(rho[cond])
    theta = (cssv[rho_star - 1] - 1) / rho_star
    return np.maximum(v - theta, 0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initialize h
    # Uniform initialization is a good start, but adding noise might help escape local minima
    # if any exist (though the problem is likely convex-ish in the right space).
    # Let's start with uniform + small noise.
    np.random.seed(42)
    h = np.ones(N)
    h += np.random.randn(N) * 0.1
    h = project_simplex(h)
    
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Gradient descent steps
    # Objective: Minimize sum(c^2) where c = h*h
    # Gradient is proportional to h*h*h
    
    # Adaptive step size
    eta = 1.0 / (N * N) # Heuristic initial step size
    # Actually, let's normalize the gradient to have unit norm or similar?
    # Or just use a fixed small step size and maybe restart with different initializations?
    
    # The gradient magnitude scales with h^3. If h ~ 1/N, h^3 ~ 1/N^3.
    # c ~ 1/N^2.
    # Let's try a few iterations.
    
    max_iters = 500
    for i in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Compute c = conv(h, h)
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        C = H * H
        c = np.real(np.fft.ifft(C))[:2 * N - 1]
        
        # Compute v = conv(c, h)
        # v = ifft(fft(c) * fft(h))
        # We already have H. We need fft(c).
        # c is length 2N-1. Pad to 2*n_fft?
        # Or just compute v = h * h * h directly.
        # V = H * H * H
        V = C * H
        v = np.real(np.fft.ifft(V))[:N]
        
        # v is the gradient of sum(c^2) w.r.t h (up to a constant factor)
        # We want to minimize, so subtract gradient.
        # h_new = h - eta * v
        
        # Scale eta based on the magnitude of v to ensure stability
        # Let's use a line search or adaptive step.
        # Simple approach: normalize v to have sum of squares 1, then use fixed step?
        # Or just use a small eta.
        
        # Let's try eta = 0.1 / (N * np.max(np.abs(v)) + 1e-9)
        v_max = np.max(np.abs(v))
        if v_max < 1e-9:
            break
        eta = 0.5 / (N * v_max)
        
        h_new = h - eta * v
        h_new = project_simplex(h_new)
        
        score_new = compute_score(h_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            # Maybe decrease eta if we are making progress?
            # Or keep it.
            h = h_new
        else:
            # If score increased, reduce eta and retry?
            # Or just accept it to move out of local minima?
            # Minimizing sum(c^2) is a surrogate for minimizing max(c).
            # It might increase max(c) temporarily while flattening the rest.
            # So we should probably follow the gradient of the surrogate,
            # but keep track of the best actual score.
            
            # Let's just follow the gradient of the surrogate (sum c^2)
            # and trust that it leads to a flatter c.
            # But if it diverges, we need to be careful.
            # Let's use the surrogate gradient direction, but step carefully.
            
            # If we reject the step, we might get stuck.
            # Let's try a smaller step if the surrogate objective increases?
            # Surrogate obj: sum(c^2)
            # Let's compute it.
            obj_curr = np.sum(c**2)
            
            # Compute obj for h_new
            c_new_padded = np.zeros(n_fft)
            c_new_padded[:2*N-1] = np.real(np.fft.ifft(H * H))[:2*N-1] # This is just c
            # Wait, I need c for h_new.
            H_new = np.fft.fft(np.concatenate([h_new, np.zeros(n_fft - N)]))
            C_new = H_new * H_new
            c_new = np.real(np.fft.ifft(C_new))[:2*N-1]
            obj_new = np.sum(c_new**2)
            
            if obj_new < obj_curr:
                h = h_new
                # If surrogate improved, accept even if max score got slightly worse?
                # Yes, because we want to flatten.
            else:
                # Surrogate got worse. Reduce eta.
                eta *= 0.5
                if eta < 1e-6:
                    break
                # Try again with smaller eta
                h_new = h - eta * v
                h_new = project_simplex(h_new)
                h = h_new # Just update to the best found in this line search?
                
    return best_h.tolist()
# EVOLVE-BLOCK-END

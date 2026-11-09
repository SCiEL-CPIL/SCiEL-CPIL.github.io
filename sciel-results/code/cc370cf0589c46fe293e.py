import numpy as np
import time

N = 600

def compute_score_fft(h):
    """
    Computes the normalized maximum of the autoconvolution.
    Metric: max(h*h) / (||h||_1 * ||h||_inf)
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return float('inf')
        
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Use np.convolve for simplicity and correctness
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    idx = np.arange(N)
    candidates = []
    
    # 1. Constant
    candidates.append(np.ones(N))
    
    # 2. Triangle
    h_tri = 1.0 - np.abs(2.0 * idx / N - 1.0)
    candidates.append(h_tri)
    
    # 3. Gaussian
    center = N / 2.0
    sigma = N / 4.0
    x = (idx - center) / sigma
    candidates.append(np.exp(-0.5 * x**2))
    
    # 4. Sparse Candidates (Crucial for breaking the ~1.0 barrier)
    # Single spike
    h_s1 = np.zeros(N); h_s1[0] = 1.0
    candidates.append(h_s1)
    
    # Two spikes with decay
    h_s2 = np.zeros(N)
    h_s2[0] = 1.0
    h_s2[1] = 0.5
    candidates.append(h_s2)
    
    # Three spikes
    h_s3 = np.zeros(N)
    h_s3[0] = 1.0
    h_s3[1] = 0.5
    h_s3[2] = 0.25
    candidates.append(h_s3)
    
    # Random sparse vectors
    np.random.seed(42)
    for _ in range(10):
        h_r = np.zeros(N)
        num_nonzero = np.random.randint(1, 20)
        indices = np.random.choice(N, num_nonzero, replace=False)
        h_r[indices] = np.random.uniform(0.1, 1.0, num_nonzero)
        candidates.append(h_r)
        
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        if np.max(h) > 0:
            h_norm = h / np.max(h)
            score = compute_score_fft(h_norm)
            if score < best_score:
                best_score = score
                best_h = h_norm.copy()
        
        if time.time() > deadline:
            break
            
    if best_h is None:
        return [1.0] * N
        
    # Local Search: Hill Climbing
    current_h = best_h
    current_score = best_score
    
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    def perturb(h, scale):
        noise = np.random.normal(0, scale, N)
        h_new = h + noise
        h_new = np.clip(h_new, 0, 1)
        if np.max(h_new) == 0:
            return h, current_score
        return h_new, compute_score_fft(h_new)
    
    while time.time() < deadline:
        h_new, score_new = perturb(current_h, step_size)
        
        if score_new < current_score:
            current_h = h_new
            current_score = score_new
            no_improve_count = 0
            step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            step_size *= 0.95
            
            if no_improve_count > max_no_improve:
                # Try a larger jump
                h_jump, score_jump = perturb(current_h, 0.5)
                if score_jump < current_score:
                    current_h = h_jump
                    current_score = score_jump
                    no_improve_count = 0
                    step_size = 0.2
                else:
                    # Random restart from a sparse point
                    if np.random.rand() < 0.5:
                        h_r = np.zeros(N)
                        num_nonzero = np.random.randint(1, 10)
                        indices = np.random.choice(N, num_nonzero, replace=False)
                        h_r[indices] = np.random.uniform(0.1, 1.0, num_nonzero)
                        current_h = h_r
                        current_score = compute_score_fft(current_h)
                    else:
                        current_h = np.random.rand(N)
                        current_score = compute_score_fft(current_h)
                    step_size = 0.2
                    no_improve_count = 0

    return list(current_h)

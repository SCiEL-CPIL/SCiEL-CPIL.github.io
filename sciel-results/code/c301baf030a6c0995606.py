import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def compute_gradient(h, eps=1e-6):
        """Compute numerical gradient of the metric w.r.t. h."""
        h = np.asarray(h, dtype=np.float64)
        grad = np.zeros_like(h)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            m_plus = compute_metric(h_plus)
            h_minus = h.copy()
            h_minus[i] -= eps
            m_minus = compute_metric(h_minus)
            grad[i] = (m_plus - m_minus) / (2 * eps)
        return grad

    def projected_gradient_step(h, lr, max_step=1e-4):
        """One projected gradient descent step with non-negativity constraint."""
        h = np.asarray(h, dtype=np.float64)
        # Compute gradient numerically (use finite differences on a subset for speed)
        # For speed, use a smaller epsilon and only compute gradient on active indices
        eps = 1e-7
        grad = np.zeros_like(h)
        # Use central differences but batch them
        for i in range(0, N, 1):
            hp = h.copy()
            hp[i] += eps
            if hp[i] < 0:
                continue
            hm = h.copy()
            hm[i] = max(0, h[i] - eps)
            grad[i] = (compute_metric(hp) - compute_metric(hm)) / (2 * eps)
        
        # Gradient step
        h_new = h - lr * grad
        # Project onto non-negativity
        h_new = np.maximum(h_new, 0.0)
        # Renormalize to sum 1
        s = np.sum(h_new)
        if s > 1e-12:
            h_new = h_new / s
        else:
            h_new = h.copy()
        return h_new

    def simulated_annealing(h_init, max_iters=50000, T0=0.1, T1=0.001, alpha=0.9999):
        """Simulated annealing with diverse moves."""
        h = h_init.copy().astype(np.float64)
        total = np.sum(h)
        if total > 0:
            h = h / total
        
        best_h = h.copy()
        best_m = compute_metric(h)
        current_m = best_m
        T = T0
        
        np.random.seed(12345)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            T = T0 * (T1 / T0) ** (it / max_iters)
            
            # Choose move type
            move_type = np.random.choice(['swap', 'shift', 'reshape', 'swap_large'])
            
            h_new = h.copy()
            
            if move_type == 'swap':
                # Swap mass between two random positions
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                delta = np.random.uniform(-0.5, 0.5) * min(h[i], h[j])
                if h[i] + delta < 0 or h[j] - delta < 0:
                    continue
                h_new[i] += delta
                h_new[j] -= delta
            elif move_type == 'shift':
                # Shift mass from one position to a nearby position
                i = np.random.randint(N)
                offset = np.random.choice([-3, -2, -1, 1, 2, 3])
                j = i + offset
                if j < 0 or j >= N:
                    continue
                delta = np.random.uniform(0, h[i] * 0.5)
                if h[i] - delta < 0:
                    continue
                h_new[i] -= delta
                h_new[j] += delta
            elif move_type == 'reshape':
                # Modify a block of consecutive values
                block_size = np.random.randint(3, 11)
                start = np.random.randint(0, N - block_size)
                end = start + block_size
                block = h_new[start:end]
                total_block = np.sum(block)
                if total_block < 1e-12:
                    continue
                # Random perturbation of the block
                noise = np.random.normal(0, 0.1, size=block_size)
                block_new = block + noise
                block_new = np.maximum(block_new, 0)
                s = np.sum(block_new)
                if s > 1e-12:
                    block_new = block_new * total_block / s
                    h_new[start:end] = block_new
            elif move_type == 'swap_large':
                # Swap larger amounts between distant positions
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                delta = np.random.uniform(-0.3, 0.3) * max(h[i], h[j])
                if h[i] + delta < 0 or h[j] - delta < 0:
                    continue
                h_new[i] += delta
                h_new[j] -= delta
            
            # Ensure non-negativity and normalize
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s < 1e-12:
                continue
            h_new = h_new / s
            
            m_new = compute_metric(h_new)
            
            # Accept or reject
            if m_new < current_m or np.random.random() < np.exp(-(m_new - current_m) / T):
                h = h_new
                current_m = m_new
                if m_new < best_m:
                    best_m = m_new
                    best_h = h_new.copy()
        
        return best_h, best_m

    def gradient_annealing(h_init, max_iters=30000):
        """Combine gradient descent with simulated annealing."""
        h = h_init.copy().astype(np.float64)
        total = np.sum(h)
        if total > 0:
            h = h / total
        
        best_h = h.copy()
        best_m = compute_metric(h)
        current_m = best_m
        
        np.random.seed(67890)
        
        lr = 1e-4
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Alternate between gradient step and random perturbation
            if it % 10 == 0:
                # Gradient step
                h_new = projected_gradient_step(h, lr)
            else:
                # Random perturbation
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                delta = np.random.uniform(-0.2, 0.2) * min(h[i], h[j])
                h_new = h.copy()
                if h_new[i] + delta >= 0 and h_new[j] - delta >= 0:
                    h_new[i] += delta
                    h_new[j] -= delta
                else:
                    continue
            
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s < 1e-12:
                continue
            h_new = h_new / s
            
            m_new = compute_metric(h_new)
            
            # Accept with probability based on improvement
            if m_new < current_m:
                h = h_new
                current_m = m_new
                if m_new < best_m:
                    best_m = m_new
                    best_h = h_new.copy()
                # Decrease learning rate slightly
                lr = max(lr * 0.999, 1e-6)
            else:
                # Occasionally accept worse solutions to escape local minima
                if np.random.random() < 0.1:
                    h = h_new
                    current_m = m_new
        
        return best_h, best_m

    # Design asymmetric initializations to escape uniform critical point
    i_arr = np.arange(N)
    mid = N / 2.0
    
    initializations = [
        # Uniform
        np.ones(N),
        # Cosine bump
        0.5 * (1.0 + np.cos(2 * np.pi * (i_arr - mid) / N)),
        # Gaussian
        np.exp(-4.0 * ((i_arr - mid) / (N / 4.0)) ** 2),
        # Asymmetric: more mass on left
        np.exp(-3.0 * ((i_arr - mid * 0.7) / (N / 5.0)) ** 2),
        # Asymmetric: more mass on right
        np.exp(-3.0 * ((i_arr - mid * 1.3) / (N / 5.0)) ** 2),
        # Two bumps
        np.exp(-6.0 * ((i_arr - mid * 0.6) / (N / 8.0)) ** 2) + np.exp(-6.0 * ((i_arr - mid * 1.4) / (N / 8.0)) ** 2),
        # Random asymmetric
        np.random.RandomState(42).exponential(1.0, size=N),
    ]
    
    best_h = None
    best_m = float('inf')
    
    # Phase 1: Gradient annealing from multiple initializations
    for init in initializations:
        if time.time() > deadline - 20:
            break
        init = np.maximum(init, 0.0)
        s = np.sum(init)
        if s < 1e-12:
            continue
        init = init / s
        
        h_opt, m_opt = gradient_annealing(init, max_iters=20000)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    # Phase 2: Simulated annealing refinement on best result
    if best_h is not None and time.time() < deadline - 5:
        h_ref, m_ref = simulated_annealing(best_h, max_iters=30000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref

    if best_h is None:
        return [1.0] * N

    return list(best_h)

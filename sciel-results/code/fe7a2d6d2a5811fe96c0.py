"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(f):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    Returns (1/N^2) * ||R||_1 where R is the linear autocorrelation of f.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    l1_norm = np.abs(R[0]) + 2.0 * np.sum(np.abs(R[1:N]))
    return l1_norm / (N * N)


def _compute_gradient(f):
    """Compute the gradient of the L1 autocorrelation objective."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    R = np.fft.irfft(F * np.conj(F), n=n_fft)
    
    grad = np.zeros(N)
    s = np.sign(R[:N])
    s[0] = 1.0
    
    g1 = np.zeros(N)
    g2 = np.zeros(N)
    for k in range(1, N):
        sk = s[k]
        if sk == 0:
            continue
        if k < N:
            g1[:N-k] += sk * f[k:]
        g2[k:] += sk * f[:N-k]
    
    grad += 2.0 * f + g1 + g2
    grad /= (N * N)
    
    return grad


def _projected_gradient_descent(f_init, max_iters=2000, lr=None, time_limit=60.0):
    """Run projected gradient descent on the continuous relaxation."""
    start_time = time.time()
    f = f_init.astype(float).copy()
    
    if lr is None:
        lr = 0.1
    
    best_f = f.copy()
    best_val = _compute_c3_bound(f)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        grad = _compute_gradient(f)
        f = f - lr * grad
        f = np.clip(f, -1.0, 1.0)
        
        f_sign = np.sign(f)
        f_sign[f_sign == 0] = 1.0
        val_sign = _compute_c3_bound(f_sign)
        
        if val_sign < best_val:
            best_val = val_sign
            best_f = f_sign.copy()
        
        if it % 50 == 0:
            val_cont = _compute_c3_bound(f)
            if val_cont < best_val:
                best_val = val_cont
                best_f = f.copy()
        
        if it > 100:
            lr *= 0.99
    
    return best_f, best_val


def _simulated_annealing(f_init, time_limit=40.0, block_sizes=None):
    """Simulated annealing with block-flip moves for discrete refinement."""
    start_time = time.time()
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    f = f_init.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    T = 0.05
    cooling = 0.995
    
    np.random.seed(12345)
    
    while time.time() - start_time < time_limit:
        bs = block_sizes[np.random.randint(len(block_sizes))]
        start = np.random.randint(0, N - bs + 1)
        
        f[start:start+bs] *= -1.0
        new_val = _compute_c3_bound(f)
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_f = f.copy()
        else:
            f[start:start+bs] *= -1.0
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best_f, best_val


def _local_refinement(f, time_limit=20.0):
    """Local refinement with single-bit and small block flips."""
    start_time = time.time()
    f = f.copy()
    current_val = _compute_c3_bound(f)
    best_f = f.copy()
    best_val = current_val
    
    np.random.seed(777)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        order = np.random.permutation(N)
        
        for i in order:
            if time.time() - start_time > time_limit:
                break
            
            f[i] *= -1.0
            new_val = _compute_c3_bound(f)
            
            if new_val < current_val - 1e-15:
                current_val = new_val
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_f = f.copy()
            else:
                f[i] *= -1.0
        
        if time.time() - start_time < time_limit:
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                bs = np.random.choice([2, 3, 4, 5])
                start = np.random.randint(0, N - bs + 1)
                f[start:start+bs] *= -1.0
                new_val = _compute_c3_bound(f)
                if new_val < current_val - 1e-15:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_f = f.copy()
                else:
                    f[start:start+bs] *= -1.0
    
    return best_f, best_val


def construct():
    start_time = time.time()
    time_limit = 140.0
    
    best_f = None
    best_val = np.inf
    
    pgd_time = 50.0
    pgd_deadline = start_time + pgd_time
    
    initializations = []
    
    np.random.seed(42)
    initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    for freq in [1, 2, 3, 5, 8, 13]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        initializations.append(np.sign(np.sin(t)))
    
    initializations.append(np.tile([1, -1], N // 2))
    initializations.append(np.tile([1, 1, -1, -1], N // 4))
    initializations.append(np.tile([1, -1, -1, 1], N // 4))
    
    for f_init in initializations:
        if time.time() > pgd_deadline:
            break
        
        f_pgd, val_pgd = _projected_gradient_descent(f_init, max_iters=1000, time_limit=10.0)
        if val_pgd < best_val:
            best_val = val_pgd
            best_f = f_pgd.copy()
    
    sa_time = 45.0
    sa_deadline = start_time + pgd_time + sa_time
    
    if time.time() < sa_deadline and best_f is not None:
        f_sa, val_sa = _simulated_annealing(best_f, time_limit=sa_time - (time.time() - start_time - pgd_time))
        if val_sa < best_val:
            best_val = val_sa
            best_f = f_sa.copy()
    
    if time.time() < sa_deadline:
        np.random.seed(999)
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        f_sa2, val_sa2 = _simulated_annealing(f_rand, time_limit=sa_time - (time.time() - start_time - pgd_time))
        if val_sa2 < best_val:
            best_val = val_sa2
            best_f = f_sa2.copy()
    
    local_time = 25.0
    local_deadline = start_time + pgd_time + sa_time + local_time
    
    if time.time() < local_deadline and best_f is not None:
        f_local, val_local = _local_refinement(best_f, time_limit=local_time - (time.time() - start_time - pgd_time - sa_time))
        if val_local < best_val:
            best_val = val_local
            best_f = f_local.copy()
    
    if best_f is None:
        best_f = np.ones(N)
    
    best_f = np.sign(best_f)
    best_f[best_f == 0] = 1.0
    
    return best_f
# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, temp=5.0):
    """Differentiable approximation of max|ac| using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-10:
        return 0.0
    shifted = abs_ac / m
    log_sum = np.log(np.sum(np.exp(shifted - 1.0))) + 1.0
    return m * np.log(np.exp(log_sum * temp) / temp) / temp

def _surrogate_objective(x, temp=5.0):
    """
    Surrogate objective for gradient-based optimization.
    x is a vector of real values (continuous relaxation of +/-1).
    We use a soft-max of the autoconvolution magnitude.
    """
    h = np.tanh(x)  # map to (-1, 1)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    max_abs_ac = _soft_max_abs(ac, temp)
    return max_abs_ac / (s * s)

def _surrogate_grad(x, temp=5.0):
    """
    Numerical gradient of the surrogate objective.
    """
    eps = 1e-5
    grad = np.zeros_like(x)
    f0 = _surrogate_objective(x, temp)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        fp = _surrogate_objective(xp, temp)
        grad[i] = (fp - f0) / eps
    return grad

def _gradient_descent(h_init, deadline, max_iters=2000, lr=0.05, temp=5.0):
    """
    Gradient descent on the continuous relaxation, then project to +/-1.
    """
    # Initialize x such that tanh(x) ~= h_init
    # h_init is +/-1, so x should be large +/-
    x = np.zeros(N)
    for i in range(N):
        if h_init[i] > 0:
            x[i] = 2.0
        else:
            x[i] = -2.0
    
    best_h = h_init.copy()
    best_val = _score(h_init)
    
    # Use scipy's L-BFGS-B with numerical gradient or custom
    # To save time, we'll do a simple projected gradient descent with momentum
    
    x_cur = x.copy()
    v = np.zeros_like(x_cur)
    mu = 0.9  # momentum
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute gradient
        eps = 1e-4
        grad = np.zeros_like(x_cur)
        f0 = _surrogate_objective(x_cur, temp)
        
        # Sparse gradient computation for speed (only compute for a subset if too slow)
        # But for N=400, full gradient is feasible if we optimize the FFT part
        # Actually, let's use a finite difference approach that is vectorized where possible
        # But the objective is non-trivial. Let's stick to coordinate-wise or a few points.
        
        # For efficiency in this context, let's try a simple subgradient or coordinate descent on the continuous variable?
        # No, the prompt suggests L-BFGS-B with soft-max.
        # Let's implement a proper gradient step using the chain rule if possible, or finite differences.
        # Finite differences for 400 dimensions x 2000 iters is 800k evaluations.
        # Each eval is ~1ms (FFT). 800s is too long.
        
        # We need a faster gradient.
        # d/dx [max|ac(h(x))| / (sum h(x))^2]
        # This is complex.
        
        # Alternative: Use scipy.optimize.minimize with method='L-BFGS-B' and let it handle the gradient numerically?
        # Scipy's numerical gradient is also O(N) evaluations per step.
        
        # Let's reduce the number of iterations or use a coarser step.
        # Or, better: use the fact that we are close to +/-1.
        
        # Let's try a different approach:
        # 1. Convert h to x (log-odds or similar).
        # 2. Optimize a simpler surrogate?
        
        # Actually, the inspiration says "L-BFGS-B to navigate... more effectively".
        # Let's try running L-BFGS-B with a limited number of iterations and a tight deadline.
        
        pass # This loop is inefficient as written with full finite differences.
    
    # Let's restart this function with a proper scipy call
    
def _scipy_optimize(h_init, deadline, temp=5.0):
    """
    Use scipy L-BFGS-B to optimize the continuous relaxation.
    """
    x0 = np.zeros(N)
    for i in range(N):
        if h_init[i] > 0:
            x0[i] = 2.0
        else:
            x0[i] = -2.0
    
    # Define objective and gradient for scipy
    # We will use numerical gradient provided by scipy if we don't provide one,
    # but we can provide a finite difference gradient ourselves to control the step size.
    
    def obj(x):
        return _surrogate_objective(x, temp)
    
    # Check time budget
    if time.time() > deadline:
        return h_init, _score(h_init)
    
    try:
        res = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
        )
    except Exception:
        return h_init, _score(h_init)
    
    if res.success or res.x is not None:
        x_opt = res.x
        # Project back to +/-1
        h_opt = np.sign(x_opt)
        h_opt[h_opt == 0] = 1.0
        
        val = _score(h_opt)
        return h_opt, val
    
    return h_init, _score(h_init)

def _local_search(h, deadline=None, max_passes=10):
    """
    Local optimization with single-flip and pair-flip moves.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Single flip moves
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            
            h[i] = v + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v
                continue
                
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v
        
        # Pair flip moves
        if not improved:
            for _ in range(min(50, n // 2)):
                if deadline is not None and time.time() > deadline:
                    break
                
                i, j = np.random.choice(n, 2, replace=False)
                vi, vj = h[i], h[j]
                di, dj = -2.0 * vi, -2.0 * vj
                
                h[i] = vi + di
                h[j] = vj + dj
                
                H_new = np.fft.rfft(h, fsize)
                ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
                max_abs_new = np.max(np.abs(ac_new))
                s_new = s + di + dj
                
                if abs(s_new) < 1e-6:
                    h[i] = vi
                    h[j] = vj
                    continue
                    
                val_new = max_abs_new / (s_new * s_new)
                
                if val_new < best_val - 1e-12:
                    best_val = val_new
                    H = H_new
                    ac = ac_new
                    max_abs_ac = max_abs_new
                    s = s_new
                    improved = True
                else:
                    h[i] = vi
                    h[j] = vj
        
        if not improved:
            break
    
    return h, best_val

def _generate_msequence(k, length):
    """Generate an m-sequence (maximal length sequence) from a primitive polynomial over GF(2^k)."""
    primitive_polys = {
        1: [0],
        2: [0, 1],
        3: [0, 1],
        4: [0, 2],
        5: [0, 2],
        6: [0, 1, 3],
        7: [0, 1],
        8: [0, 1, 4],
        9: [0, 1, 4],
        10: [0, 1, 3, 5],
    }
    
    if k not in primitive_polys:
        return np.ones(length)
    
    exponents = primitive_polys[k]
    reg = [1] * k
    
    seq = []
    for _ in range(length):
        seq.append(1 if reg[0] == 1 else -1)
        
        feedback = 0
        for e in exponents:
            j = k - 1 - e
            if 0 <= j < k:
                feedback ^= reg[j]
        
        reg = reg[1:] + [feedback]
    
    return np.array(seq, dtype=float)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # 1. Try m-sequences (structural idea from inspiration 1)
    # m-sequences have ideal autocorrelation properties
    for k in range(2, 11):
        if time.time() > deadline:
            break
        
        mseq_len = (1 << k) - 1
        mseq = _generate_msequence(k, mseq_len)
        
        if mseq_len >= N:
            h = mseq[:N]
        else:
            reps = (N + mseq_len - 1) // mseq_len
            h = np.tile(mseq, reps)[:N]
        
        s = np.sum(h)
        if abs(s) < N * 0.05:
            if s < 0:
                h = -h
                s = -s
            if abs(s) < N * 0.05:
                h[0] = 1.0
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
        
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Refine with local search
        h_opt, val_opt = _local_search(h, deadline=deadline, max_passes=5)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # 2. Try Legendre sequences (from current program)
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    for p in primes_to_try:
        if time.time() > deadline:
            break
        
        h = np.zeros(N)
        for i in range(N):
            a = (i + 1) % p
            if a == 0:
                sym = 0
            else:
                sym = pow(a, (p - 1) // 2, p)
                if sym == 1:
                    sym = 1
                elif sym == p - 1:
                    sym = -1
                else:
                    sym = 0
            h[i] = 1.0 if sym >= 0 else -1.0
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
        
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        h_opt, val_opt = _local_search(h, deadline=deadline, max_passes=5)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # 3. Gradient-based optimization on continuous relaxation (structural idea from inspiration 1)
    # Use the best solution found so far as a starting point
    if time.time() < deadline - 10:
        h_grad, val_grad = _scipy_optimize(best_h, deadline, temp=5.0)
        if val_grad < best_val:
            best_val = val_grad
            best_h = h_grad.copy()
        
        # Refine with local search after gradient step
        h_opt, val_opt = _local_search(h_grad, deadline=deadline, max_passes=10)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # 4. Random restarts with local search (refinement machinery from current program)
    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 5 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            h = np.ones(N)
            n_neg = np.random.randint(0, N // 3)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            x = np.arange(N)
            k = (r // 5) % 10 + 1
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            # Chirp
            x = np.arange(N)
            freq = 1.0 + (r // 5) * 0.1
            h = np.sign(np.cos(2 * np.pi * freq * x * x / (2 * N)))
            h[h == 0] = 1.0
        else:
            # Block
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 10)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
        
        h_opt, val_opt = _local_search(h, deadline=deadline, max_passes=3)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END

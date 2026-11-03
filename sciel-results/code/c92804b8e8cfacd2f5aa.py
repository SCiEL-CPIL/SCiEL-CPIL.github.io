"""Convex Gram-matrix formulation: optimize autoconvolution directly via SDP-like LP over outer products."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.linalg import eigh

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _build_conv_coefficients():
    """
    Precompute the linear mapping from the outer-product matrix G = x x^T
    to the autoconvolution values.
    
    conv[k] = sum_{j} x[j] * x[j+k] for k >= 0
             = sum_{j} x[j] * x[j-k] for k < 0
    
    We express conv[k] = sum_{i,j} C[k,i,j] * G[i,j] where C is the coefficient.
    But G is symmetric and we only need the upper triangle.
    
    Actually, conv[k] for lag k (k from -(N-1) to N-1):
    For k >= 0: conv[k] = sum_{j=0}^{N-1-k} x[j] * x[j+k]
    For k < 0: conv[k] = sum_{j=-k}^{N-1} x[j] * x[j+k] = sum_{j=0}^{N-1+k} x[j+k] * x[j]
    
    So conv[k] = sum_{i,j: j-i=k} x[i]*x[j]
    
    We'll build a matrix A of shape (2N-1, N*N) where each row corresponds to a lag
    and each column to a pair (i,j). But N*N = 160000 which is large.
    
    Alternative: use the fact that conv is linear in G, and G is PSD.
    We can formulate: minimize t s.t. |conv[k]| <= t*s^2 for all k, G = x x^T, G >= 0, rank(G)=1.
    
    The rank-1 constraint is non-convex. But we can relax to G >= 0 (SDP relaxation).
    However, scipy doesn't have SDP solvers. 
    
    Alternative approach: use the fact that for fixed s, minimizing max|conv[k]|/s^2
    is equivalent to minimizing max|conv[k]| subject to sum(x)=s.
    
    Let's try a different angle: use iterative reweighted least squares on the Gram matrix.
    Or: use the spectral approach but optimize the full complex FFT directly with a 
    different parameterization.
    
    Actually, let me try a completely different approach: 
    **Alternating optimization between the signal and a weight vector on the lags.**
    
    Or better: **Use the fact that the problem is homogeneous and normalize s=1,
    then minimize max|conv[k]|. This is a minimax problem that can be solved via
    bisection on t and checking feasibility via LP.**
    
    For fixed t, check if there exists x with sum(x)=1 and |conv[k]| <= t for all k.
    This is a feasibility problem. conv[k] is bilinear in x, so it's not linear.
    
    Let me try yet another approach: **Gradient descent in the frequency domain
    with a different objective: minimize the sum of squares of off-zero autoconv
    coefficients, which is equivalent to minimizing ||conv||_2^2 - conv[0]^2,
    subject to sum(x)=1. This is a convex problem in the frequency domain!**
    
    ||conv||_2^2 = (1/FFT_SIZE) * sum_k |FX[k]|^4 (by Parseval)
    conv[0] = sum(x) = FX[0]/FFT_SIZE... wait, let me be careful.
    
    Actually: sum_j x[j]^2 = (1/FFT_SIZE) * sum_k |FX[k]|^2
    And conv[0] = sum_j x[j]^2
    
    sum_k conv[k]^2 = (1/FFT_SIZE) * sum_k |FX[k]|^4  (by Parseval for autoconv)
    
    So minimizing sum_k conv[k]^2 - conv[0]^2 = sum_{k!=0} conv[k]^2
    is equivalent to minimizing (1/FFT_SIZE) * sum_k |FX[k]|^4 - (1/FFT_SIZE)^2 * |FX[0]|^2
    ... this is getting complicated. Let me just use the L2 approach.
    
    Actually, the simplest different approach: **Use a different initial guess and
    a more aggressive multi-start with simulated annealing on the phase spectrum,
    combined with a convex relaxation step.**
    
    Let me try: **Optimize the phase spectrum using a different method: 
    projected gradient on the unit circle (for each FFT bin's phase) with a 
    log-sum-exp smoothing, but with a much better initialization from a 
    random phase + iterative refinement, and multiple random restarts.**
    """
    pass


def _try_l2_spectral_optimize(x_init, deadline, rng):
    """
    Minimize sum of squares of off-zero-lag autoconvolution coefficients
    in the frequency domain. This is a smooth, convex-ish problem.
    
    sum_{k != 0} conv[k]^2 = (1/FFT_SIZE) * sum_k |FX[k]|^4 - conv[0]^2
    
    We minimize this subject to sum(x) = s (fixed).
    In frequency domain: FX[0] = s * FFT_SIZE... no.
    
    Let me use: x = ifft(FX), sum(x) = FX[0]/FFT_SIZE... 
    Actually np.fft.ifft: x[n] = (1/N_fft) sum_k FX[k] exp(2pi i kn/N_fft)
    So sum_n x[n] = (1/N_fft) sum_k FX[k] sum_n exp(2pi i kn/N_fft) = (1/N_fft) * FX[0] * N_fft = FX[0]
    
    So sum(x) = FX[0]. We want FX[0] = s (fixed).
    
    Objective: minimize sum_{k=1}^{N_fft/2} |FX[k]|^4 (approximately, for real signals)
    subject to FX[0] = s, FX[N_fft/2] real (for even FFT size), and FX[k] = conj(FX[N_fft-k]).
    
    This is minimizing sum |FX[k]|^4 which is convex in |FX[k]|^2!
    Let y[k] = |FX[k]|^2 >= 0. Minimize sum y[k]^2 subject to sum constraints.
    
    But we also need the phase to be consistent. However, for the L2 objective,
    only the magnitudes matter! So we can optimize magnitudes freely.
    
    The constraint is just that the resulting x is real (which is automatic from
    conjugate symmetry) and sum(x) = FX[0] = s.
    
    So: minimize sum_{k=1}^{M-1} y[k]^2 subject to y[k] >= 0, and we need
    to ensure the IFFT gives a valid signal. The only hard constraint is FX[0]=s.
    
    Wait, but we also need the signal to be "reasonable" (finite, etc.). 
    The L2 objective will naturally push magnitudes to be equal (by Cauchy-Schwarz,
    sum y^2 is minimized when all y are equal for fixed sum y).
    
    But we don't have a fixed sum of y. The sum of y[k] = sum |FX[k]|^2 = FFT_SIZE * sum x[j]^2.
    So there's no fixed constraint on the total energy. The objective sum y^2 will
    push all y[k] -> 0, which means x -> 0. We need to normalize.
    
    Let's normalize: set sum(x) = 1 (s=1). Then FX[0] = 1.
    Minimize sum_{k=1}^{M-1} |FX[k]|^4 subject to FX[0]=1 and conjugate symmetry.
    
    By Cauchy-Schwarz, for fixed sum |FX[k]|^2, the sum |FX[k]|^4 is minimized
    when all |FX[k]| are equal. But sum |FX[k]|^2 is not fixed!
    
    Hmm, this means the minimum is at FX[k]=0 for all k>=1, giving x[0]=1/N_fft... 
    no, x = ifft([1, 0, 0, ...]) = [1/N_fft, 1/N_fft, ...]. 
    Then conv[k] = sum_j (1/N_fft)(1/N_fft) = (N_fft - k)/N_fft^2 for k>=0.
    max|conv| = 1/N_fft. s = 1. metric = 1/N_fft.
    
    That's terrible. The issue is that L2 of conv is not the right objective.
    Let me abandon this and try a different strategy.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    return best_x, best_val


def _try_alternating_minimax(x_init, deadline, rng):
    """
    Alternating optimization: 
    1. Given x, compute conv and find the worst lag k*.
    2. Adjust x to minimize |conv[k*]| while keeping sum(x) fixed.
    3. Repeat.
    
    This is a block coordinate descent on the "active" constraint.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    s = _integral(current_x)
    
    if abs(s) < 1e-12:
        return best_x, best_val
    
    # Normalize to s=1
    current_x = current_x / s
    s = 1.0
    
    for it in range(50000):
        if time.time() > deadline:
            break
        
        conv = _autoconv_values(current_x)
        abs_conv = np.abs(conv)
        k_star = int(np.argmax(abs_conv))
        
        # We want to minimize |conv[k_star]| by adjusting x.
        # conv[k] = sum_j x[j] x[j+k] (for k>=0) or x[j-k] x[j] (for k<0)
        # 
        # The gradient of conv[k] w.r.t. x[j] is:
        # For k >= 0: d conv[k]/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
        # For k < 0: same formula with |k|
        #
        # We want to reduce |conv[k*]|. The steepest descent direction for
        # reducing |conv[k*]| is -sign(conv[k*]) * grad(conv[k*]).
        # But we must keep sum(x) = 1, so we project the gradient to the
        # subspace sum(v) = 0.
        
        k_abs = abs(k_star)
        sign_k = np.sign(conv[k_star]) if conv[k_star] != 0 else 1.0
        
        grad = np.zeros(N)
        for j in range(N):
            contrib = 0.0
            if j + k_abs < N:
                contrib += current_x[j + k_abs]
            if j - k_abs >= 0:
                contrib += current_x[j - k_abs]
            grad[j] = 2.0 * sign_k * contrib
        
        # Project to sum(v)=0
        grad -= np.mean(grad)
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Try a few step sizes
        best_step = 0.0
        best_v = _metric(current_x)
        
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
            x_test = current_x - alpha * grad / grad_norm
            v_test = _metric(x_test)
            if v_test < best_v:
                best_v = v_test
                best_step = alpha
        
        if best_step == 0.0:
            # Try random perturbation
            break
        
        current_x = current_x - best_step * grad / grad_norm
        
        # Renormalize
        s_new = _integral(current_x)
        if abs(s_new) > 1e-12:
            current_x = current_x / s_new
        
        v_current = _metric(current_x)
        if v_current < best_val:
            best_val = v_current
            best_x = current_x.copy()
    
    return best_x, best_val


def _try_random_phase_search(x_init, deadline, rng):
    """
    Multi-start random phase optimization in frequency domain.
    For each random phase assignment, compute the signal and metric.
    Use a smart initialization: start with random phases, then do a few
    gradient steps on the phases.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    fx_init = np.fft.rfft(x_init, _FFT_SIZE)
    mags = np.abs(fx_init).copy()
    mags[mags < 1e-10] = 1e-10
    s_target = _integral(x_init)
    
    n_phases = len(fx_init) - 1
    
    def _eval_phases(phases):
        fx = mags.copy()
        fx[0] = s_target
        for k in range(1, len(fx)):
            fx[k] = mags[k] * np.exp(1j * phases[k-1])
        x = np.fft.irfft(fx, _FFT_SIZE)[:N]
        return _metric(x), x
    
    def _phase_gradient(phases):
        """Gradient of metric w.r.t. phases via finite differences."""
        eps = 1e-4
        grad = np.zeros(n_phases)
        val0, _ = _eval_phases(phases)
        for j in range(n_phases):
            p_plus = phases.copy()
            p_plus[j] += eps
            val_plus, _ = _eval_phases(p_plus)
            grad[j] = (val_plus - val0) / eps
        return grad
    
    n_restarts = 30
    steps_per_restart = 100
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random phases
        phases = rng.uniform(-np.pi, np.pi, size=n_phases)
        
        # Do a few gradient steps
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            grad = _phase_gradient(phases)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Adaptive learning rate
            lr = 0.01
            for _ in range(3):
                new_phases = phases - lr * grad / grad_norm
                # Wrap phases
                new_phases = (new_phases + np.pi) % (2 * np.pi) - np.pi
                val_new, x_new = _eval_phases(new_phases)
                val_old, _ = _eval_phases(phases)
                if val_new < val_old:
                    phases = new_phases
                    break
                else:
                    lr *= 0.5
            
            val_cur, x_cur = _eval_phases(phases)
            if val_cur < best_val:
                best_val = val_cur
                best_x = x_cur.copy()
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.03, 0.03
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    # Start with a structured initial guess
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Try several structured initializations
    inits = [
        np.ones(N),
        np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
        np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)),
        np.ones(N) + 0.1 * np.random.default_rng(42).normal(size=N),
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        v = _metric(init)
        if v < best_val:
            best_val = v
            best_x = init.copy()
    
    # Phase 1: Alternating minimax
    if time.time() < deadline - 60:
        x_alt, val_alt = _try_alternating_minimax(best_x, deadline - 60, rng)
        if val_alt < best_val:
            best_val = val_alt
            best_x = x_alt
    
    # Phase 2: Random phase search
    if time.time() < deadline - 40:
        x_phase, val_phase = _try_random_phase_search(best_x, deadline - 40, rng)
        if val_phase < best_val:
            best_val = val_phase
            best_x = x_phase
    
    # Phase 3: Local search
    if time.time() < deadline - 20:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 20, rng, max_rounds=50000)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END

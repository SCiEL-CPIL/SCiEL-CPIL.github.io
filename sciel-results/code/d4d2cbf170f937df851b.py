"""Convex spectral magnitude minimization: minimize L1(autoconv) subject to fixed DC component."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

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

def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _solve_convex_spectral(deadline):
    """
    Solve the convex problem:
    Minimize ||x||_2^2
    Subject to:
      1. sum(x) = S (Fixed DC component)
      2. max_k |autoconv[k]| <= 1 (This is equivalent to L_infinity constraint on autoconv)
    
    Wait, the objective is max|autoconv| / S^2.
    If we fix S, we want to minimize max|autoconv|.
    However, minimizing max|autoconv| is not convex in x (it's quasi-convex).
    
    Let's use the dual of the L1 norm.
    Minimize ||autoconv(x)||_1
    Subject to sum(x) = S
    
    This is a Convex Quadratic Program (CQP) because autoconv(x) is linear in the outer product xx^T, 
    but wait, autoconv(x) is quadratic in x.
    Actually, autoconv(x) = IFFT(|FFT(x)|^2).
    The mapping x -> autoconv(x) is quadratic.
    Minimizing L1(quadratic(x)) is non-convex.
    
    HOWEVER, if we work in the frequency domain:
    Let X = FFT(x).
    autoconv(x) = IFFT(|X|^2).
    We want to minimize max_k |IFFT(|X|^2)_k| / (Re(X[0])^2).
    Let S = Re(X[0]).
    We want to minimize max_k |IFFT(|X|^2)_k| subject to Re(X[0]) = S.
    
    This is still non-convex because |X|^2 is quadratic in X.
    
    Alternative:
    Let's look at the structure.
    The "flat spectrum" heuristic suggests that we should try to make |X[k]| as flat as possible for k>0.
    
    Let's try a different convex relaxation.
    Consider the problem:
    Minimize t
    Subject to:
      -t <= autoconv_k(x) <= t  for all k
      sum(x) = S
    
    This is a Quadratically Constrained Quadratic Program (QCQP).
    Specifically, the constraints are of the form:
      -t <= x^T A_k x <= t
      where A_k is the circulant matrix corresponding to the shift k.
    
    This is a non-convex QCQP in general.
    
    However, there is a known result for "Minimax Autoconvolution".
    The optimal solution often has a specific structure.
    
    Let's try a different approach: **Alternating Minimization on the Gram Matrix**.
    Let G = x x^T.
    autoconv_k = sum_{i,j} G_{i,j} delta(i+j-k) ... no, autoconv is x * x.
    (x * x)[k] = sum_i x[i] x[k-i].
    This is linear in G = x x^T.
    
    So, we can formulate:
    Minimize t
    Subject to:
      -t <= C_k(G) <= t  for all k
      trace(G) = ||x||^2 (This is not fixed, but we can fix ||x|| to 1)
      G >= 0 (PSD)
      rank(G) = 1 (This is the non-convex part)
      sum(x) = S -> sum_i G_{i,i} + sum_{i!=j} G_{i,j} = S^2? No.
      (sum x)^2 = x^T 1 1^T x = trace(G 1 1^T).
    
    If we drop the rank constraint, we have an SDP.
    Minimize t
    Subject to:
      -t <= C_k(G) <= t
      trace(G 1 1^T) = S^2
      G >= 0
    
    Solving this SDP gives a lower bound on the optimal max|autoconv| for a given S.
    We can then extract a rank-1 approximation or use it as a guide.
    
    However, SDP solvers might be too slow or unavailable (scipy doesn't have a robust SDP solver built-in, only linprog/minimize).
    
    Let's go back to **Spectral Magnitude Optimization** but using a **Convex** surrogate.
    
    The function f(m) = max_k |IFFT(m^2)_k| is convex in m (where m = |X|).
    Proof: IFFT is linear. m^2 is convex? No, m^2 is convex.
    Actually, let y = m^2. Then IFFT(y) is linear in y.
    max_k |IFFT(y)_k| is convex in y.
    y = m^2 is a convex transformation of m.
    So the composition is convex in m.
    
    So, minimizing max_k |IFFT(|X|^2)_k| subject to Re(X[0]) = S is a **Convex Optimization Problem** in the variable m = |X|.
    
    We can solve this using `scipy.optimize.minimize` with SLSQP or L-BFGS-B (with gradients).
    Variables: m[1..K] (m[0] is fixed by S).
    Objective: max_k |IFFT(m^2)_k|
    
    This is a much better starting point than random phases.
    
    Once we have the optimal m, we need to find phases phi such that the signal x is real and has the desired sum.
    Actually, if we fix m, the sum is determined by m[0] (DC).
    The phases affect the time-domain signal x.
    We want to minimize max|autoconv|.
    The autoconvolution depends only on |X|^2 = m^2.
    So ANY set of phases that yields a real signal x with sum S will have the SAME autoconvolution!
    
    Wait, is that true?
    autoconv(x) = IFFT(|FFT(x)|^2).
    Yes, the autoconvolution is completely determined by the power spectrum |X|^2.
    The phases of X do not affect the autoconvolution.
    
    Therefore, the problem reduces to:
    1. Find m = |X| that minimizes max_k |IFFT(m^2)_k| subject to m[0] = S (and m being a valid magnitude spectrum for a real signal, i.e., m is symmetric if we consider full FFT, but for rfft it's just m[0..K]).
    2. Construct a real signal x with FFT magnitude m and sum S.
       - x must be real, so X must satisfy conjugate symmetry.
       - For rfft, X[k] = m[k] * exp(i * phi[k]).
       - We need to choose phi[k] such that IFFT(X) is real.
       - If we set phi[k] = 0 for all k, x is real and even (symmetric).
       - Does x have sum S?
         sum(x) = X[0] = m[0] * exp(i*0) = S. Yes.
    
    So, the strategy is:
    1. Solve for m[1..K] to minimize max_k |IFFT(m^2)_k| with m[0] fixed.
    2. Set X[k] = m[k] (all phases zero).
    3. x = IFFT(X).
    4. This x is real, even, and has the optimal autoconvolution for the given S.
    
    Wait, is m[0] fixed?
    The metric is max|autoconv| / S^2.
    If we scale x by c, autoconv scales by c^2, and S scales by c.
    So max|autoconv| / S^2 is scale-invariant!
    (c^2 * max|autoconv|) / (c * S)^2 = max|autoconv| / S^2.
    
    So we can fix S = 1 (or m[0] = 1) without loss of generality.
    
    So the problem is:
    Minimize f(m) = max_k |IFFT(m^2)_k|
    Subject to:
      m[0] = 1
      m[k] >= 0
    
    This is a convex optimization problem.
    
    Let's implement this.
    """
    deadline = deadline
    K = _FFT_SIZE // 2 + 1  # Number of points in rfft
    
    # Variables: m[1..K-1]
    # m[0] is fixed to 1.
    
    def objective(m_vars):
        m = np.zeros(K)
        m[0] = 1.0
        m[1:] = m_vars
        
        # Ensure non-negativity (handled by bounds in minimize)
        # Compute autoconv
        # |X|^2 = m^2
        power = m * m
        # IFFT of power
        # Note: IFFT assumes input is full spectrum or rfft?
        # np.fft.irfft takes rfft input.
        # power is the rfft of |X|^2?
        # X is rfft. |X|^2 is not necessarily the rfft of autoconv?
        # autoconv = IFFT( X * conj(X) )
        # For real x, X[k] = conj(X[N-k]).
        # |X|^2 is symmetric.
        # So irfft(|X|^2) gives the autoconvolution.
        
        conv = np.fft.irfft(power, _FFT_SIZE)
        max_abs = np.max(np.abs(conv[:2 * N - 1]))
        return max_abs
    
    def gradient(m_vars):
        # Finite difference for simplicity, or analytical?
        # Analytical gradient of max|conv| is tricky due to max.
        # Use subgradient or smooth approximation.
        # Let's use smooth approximation: log-sum-exp of |conv|^2?
        # Or just use finite differences.
        
        eps = 1e-7
        grad = np.zeros_like(m_vars)
        for i in range(len(m_vars)):
            m_plus = m_vars.copy()
            m_plus[i] += eps
            val_plus = objective(m_plus)
            
            m_minus = m_vars.copy()
            m_minus[i] -= eps
            val_minus = objective(m_minus)
            
            grad[i] = (val_plus - val_minus) / (2 * eps)
        
        return grad
    
    # Initial guess: flat spectrum
    m_init = np.ones(K - 1) * 0.1
    
    # Bounds: m[k] >= 0
    bounds = [(0, None)] * (K - 1)
    
    try:
        result = minimize(
            objective,
            m_init,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
        )
        
        m_opt = np.zeros(K)
        m_opt[0] = 1.0
        m_opt[1:] = result.x
        
        # Construct X
        X = m_opt.astype(complex)
        
        # Construct x
        x = np.fft.irfft(X, _FFT_SIZE)[:N]
        
        return x, _metric(x)
        
    except Exception:
        return None, float('inf')

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
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
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
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Convex Spectral Optimization
    # This finds the theoretical lower bound for the max|autoconv| given the sum.
    x_spec, val_spec = _solve_convex_spectral(deadline - 10)
    
    if x_spec is not None and val_spec < float('inf'):
        best_x = x_spec
        best_val = val_spec
    else:
        # Fallback to ones
        best_x = np.ones(N)
        best_val = _metric(best_x)
    
    # Phase 2: Local Search
    # The convex solution is symmetric (even). Local search can break symmetry if it helps?
    # No, autoconv is determined by power spectrum. Breaking symmetry in time domain
    # changes the power spectrum if we don't adjust phases.
    # But wait, local search changes x directly.
    # Changing x changes |X|^2.
    # So local search can potentially find a better |X|^2 if the convex solution was not global?
    # But the convex solution IS the global minimum for the magnitude spectrum.
    # So local search should not improve the autoconvolution magnitude, unless there's a numerical issue
    # or the "real signal" constraint forces a suboptimal magnitude spectrum?
    # No, any magnitude spectrum m can be realized by a real signal (with appropriate phases).
    # So the convex solution is globally optimal for the metric.
    
    # Why would local search help?
    # Maybe the convex solver didn't converge fully?
    # Or maybe there are multiple local minima in the phase space that affect the "realness"?
    # No, we constructed x from m directly.
    
    # Let's run local search just in case, but expect little improvement.
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=50000)
    
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)

def _epigraph_objective(params):
    """
    Objective for epigraph optimization.
    params: [x (length N), t (scalar)]
    We want to minimize t such that |conv_k| <= t for all k != 0.
    To handle the division by (sum(x))^2, we normalize x such that sum(x)=1 
    during optimization? No, scaling is tricky.
    
    Let's define the objective as:
    t + lambda * (max(0, max(|conv[1:]|) - t))^2
    But we need to handle the denominator (sum(x))^2.
    
    Actually, the score is max(|conv[1:]|) / (sum(x))^2.
    Let S = sum(x). We want to minimize M / S^2.
    This is equivalent to minimizing M * (1/S^2).
    
    Let's use a variable transformation or just optimize the ratio directly?
    The ratio is not differentiable if S crosses 0, but we can assume S > 0 
    or S < 0. Let's assume we can scale x.
    
    Alternative: Fix sum(x) = 1?
    If we fix sum(x) = 1, then score = max(|conv[1:]|).
    However, the optimal solution might not have sum(x)=1.
    But note that if we scale x by c, conv scales by c^2 and S^2 scales by c^2.
    So the score is scale-invariant!
    Therefore, we can constrain sum(x) = 1 (or any non-zero constant) without loss of generality.
    
    So, let's optimize x subject to sum(x) = 1, minimizing max(|conv[1:]|).
    We can use a soft constraint or just project.
    
    Let's use the epigraph formulation on the normalized signal.
    We optimize x (length N) and t.
    Objective: t
    Constraints:
      sum(x) = 1
      -t <= conv_k(x) <= t for all k != 0
    
    Since scipy.optimize.minimize doesn't handle many inequality constraints well 
    for N=400 (799 constraints), we can use a penalty method or just 
    minimize a smooth approximation of the max.
    
    Let's use the Log-Sum-Exp approximation for the max, with a large beta.
    f(x) = (1/beta) * log( sum_{k!=0} exp(beta * |conv_k(x)|) )
    We want to minimize f(x) subject to sum(x)=1.
    
    We can use Lagrange multipliers or just penalize the sum deviation.
    Objective: f(x) + mu * (sum(x) - 1)^2
    
    Let's try this.
    """
    x = params[:N]
    t = params[N] # Not using t directly in this version, just minimizing soft max
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e12
    
    # Normalize to ensure we are comparing fairly? 
    # Since the score is scale invariant, we can just minimize the soft max of 
    # the normalized autoconv.
    # x_norm = x / s
    # conv_norm = conv(x_norm)
    # score = max(|conv_norm[1:]|)
    
    # Let's compute conv of x, then divide by s^2?
    # No, let's just work with x directly and add a penalty for sum(x) != 1.
    # If we don't fix sum, the optimizer will try to make x small to reduce conv, 
    # but the score is ratio. 
    # Wait, if I minimize max(|conv|) / sum(x)^2, and I scale x by 1/100, 
    # conv becomes 1/10000, sum^2 becomes 1/10000. Ratio is invariant.
    # So the optimizer has a flat valley along the scaling direction.
    # This makes gradient descent hard.
    
    # Better: Constrain sum(x) = 1.
    # We will use a penalty term.
    penalty = 100.0 * (s - 1.0)**2
    
    if abs(s) < 1e-6:
        return penalty + 1e10
        
    x_norm = x / s
    conv = _autoconv_full(x_norm)
    
    if len(conv) <= 1:
        return penalty
        
    abs_c = np.abs(conv[1:])
    
    # Soft max
    beta = 50.0
    m = np.max(abs_c)
    if m < 1e-15:
        soft_max = 0.0
    else:
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        
    return soft_max + penalty

def _gradient_descent_optimize(x_init, deadline, rng):
    """
    Optimize using gradient descent on the continuous signal values.
    """
    x = x_init.copy()
    # Normalize initial x
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    else:
        x = np.ones(N) / N
        
    # Use L-BFGS-B
    # We need to provide gradient? scipy can estimate it, but it's slow for N=400.
    # Let's try analytic gradient or just default.
    # Default is forward difference, which is O(N) function calls per step.
    # N=400, so 400 calls per step.
    # Each call does FFT (O(N log N)).
    # Should be fast enough.
    
    try:
        result = scipy.optimize.minimize(
            _epigraph_objective,
            x0=x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e10:
            x_opt = result.x
            s = np.sum(x_opt)
            if abs(s) > 1e-12:
                x_opt = x_opt / s
            score_opt = _score_offdiag(x_opt)
            return x_opt, score_opt
    except Exception:
        pass
        
    return x, _score_offdiag(x)

def _sign_projection(x):
    """
    Project continuous signal to signed steps (+1/-1).
    We want to find the closest signed sequence that preserves the "shape" 
    or just threshold at 0?
    Thresholding at 0 is simple.
    """
    return np.sign(x)

def _refine_signed(h, deadline, rng):
    """
    Refine the signed sequence using local search (bit flips).
    """
    n = len(h)
    current = h.copy()
    current_score = _score_offdiag(current)
    best_score = current_score
    best_h = current.copy()
    
    size = 1 << (2 * n - 1).bit_length()
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try flipping a random bit
        i = int(rng.integers(0, n))
        old_val = current[i]
        current[i] = -old_val
        
        s = np.sum(current)
        if abs(s) < 1e-12:
            new_score = np.inf
        else:
            H = np.fft.rfft(current, size)
            ac = np.fft.irfft(H * H, size)[:2 * n - 1]
            max_abs = np.max(np.abs(ac[1:]))
            new_score = max_abs / (s * s)
            
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_h = current.copy()
        else:
            current[i] = old_val
            no_improve += 1
            
    return best_h, best_score

def construct():
    t0 = time.time()
    # Safety margin: 150s limit. Use 140s.
    deadline = t0 + 140.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Hanning window
    candidates.append(np.hanning(N))
    
    # 2. Hamming window
    candidates.append(np.hamming(N))
    
    # 3. M-sequences
    def _m_sequence(n, k):
        polys = {
            6: [5, 3, 0],
            7: [6, 5, 3, 0],
            8: [7, 5, 3, 2, 0],
        }
        if k not in polys:
            return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
        
        taps = polys[k]
        state = [0] * k
        state[0] = 1
        seq = []
        for _ in range(n):
            fb = 0
            for t in taps:
                fb ^= state[t]
            for i in range(k - 1, 0, -1):
                state[i] = state[i - 1]
            state[0] = fb
            seq.append(1.0 if state[0] else -1.0)
        return np.array(seq, dtype=np.float64)
        
    for k in [6, 7, 8]:
        candidates.append(_m_sequence(N, k))
        
    # 4. Random
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Evaluate candidates
    best_score = np.inf
    best_h = np.ones(N)
    
    for h in candidates:
        if time.time() > deadline - 80:
            break
        s = _score_offdiag(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Phase 2: Continuous optimization from best candidate
    # We can run this on multiple candidates if time permits
    remaining_time = deadline - time.time()
    
    # Run on the best candidate
    if remaining_time > 30:
        x_cont, score_cont = _gradient_descent_optimize(best_h, time.time() + 25, rng)
        # Project to signed
        h_proj = _sign_projection(x_cont)
        # Refine the projected version
        h_ref, score_ref = _refine_signed(h_proj, time.time() + 15, rng)
        
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref
            
    # Also try optimizing from a few other good candidates?
    # Let's pick the top 3 from candidates
    scored_candidates = []
    for h in candidates:
        s = _score_offdiag(h)
        scored_candidates.append((s, h))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Try optimizing the 2nd and 3rd best if time remains
    for i in range(1, 3):
        if time.time() > deadline - 30:
            break
        s, h = scored_candidates[i]
        x_cont, score_cont = _gradient_descent_optimize(h, time.time() + 15, rng)
        h_proj = _sign_projection(x_cont)
        h_ref, score_ref = _refine_signed(h_proj, time.time() + 10, rng)
        
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref
            
    # Phase 3: Final refinement of the best signed sequence
    if time.time() < deadline - 5:
        best_h, best_score = _refine_signed(best_h, deadline - 5, rng)
        
    return best_h

# EVOLVE-BLOCK-END

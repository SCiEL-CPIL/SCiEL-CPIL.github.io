<<<<<<< SEARCH
def _coordinate_descent(x, max_iters=5000, step_scale=0.1):
    """
    Coordinate descent polish: adjust individual coordinates to reduce objective.
    """
    best_val = _objective(x)
    best_x = x.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                trial = best_x.copy()
                trial[j] += delta
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_x, best_val

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x
=======
def _psd_amplitude(N, bandwidth=0.5):
    """
    Compute prolate spheroidal sequence (PSD) amplitude spectrum.
    PSD maximizes energy concentration in both time and frequency bands.
    We approximate using the dominant eigenvector of the time-bandwidth matrix,
    or use a well-known approximation for the PSD.
    
    For practical purposes, we use a raised-cosine or Gaussian-like envelope
    that approximates PSD behavior. The key property is that it concentrates
    spectral energy in a narrow band, which helps minimize autoconvolution peaks.
    """
    k = np.arange(N)
    kf = (k - N//2) / N  # centered frequency
    
    # PSD approximation: use a smooth window that concentrates energy
    # A good approximation is the Dirichlet kernel squared or a Gaussian
    # For time-bandwidth product BT, the PSD has ~2BT significant coefficients
    
    BT = bandwidth * N  # time-bandwidth product
    # Use a Gaussian-like envelope approximating PSD
    sigma = 1.0 / np.sqrt(2 * BT) if BT > 0.1 else 1.0
    amp = np.exp(-kf**2 / (2 * sigma**2))
    
    # Normalize
    amp = amp / np.max(amp)
    return amp

def _spectral_phase_optimize(amp, init_phase=None, max_iters=1000, lr=0.01, beta=50.0):
    """
    Optimize the phase of the spectrum to minimize max|autoconv|/(sum x)^2.
    
    Given fixed amplitude spectrum amp, optimize phase theta such that
    x = IFFT(amp * exp(i*theta)) is real-valued and minimizes the objective.
    
    We use gradient descent on the phase, enforcing reality by taking the real part.
    """
    if init_phase is None:
        init_phase = np.zeros(N)
    
    theta = init_phase.copy().astype(float)
    
    def _x_from_phase(theta):
        """Compute real-valued x from amplitude and phase."""
        X = amp * np.exp(1j * theta)
        x = np.fft.ifft(X).real
        return x
    
    def _phase_objective(theta):
        """Objective: max|autoconv| / (sum x)^2, with smooth surrogate."""
        x = _x_from_phase(theta)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(theta)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-15:
            return 1e10, np.zeros_like(theta)
        
        # Soft-max
        shifted = beta * (abs_ac - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
        val = soft_max / (s * s)
        
        # Gradient with respect to theta
        # d(soft_max)/dtheta[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dtheta[j]
        # d(ac[n])/dtheta[j] = Im(conj(X[n-j]) * 1j * amp[j]) + Im(conj(X[n+j]) * 1j * amp[j])
        # This is complex; let's use automatic differentiation via finite differences
        # for the gradient, or use a simpler approach
        
        return val, None
    
    # Use gradient descent with finite-difference gradients (simpler and robust)
    best_theta = theta.copy()
    best_val = _objective(_x_from_phase(theta))
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Compute gradient via finite differences (sparse: only a few coordinates)
        grad = np.zeros_like(theta)
        eps = 1e-4
        
        # For efficiency, only compute gradient for a subset of coordinates per iteration
        # Use random subset
        subset_size = min(20, N)
        indices = np.random.choice(N, subset_size, replace=False)
        
        for j in indices:
            theta_plus = theta.copy()
            theta_plus[j] += eps
            val_plus = _objective(_x_from_phase(theta_plus))
            
            theta_minus = theta.copy()
            theta_minus[j] -= eps
            val_minus = _objective(_x_from_phase(theta_minus))
            
            grad[j] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient step
        theta_new = theta - lr * grad
        x_new = _x_from_phase(theta_new)
        val_new = _objective(x_new)
        
        if val_new < best_val:
            best_val = val_new
            best_theta = theta_new.copy()
            theta = theta_new
            # Adaptive learning rate
            lr = min(0.1, lr * 1.01)
        else:
            lr = max(0.001, lr * 0.9)
    
    return best_theta, best_val, _x_from_phase(best_theta)

def _spectral_design(a, tilt=0.0, mod=0.0):
    """
    Generate a step function from spectral parameters.
    """
    a = np.clip(a, 0.5, 3.0)
    tilt = np.clip(tilt, -1.0, 1.0)
    mod = np.clip(mod, 0.0, 1.0)
    
    k = np.arange(N)
    c = 0.5 + a
    kf = k / N
    
    mag = np.exp(-c * kf**2)
    mag = mag * np.exp(tilt * kf)
    
    X = mag.astype(complex)
    x = np.fft.ifft(X).real
    x = x / np.max(np.abs(x))
    
    if mod > 0.01:
        t = np.linspace(0, 1, N)
        x = x * (1.0 + mod * 0.5 * np.sin(2*np.pi*t))
        x = x / np.max(np.abs(x))
    
    return x
>>>>>>> REPLACE


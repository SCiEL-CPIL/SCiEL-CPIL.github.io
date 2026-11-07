"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.
    
    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin
    
    # Generate DRAG-style Gaussian seed
    t = np.linspace(0, T, N, endpoint=False) + dt/2
    sigma = T / 6.0
    center = T / 2.0
    
    gaussian = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
    gaussian = gaussian - edge_val
    gaussian = np.maximum(gaussian, 0)
    
    area = np.sum(gaussian) * dt
    if area > 0:
        gaussian = gaussian * (np.pi / area)
    
    gaussian_deriv = np.gradient(gaussian, dt)
    
    def make_pulse(sigma2, amp_scale=1.0):
        I = amp_scale * gaussian.copy()
        Q = amp_scale * sigma2 * gaussian_deriv.copy()
        
        # Check and fix constraints
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I *= factor
            Q *= factor
        
        # Check slew
        def check_slew(I, Q):
            slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI**2 + dQ**2) / dt
            slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
            return max(slew0, np.max(slews) if len(slews) > 0 else 0, slewN) <= slew_bound
        
        if not check_slew(I, Q):
            for _ in range(100):
                I *= 0.95
                Q *= 0.95
                if check_slew(I, Q):
                    break
        
        return I, Q
    
    # Estimate optimal DRAG parameter
    drive_amp = np.max(gaussian)
    if drive_amp > 0:
        optimal_sigma2 = abs(anharmonicity) / (2 * drive_amp * sigma)
    else:
        optimal_sigma2 = 0.2
    
    # Try a range of DRAG parameters and pick the best
    best_I, best_Q = None, None
    best_score = float('inf')
    
    for sigma2 in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        I, Q = make_pulse(sigma2)
        
        # Estimate infidelity: leakage ~ (chi/Omega)^2 * f(sigma2)
        # Better sigma2 (closer to optimal) reduces leakage
        score_est = abs(sigma2 - optimal_sigma2)
        
        if score_est < best_score:
            best_score = score_est
            best_I, best_Q = I, Q
    
    I, Q = best_I, best_Q
    
    # Final validity check
    amp = np.sqrt(I**2 + Q**2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"
    
    def check_slew(I, Q):
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI**2 + dQ**2) / dt
        slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        return max(slew0, np.max(slews) if len(slews) > 0 else 0, slewN) <= slew_bound
    
    assert check_slew(I, Q), "Slew rate constraint violated"
    
    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END

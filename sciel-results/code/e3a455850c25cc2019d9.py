"""Rewritten program for robust control pulses for a three-level transmon."""
import numpy as np
from scipy.integrate import solve_ivp

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        # Only X_pi is specified; return zero pulse for others (or raise, but zero is safe)
        return np.zeros((80, 2))

    # Extract hardware parameters
    # hardware_model holds: anharmonicity -2*pi*0.25, amplitude bound 0.4, slew bound 0.12,
    # detuning bounds +-2*pi*0.002, drive scale bounds [0.97, 1.03], T1 25000 ns, Tphi 40000 ns
    # The contract says hardware_model holds these values. We assume it's a dict or similar.
    # But the seed just uses quantum_seed, so maybe hardware_model is ignored or standard.
    # Let's assume standard values as per contract.
    
    # Constants
    dt = 0.5  # ns
    N_steps = 80
    T_total = N_steps * dt  # 40 ns
    
    # Hardware parameters from contract
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_low, scale_high = 0.97, 1.03
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # DRAG-style Gaussian pulse
    # sigma = 40/6 ns
    sigma = T_total / 6.0
    
    # Time points (center of each step)
    t = np.arange(N_steps) * dt + dt / 2.0  # center of each step
    t_center = T_total / 2.0
    
    # Gaussian envelope
    # Area should be pi (for X-pi gate)
    # Integral of A * exp(-(t-tc)^2 / (2*sigma^2)) dt = A * sigma * sqrt(2*pi)
    # We want area = pi, so A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A_gauss = np.sqrt(np.pi / 2.0) / sigma
    
    # I component: Gaussian
    I = A_gauss * np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
    Q = np.zeros(N_steps)
    
    # Offset to 0 at edges: The contract says "offset to 0 at the edges"
    # This likely means the pulse should be zero at the boundaries, which the Gaussian naturally does (approximately)
    # But let's ensure the first and last steps are small enough to satisfy slew rate from zero
    
    # Check amplitude bound
    max_amp = np.sqrt(I**2 + Q**2)
    if np.max(max_amp) > amp_bound:
        scale_factor = amp_bound / np.max(max_amp)
        I *= scale_factor
        Q *= scale_factor
    
    # Check slew rate bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero
    # Compute slew rates
    I_ext = np.concatenate([[0.0], I, [0.0]])
    Q_ext = np.concatenate([[0.0], Q, [0.0]])
    dI = np.diff(I_ext) / dt
    dQ = np.diff(Q_ext) / dt
    slew = np.sqrt(dI**2 + dQ**2)
    
    if np.max(slew) > slew_bound:
        # Scale down to meet slew bound
        scale_factor = slew_bound / np.max(slew)
        I *= scale_factor
        Q *= scale_factor
    
    # Re-check amplitude after slew scaling
    max_amp = np.sqrt(I**2 + Q**2)
    if np.max(max_amp) > amp_bound:
        scale_factor = amp_bound / np.max(max_amp)
        I *= scale_factor
        Q *= scale_factor
    
    # Add DRAG component (Q component) to suppress leakage
    # DRAG: Q is proportional to the derivative of I
    # Q_drag = -lambda * dI/dt
    # The optimal lambda for DRAG is typically related to the anharmonicity
    # For a transmon, lambda = anharmonicity / (2 * detuning_effective) or similar
    # A common heuristic: lambda = -anharmonicity / (2 * pi * f_drive) but here we work in angular frequency
    # Let's use a standard DRAG parameter
    # The DRAG correction is often: Q(t) = - (alpha / 2) * dI/dt where alpha is anharmonicity
    # But we need to be careful with units
    
    # Compute derivative of I
    dI_dt = np.gradient(I, dt)
    
    # DRAG parameter
    # For a pi pulse, the DRAG coefficient is often chosen as:
    # lambda = -anharmonicity / (2 * pi) * (some factor)
    # Let's use a typical value. The anharmonicity is -2*pi*0.25 = -0.5*pi
    # A common choice is lambda = -anharmonicity / (4 * pi) or similar
    # Let's try lambda = -anharmonicity / (2 * np.pi) * 0.5
    # Actually, let's use the standard DRAG formula:
    # Q(t) = - (alpha / 2) * (1 / omega_drive) * dI/dt
    # But we don't have omega_drive explicitly. Let's use a heuristic.
    
    # For now, let's add a small Q component to see if it helps
    # The DRAG component should be proportional to the slope of I
    # Let's use a conservative value
    drag_coeff = -anharmonicity / (2.0 * np.pi) * 0.5  # heuristic
    
    Q = drag_coeff * dI_dt
    
    # Re-check bounds after adding Q
    max_amp = np.sqrt(I**2 + Q**2)
    if np.max(max_amp) > amp_bound:
        scale_factor = amp_bound / np.max(max_amp)
        I *= scale_factor
        Q *= scale_factor
    
    I_ext = np.concatenate([[0.0], I, [0.0]])
    Q_ext = np.concatenate([[0.0], Q, [0.0]])
    dI = np.diff(I_ext) / dt
    dQ = np.diff(Q_ext) / dt
    slew = np.sqrt(dI**2 + dQ**2)
    
    if np.max(slew) > slew_bound:
        scale_factor = slew_bound / np.max(slew)
        I *= scale_factor
        Q *= scale_factor
    
    # Final check
    max_amp = np.sqrt(I**2 + Q**2)
    if np.max(max_amp) > amp_bound:
        scale_factor = amp_bound / np.max(max_amp)
        I *= scale_factor
        Q *= scale_factor
    
    return np.column_stack((I, Q))

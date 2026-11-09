"""Rewritten program for robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def _liouvillian(t, y, I, Q, detuning, scale, T1, Tphi, anharmonicity):
    """
    Compute the Liouvillian evolution for a 3-level system.
    y is a flattened density matrix (9 complex numbers).
    """
    rho = y.reshape(3, 3)
    
    # Construct operators
    a = np.diag(np.sqrt([1.0, 2.0]), 1)
    adag = a.conj().T
    n = adag @ a
    I_op = np.eye(3)
    
    # Hamiltonian
    H = detuning * n + 0.5 * anharmonicity * (n @ (n - I_op)) + \
        0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Liouvillian: d rho / dt = -i [H, rho] + sum (L rho L^dag - 0.5 {L^dag L, rho})
    d_rho_dt = -1j * (H @ rho - rho @ H)
    for L in [L1, L2]:
        Ldag = L.conj().T
        d_rho_dt += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
    
    return d_rho_dt.flatten()

def _simulate_pulse(I, Q, detuning, scale, T1, Tphi, anharmonicity, dt=0.5):
    """
    Simulate the pulse and return the final density matrix.
    """
    n_steps = len(I)
    psi0 = np.zeros(3, dtype=complex)
    
    # We simulate the density matrix evolution step by step
    # Each step is a constant drive for duration dt
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(n_steps):
        # Solve ODE for this step
        t_span = (0, dt)
        y0 = rho.flatten()
        
        sol = solve_ivp(_liouvillian, t_span, y0, method='DOP853', 
                       args=(I[i], Q[i], detuning, scale, T1, Tphi, anharmonicity),
                       rtol=1e-10, atol=1e-12)
        rho = sol.y[:, -1].reshape(3, 3)
    
    return rho

def _fidelity(rho, psi):
    """Compute fidelity F = <psi|rho|psi>"""
    return np.real(np.conj(psi) @ rho @ psi)

def _compute_score(I, Q, detuning, scale, T1, Tphi, anharmonicity):
    """Compute mean fidelity to X gate for the 4 test states."""
    # Test states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_iplus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_iminus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Target: X gate swaps |0> and |1>
    # So X|0> = |1>, X|1> = |0>, X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2), etc.
    # Fidelity to X means: for input state psi, we want output to be X|psi>
    
    fidelities = []
    for psi in [psi0, psi1, psi_plus, psi_minus, psi_iplus, psi_iminus]:
        rho = _simulate_pulse(I, Q, detuning, scale, T1, Tphi, anharmonicity)
        psi_target = np.zeros(3, dtype=complex)
        if np.allclose(psi, psi0):
            psi_target = psi1
        elif np.allclose(psi, psi1):
            psi_target = psi0
        elif np.allclose(psi, psi_plus):
            psi_target = psi_plus
        elif np.allclose(psi, psi_minus):
            psi_target = psi_minus
        elif np.allclose(psi, psi_iplus):
            psi_target = psi_iplus
        elif np.allclose(psi, psi_iminus):
            psi_target = psi_iminus
        
        F = _fidelity(rho, psi_target)
        fidelities.append(F)
    
    return np.mean(fidelities)

def _objective(params):
    """Objective function for optimization: minimize infidelity."""
    I = params[:80]
    Q = params[80:160]
    
    # Check amplitude constraint
    if np.any(np.sqrt(I**2 + Q**2) > 0.4 + 1e-6):
        return 1.0 + 1e6 * (np.max(np.sqrt(I**2 + Q**2)) - 0.4)
    
    # Check slew constraint
    for i in range(79):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        if np.sqrt(dI**2 + dQ**2) / 0.5 > 0.12 + 1e-6:
            return 1.0 + 1e6 * (np.sqrt(dI**2 + dQ**2) / 0.5 - 0.12)
    # From zero into step 1
    if np.sqrt(I[0]**2 + Q[0]**2) / 0.5 > 0.12 + 1e-6:
        return 1.0 + 1e6 * (np.sqrt(I[0]**2 + Q[0]**2) / 0.5 - 0.12)
    # From last step to zero
    if np.sqrt(I[-1]**2 + Q[-1]**2) / 0.5 > 0.12 + 1e-6:
        return 1.0 + 1e6 * (np.sqrt(I[-1]**2 + Q[-1]**2) / 0.5 - 0.12)
    
    # Compute score over 9 cases
    T1 = 25000.0
    Tphi = 40000.0
    anharmonicity = -2 * np.pi * 0.25
    
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            F = _compute_score(I, Q, det, sc, T1, Tphi, anharmonicity)
            infidelities.append(1.0 - F)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse."""
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt
    T_total = 40.0
    
    # Gaussian envelope with sigma = 40/6 ns
    sigma = 40.0 / 6.0
    center = T_total / 2.0
    
    # Area should be pi (for X-pi gate)
    # Integral of A * exp(-(t-center)^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi) = pi
    # So A = 1 / (sigma * sqrt(2*pi))
    A = 1.0 / (sigma * np.sqrt(2 * np.pi))
    
    # Offset to 0 at edges
    envelope = A * np.exp(-(t - center)**2 / (2 * sigma**2))
    envelope[0] = 0.0
    envelope[-1] = 0.0
    
    I = envelope
    Q = np.zeros(n_steps)
    
    return I, Q

def pulse(target_gate, hardware_model):
    """
    Generate control pulse for the specified target gate and hardware model.
    Returns (80, 2) array of I and Q amplitudes (rad/ns).
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Extract hardware parameters
    T1 = 25000.0
    Tphi = 40000.0
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    det_bound = 2 * np.pi * 0.002
    scale_low = 0.97
    scale_high = 1.03
    
    # Start with seed pulse
    I, Q = _make_seed_pulse()
    
    # Define bounds for optimization
    bounds = []
    for i in range(80):
        bounds.append((-amp_bound, amp_bound))  # I[i]
    for i in range(80):
        bounds.append((-amp_bound, amp_bound))  # Q[i]
    
    # Initial parameters
    x0 = np.concatenate([I, Q])
    
    # Optimize
    result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
    
    I_opt = result.x[:80]
    Q_opt = result.x[80:160]
    
    # Ensure constraints are satisfied
    # Clip amplitudes
    amp = np.sqrt(I_opt**2 + Q_opt**2)
    mask = amp > amp_bound
    if np.any(mask):
        scale_factor = amp_bound / amp[mask]
        I_opt[mask] *= scale_factor
        Q_opt[mask] *= scale_factor
    
    # Enforce slew rate by smoothing if needed
    for i in range(79):
        dI = I_opt[i+1] - I_opt[i]
        dQ = Q_opt[i+1] - Q_opt[i]
        slew = np.sqrt(dI**2 + dQ**2) / 0.5
        if slew > slew_bound:
            scale = slew_bound / slew
            I_opt[i+1] = I_opt[i] + dI * scale
            Q_opt[i+1] = Q_opt[i] + dQ * scale
    
    # Ensure first and last steps satisfy slew from/to zero
    slew0 = np.sqrt(I_opt[0]**2 + Q_opt[0]**2) / 0.5
    if slew0 > slew_bound:
        scale = slew_bound / slew0
        I_opt[0] *= scale
        Q_opt[0] *= scale
    
    sleven = np.sqrt(I_opt[-1]**2 + Q_opt[-1]**2) / 0.5
    if sleven > slew_bound:
        scale = slew_bound / sleven
        I_opt[-1] *= scale
        Q_opt[-1] *= scale
    
    return np.column_stack([I_opt, Q_opt])
# EVOLVE-BLOCK-END

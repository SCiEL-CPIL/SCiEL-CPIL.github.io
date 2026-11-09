"""Robust control pulses for a three-level transmon using basis expansion + DRAG."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
ANHARMONICITY = -2 * np.pi * 0.25
T1 = 25000.0
T2 = 40000.0  # Tphi

# Time grid
t = np.linspace(0, T_TOTAL, N_STEPS + 1)[:-1] + DT / 2  # center of each step
# Actually, let's use the step centers
t_centers = (np.arange(N_STEPS) + 0.5) * DT


def _liouvillian(t_step, I_val, Q_val, detuning, scale):
    """Build the Liouvillian for one time step."""
    # 3x3 matrices for 3-level system
    # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    adag = a.conj().T
    
    # n = a^dag a
    n = adag @ a  # diag(0, 1, 2)
    
    # Hamiltonian
    H = (detuning * n 
         + 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) 
         + 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag)))
    
    # Lindblad operators
    # L1 = a / sqrt(T1)
    L1 = a / np.sqrt(T1)
    # L2 = n * sqrt(2/Tphi) ... wait, the contract says "n sqrt(2 / Tphi)"
    # Let me re-read: "Lindblad operators a / sqrt(T1) and n sqrt(2 / Tphi)"
    # This likely means L2 = n * sqrt(2 / Tphi) or L2 = sqrt(2/Tphi) * n
    L2 = np.sqrt(2 / T2) * n
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We'll build it as a superoperator acting on vec(rho)
    # For efficiency, we'll use the commutator form directly in the ODE
    
    return H, L1, L2


def _simulate_pulse(I, Q, detuning, scale, method='RK45', rtol=1e-9, atol=1e-12):
    """Simulate the 3-level system with given pulse and return fidelity to X gate."""
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
    ]
    
    # Target: X gate maps |0>->|1>, |1>->|0>
    # So for each initial state, the target final state is X @ state
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    for psi0 in states:
        psi_target = X_gate @ psi0
        
        def ode_func(t_val, psi):
            idx = int((t_val / DT - 0.5))
            if idx < 0:
                idx = 0
            if idx >= N_STEPS:
                idx = N_STEPS - 1
            
            I_val = I[idx]
            Q_val = Q[idx]
            
            a = np.zeros((3, 3), dtype=complex)
            a[0, 1] = 1.0
            a[1, 2] = np.sqrt(2)
            adag = a.conj().T
            n = adag @ a
            
            H = (detuning * n 
                 + 0.5 * ANHARMONICITY * (n @ n - n) 
                 + 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag)))
            
            # dpsi/dt = -i H psi
            return -1j * H @ psi
        
        sol = solve_ivp(ode_func, [0, T_TOTAL], psi0, method=method, 
                       rtol=rtol, atol=atol, max_step=DT)
        if not sol.success:
            return 0.0  # Invalid
        
        psi_final = sol.y[:, -1]
        # Fidelity = |<psi_target|psi_final>|^2
        fid = np.abs(np.vdot(psi_target, psi_final)) ** 2
        fidelities.append(fid)
    
    return np.mean(fidelities)


def _compute_score(I, Q):
    """Compute the score for a given pulse (lower is better)."""
    # Check validity
    for i in range(N_STEPS):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-12:
            return 1e6  # Invalid
        
        # Slew rate check
        if i == 0:
            dI = I[0] - 0
            dQ = Q[0] - 0
        else:
            dI = I[i] - I[i-1]
            dQ = Q[i] - Q[i-1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return 1e6  # Invalid
    
    # Check slew from last step to zero
    dI = 0 - I[-1]
    dQ = 0 - Q[-1]
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-12:
        return 1e6  # Invalid
    
    # Evaluate for 9 cases
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _simulate_pulse(I, Q, det, sc)
            infidelities.append(1 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _basis_expansion_to_pulse(params):
    """Convert basis expansion parameters to I and Q pulse arrays."""
    # params = [a0, a1, a2, a3, b0, b1, b2, b3]
    # I(t) = sum_k a_k * phi_k(t)
    # Q(t) = sum_k b_k * phi_k(t)
    # phi_k are Hermite-Gaussian basis functions
    
    a_coeffs = params[:4]
    b_coeffs = params[4:8]
    
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    
    for k in range(4):
        # Hermite-Gaussian: H_k(t/sigma) * exp(-t^2/(2*sigma^2))
        sigma = T_TOTAL / 6.0  # ~6.67 ns
        # Normalize to have unit area for k=0
        t_norm = t_centers / sigma
        # Hermite polynomials
        if k == 0:
            Hk = np.ones_like(t_centers)
        elif k == 1:
            Hk = 2 * t_norm
        elif k == 2:
            Hk = 4 * t_norm**2 - 2
        else:
            Hk = 8 * t_norm**3 - 12 * t_norm
        
        gaussian = np.exp(-t_centers**2 / (2 * sigma**2))
        phi_k = Hk * gaussian
        
        # Normalize
        phi_k = phi_k / np.sum(np.abs(phi_k)) * N_STEPS  # Rough normalization
        
        I += a_coeffs[k] * phi_k
        Q += b_coeffs[k] * phi_k
    
    return I, Q


def _drag_seed():
    """Generate DRAG-style seed pulse."""
    sigma = T_TOTAL / 6.0
    t = t_centers
    gaussian = np.exp(-t**2 / (2 * sigma**2))
    
    # Area = pi
    area = np.sum(gaussian) * DT
    I = np.pi / area * gaussian
    
    # DRAG: Q is proportional to derivative of I
    dI_dt = np.gradient(I, DT)
    Q = -0.1 * dI_dt  # Small DRAG coefficient
    
    # Ensure amplitude and slew constraints
    I = np.clip(I, -AMPLITUDE_BOUND, AMPLITUDE_BOUND)
    Q = np.clip(Q, -AMPLITUDE_BOUND, AMPLITUDE_BOUND)
    
    # Check slew
    for i in range(1, N_STEPS):
        slew_I = abs(I[i] - I[i-1]) / DT
        slew_Q = abs(Q[i] - Q[i-1]) / DT
        if slew_I > SLEW_BOUND:
            I[i] = I[i-1] + SLEW_BOUND * DT * np.sign(I[i] - I[i-1])
        if slew_Q > SLEW_BOUND:
            Q[i] = Q[i-1] + SLEW_BOUND * DT * np.sign(Q[i] - Q[i-1])
    
    # First step from zero
    if abs(I[0]) / DT > SLEW_BOUND:
        I[0] = SLEW_BOUND * DT * np.sign(I[0])
    if abs(Q[0]) / DT > SLEW_BOUND:
        Q[0] = SLEW_BOUND * DT * np.sign(Q[0])
    
    # Last step to zero
    if abs(I[-1]) / DT > SLEW_BOUND:
        I[-1] = SLEW_BOUND * DT * np.sign(I[-1])
    if abs(Q[-1]) / DT > SLEW_BOUND:
        Q[-1] = SLEW_BOUND * DT * np.sign(Q[-1])
    
    return I, Q


def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 300  # 300 second budget
    
    # Generate seed
    I_seed, Q_seed = _drag_seed()
    best_score = _compute_score(I_seed, Q_seed)
    best_I, best_Q = I_seed.copy(), Q_seed.copy()
    
    # Use differential evolution on basis expansion parameters
    # params = [a0, a1, a2, a3, b0, b1, b2, b3]
    bounds = [(-0.5, 0.5)] * 8
    
    def objective(params):
        I, Q = _basis_expansion_to_pulse(params)
        return _compute_score(I, Q)
    
    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=10, 
            tol=1e-4, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=True
        )
        
        if result.fun < best_score:
            I_opt, Q_opt = _basis_expansion_to_pulse(result.x)
            best_score = result.fun
            best_I, best_Q = I_opt, Q_opt
        
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return np.column_stack([best_I, best_Q])
    
    # Refine with local optimization on the full pulse
    from scipy.optimize import minimize
    
    def full_objective(params):
        I = params[:80]
        Q = params[80:]
        return _compute_score(I, Q)
    
    x0 = np.concatenate([best_I, best_Q])
    
    try:
        result2 = minimize(full_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8})
        
        if result2.fun < best_score:
            I_ref = result2.x[:80]
            Q_ref = result2.x[80:]
            best_score = result2.fun
            best_I, best_Q = I_ref, Q_ref
            
    except Exception:
        pass
    
    return np.column_stack([best_I, best_Q])

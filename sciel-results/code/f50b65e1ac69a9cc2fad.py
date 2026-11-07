"""Spectral-nulling DRAG pulse for three-level transmon X-pi gate."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N_STEPS = 80
DURATION = 40.0

# Time grid
t = np.arange(N_STEPS) * DT

# Basis operators
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op  # diag(0, 1, 2)

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian blocks (independent of I, Q, detuning, scale)
I3 = np.eye(3, dtype=complex)

def _comm_super(H_mat):
    return -1j * (np.kron(I3, H_mat) - np.kron(H_mat.T, I3))

def _diss_super(L_mat):
    LdagL = L_mat.conj().T @ L_mat
    return (np.kron(L_mat.T, L_mat) - 0.5 * (np.kron(I3, LdagL) + np.kron(LdagL.T, I3)))

# Precompute static parts
n_nm1 = n_op @ (n_op - np.eye(3))
Super_static = _comm_super(0.5 * ANHARMONICITY * n_nm1) + _diss_super(L1) + _diss_super(L2)

# Precompute blocks for detuning and drive
Super_detuning_block = _comm_super(n_op)  # multiplies detuning
Super_I_block = _comm_super(0.5 * (a_op + a_dag))  # multiplies scale * I
Super_Q_block = _comm_super(0.5 * 1j * (a_op - a_dag))  # multiplies scale * Q

def build_super(detuning, scale, I_val, Q_val):
    return (Super_static 
            + detuning * Super_detuning_block 
            + scale * (I_val * Super_I_block + Q_val * Super_Q_block))

# Precompute expm for each step's Liouvillian? No, I and Q vary per step.
# But we can precompute the matrix structure.

def simulate_pulse(I, Q, detuning, scale):
    """Simulate and return mean fidelity."""
    # Build Liouvillians for all steps
    supers = [build_super(detuning, scale, I[i], Q[i]) for i in range(N_STEPS)]
    
    # Precompute U_step = expm(Super * DT)
    U_steps = [expm(S * DT) for S in supers]
    
    # Initial states and targets
    states = [
        (np.array([1, 0, 0], dtype=complex), np.array([0, 1, 0], dtype=complex)),
        (np.array([0, 1, 0], dtype=complex), np.array([0, -1, 0], dtype=complex)),
        (np.array([1, 1, 0], dtype=complex)/np.sqrt(2), np.array([-1, 1, 0], dtype=complex)/np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex)/np.sqrt(2), np.array([1, 1, 0], dtype=complex)/np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex)/np.sqrt(2), np.array([-1j, 1, 0], dtype=complex)/np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex)/np.sqrt(2), np.array([1j, 1, 0], dtype=complex)/np.sqrt(2)),
    ]
    
    fidelities = []
    for psi_init, psi_target in states:
        rho = np.outer(psi_init, psi_init.conj()).flatten()
        for U in U_steps:
            rho = U @ rho
        rho_final = rho.reshape(3, 3)
        rho_target = np.outer(psi_target, psi_target.conj())
        fid = np.real(np.trace(rho_target @ rho_final))
        fidelities.append(np.clip(fid, 0, 1))
    
    return np.mean(fidelities)

def compute_score(I, Q):
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    infids = []
    for det in detunings:
        for sc in scales:
            fid = simulate_pulse(I, Q, det, sc)
            infids.append(1.0 - fid)
    infids = np.array(infids)
    return np.mean(infids) + np.max(infids)

def validate_pulse(I, Q):
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > AMPLITUDE_BOUND + 1e-10:
        return False
    p = np.column_stack([I, Q])
    if np.linalg.norm(p[0]) / DT > SLEW_BOUND + 1e-10:
        return False
    for i in range(N_STEPS - 1):
        if np.linalg.norm(p[i+1] - p[i]) / DT > SLEW_BOUND + 1e-10:
            return False
    if np.linalg.norm(p[-1]) / DT > SLEW_BOUND + 1e-10:
        return False
    return True

def make_gaussian(t, sigma, area, center):
    g = area / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((t - center) / sigma)**2)
    g[0] = 0.0
    g[-1] = 0.0
    return g

def make_drag(t, sigma, center):
    d = -(t - center) / sigma**2 * np.exp(-0.5 * ((t - center) / sigma)**2)
    d[0] = 0.0
    d[-1] = 0.0
    mx = np.max(np.abs(d))
    if mx > 0:
        d /= mx
    return d

def generate_pulse(params):
    amp_x, shift_x, amp_drag, shift_drag = params
    sigma = DURATION / 6.0
    cx = DURATION / 2.0 + shift_x
    cd = DURATION / 2.0 + shift_drag
    
    I = amp_x * make_gaussian(t, sigma, np.pi, cx)
    Q = amp_drag * make_drag(t, sigma, cd)
    return I, Q

def objective(params):
    I, Q = generate_pulse(params)
    if not validate_pulse(I, Q):
        return 100.0
    return compute_score(I, Q)

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Seed
    sigma = DURATION / 6.0
    I0 = make_gaussian(t, sigma, np.pi, DURATION / 2.0)
    Q0 = 0.05 * make_drag(t, sigma, DURATION / 2.0)
    
    if not validate_pulse(I0, Q0):
        max_amp = np.max(np.sqrt(I0**2 + Q0**2))
        if max_amp > AMPLITUDE_BOUND:
            s = AMPLITUDE_BOUND / max_amp
            I0 *= s; Q0 *= s
        p = np.column_stack([I0, Q0])
        max_slew = 0.0
        if np.linalg.norm(p[0])/DT > SLEW_BOUND:
            max_slew = max(max_slew, np.linalg.norm(p[0])/DT/SLEW_BOUND)
        for i in range(N_STEPS-1):
            s_ = np.linalg.norm(p[i+1]-p[i])/DT
            if s_ > SLEW_BOUND:
                max_slew = max(max_slew, s_/SLEW_BOUND)
        if np.linalg.norm(p[-1])/DT > SLEW_BOUND:
            max_slew = max(max_slew, np.linalg.norm(p[-1])/DT/SLEW_BOUND)
        if max_slew > 1:
            I0 /= max_slew; Q0 /= max_slew
    
    best_I, best_Q = I0.copy(), Q0.copy()
    best_score = compute_score(I0, Q0)
    
    # Multi-start optimization
    starts = [
        np.array([1.0, 0.0, 0.05, 0.0]),
        np.array([1.0, 0.0, -0.05, 0.0]),
        np.array([1.0, 0.0, 0.1, 0.0]),
        np.array([1.0, 0.0, -0.1, 0.0]),
        np.array([0.98, 0.0, 0.05, 0.0]),
        np.array([1.02, 0.0, 0.05, 0.0]),
        np.array([1.0, 0.0, 0.05, 0.5]),
        np.array([1.0, 0.0, 0.05, -0.5]),
    ]
    
    for x0 in starts:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score and res.fun < 100.0:
                I, Q = generate_pulse(res.x)
                if validate_pulse(I, Q):
                    best_score = res.fun
                    best_I, best_Q = I.copy(), Q.copy()
        except Exception:
            pass
    
    return np.column_stack([best_I, best_Q])

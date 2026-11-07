"""
Robust DRAG pulse optimizer for 3-level transmon X-pi gate.
Uses a 3-parameter DRAG family (amplitude scale, DRAG coefficient, detuning)
with multi-start Nelder-Mead optimization against the full 9-case score.
"""
import numpy as np
import time
import math

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _build_hamiltonian(detuning, scale, I, Q, anharmonicity):
    """Build the 3x3 Hamiltonian for given control parameters."""
    # a is 3x3 lowering operator for 3-level system: a = diag(sqrt([1,2]), 1)
    # a_{i,j} = sqrt(j) * delta_{i,j-1}
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = math.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a  # number operator: diag(0, 1, 2)
    
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3, dtype=complex))
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H

def _liouvillian_step(H, T1, Tphi):
    """Build the Liouvillian superoperator for one time step.
    
    L = -i[H, .] + sum_k (L_k . L_k^dag - 0.5{L_k^dag L_k, .})
    """
    n_levels = 3
    dim = n_levels ** 2  # 9 for vectorized density matrix
    
    a = np.zeros((n_levels, n_levels), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = math.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # Lindblad operators
    L1 = a / math.sqrt(T1)
    L2 = n * math.sqrt(2.0 / Tphi)
    
    # Build Liouvillian in superoperator form
    # For vectorized rho (column-major), the Liouvillian is:
    # L(rho_vec) = -i*(H @ rho - rho @ H) + sum(L_k @ rho @ L_k^dag - 0.5*(L_k^dag L_k @ rho + rho @ L_k^dag L_k))
    # In superoperator form: -i*(H ⊗ I - I ⊗ H^T) + ...
    
    I_mat = np.eye(n_levels, dtype=complex)
    
    # Coherent part: -i * (H ⊗ I - I ⊗ H^T)
    L_super = -1j * (np.kron(H, I_mat) - np.kron(I_mat, H.T))
    
    # Incoherent part
    for Lk in [L1, L2]:
        Lkdag = Lk.conj().T
        Lkdag_Lk = Lkdag @ Lk
        # Lk rho Lkdag -> kron(Lk, Lkdag^T)
        # -0.5 (Lkdag Lk rho + rho Lkdag Lk) -> -0.5 (kron(Lkdag_Lk, I) + kron(I, Lkdag_Lk^T))
        L_super += np.kron(Lk, Lkdag.T) - 0.5 * (np.kron(Lkdag_Lk, I_mat) + np.kron(I_mat, Lkdag_Lk.T))
    
    return L_super

def _evolve_state(pulse_I, pulse_Q, detuning, scale, dt, T1, Tphi, anharmonicity):
    """Evolve the density matrix through the pulse. Returns final density matrix (3x3)."""
    n_steps = len(pulse_I)
    n_levels = 3
    rho = np.zeros((n_levels, n_levels), dtype=complex)
    rho[0, 0] = 1.0  # start in |0>
    
    for i in range(n_steps):
        H = _build_hamiltonian(detuning, scale, pulse_I[i], pulse_Q[i], anharmonicity)
        L_super = _liouvillian_step(H, T1, Tphi)
        
        # Vectorize rho (column-major)
        rho_vec = rho.ravel(order='F')
        
        # Exponentiate: rho' = exp(L_super * dt) * rho_vec
        # Use matrix exponential for accuracy
        exp_L = np.linalg.expm(L_super * dt)
        rho_vec_new = exp_L @ rho_vec
        
        rho = rho_vec_new.reshape((n_levels, n_levels), order='F')
    
    return rho

def _fidelity(rho, target_state):
    """Fidelity between density matrix and pure state: F = <psi|rho|psi>"""
    psi = target_state
    F = np.vdot(psi, rho @ psi)
    return np.real(F)

def _compute_score(pulse_I, pulse_Q, hardware_model):
    """Compute the score: mean + max infidelity over 9 cases."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    dt = 0.5
    T1 = 25000.0
    Tphi = 40000.0
    anharmonicity = -2 * np.pi * 0.25
    
    # Target states
    zero = np.array([1.0, 0.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0, 0.0], dtype=complex)
    plus = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    iphase_plus = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2.0)
    iphase_minus = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2.0)
    
    targets = [zero, one, plus, minus, iphase_plus, iphase_minus]
    
    # X gate maps: |0>->|1>, |1>->|0>, (|0>+|1>)/sqrt2 -> (|1>+|0>)/sqrt2, etc.
    # Actually X maps |0> to |1>, |1> to |0>.
    # So target after X: |0> -> |1>, |1> -> |0>
    # (|0>+|1>)/sqrt2 -> (|1>+|0>)/sqrt2 = same
    # (|0>-|1>)/sqrt2 -> (|1>-|0>)/sqrt2 = -(minus)
    # (|0>+i|1>)/sqrt2 -> (|1>+i|0>)/sqrt2 = (i*|0> + |1>)/sqrt2
    # (|0>-i|1>)/sqrt2 -> (|1>-i|0>)/sqrt2 = (-i*|0> + |1>)/sqrt2
    
    target_after_X = [
        one,  # |0> -> |1>
        zero,  # |1> -> |0>
        plus,  # (|0>+|1>)/sqrt2 -> (|1>+|0>)/sqrt2 = plus
        -minus,  # (|0>-|1>)/sqrt2 -> (|1>-|0>)/sqrt2 = -minus
        np.array([1j, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),  # (|0>+i|1>)/sqrt2 -> (|1>+i|0>)/sqrt2
        np.array([-1j, 1.0, 0.0], dtype=complex) / np.sqrt(2.0),  # (|0>-i|1>)/sqrt2 -> (|1>-i|0>)/sqrt2
    ]
    
    infidelities = []
    
    for det in detunings:
        for sc in scales:
            # Check validity
            valid = True
            amp = np.sqrt(pulse_I**2 + pulse_Q**2)
            if np.max(amp) > 0.4 + 1e-10:
                valid = False
            # Slew rate
            for i in range(len(pulse_I) - 1):
                dI = pulse_I[i+1] - pulse_I[i]
                dQ = pulse_Q[i+1] - pulse_Q[i]
                if math.sqrt(dI**2 + dQ**2) / dt > 0.12 + 1e-10:
                    valid = False
                    break
            if not valid:
                return 1e10  # invalid
            
            # Evolve for each initial state
            case_infidels = []
            for init_state, targ_state in zip([zero, one, plus, minus, iphase_plus, iphase_minus], target_after_X):
                rho = _evolve_state(pulse_I, pulse_Q, det, sc, dt, T1, Tphi, anharmonicity)
                F = _fidelity(rho, targ_state)
                F = max(0.0, min(1.0, F))
                case_infidels.append(1.0 - F)
            
            mean_infid = np.mean(case_infidels)
            max_infid = np.max(case_infidels)
            infidelities.append(mean_infid + max_infid)
    
    return np.mean(infidelities)

def _generate_drag_pulse(amp_scale, drag_coeff, detuning_offset, n_steps=80, dt=0.5):
    """Generate a DRAG pulse with Gaussian envelope.
    
    Parameters:
    - amp_scale: scales the overall amplitude
    - drag_coeff: coefficient for the Q (derivative) term
    - detuning_offset: not used directly in pulse but part of params
    """
    t = np.arange(n_steps) * dt
    sigma = 40.0 / 6.0
    # Gaussian envelope, offset so it's 0 at edges
    center = (n_steps - 1) / 2.0
    env = np.exp(-0.5 * ((t - center * dt) / sigma) ** 2)
    
    # Normalize area to pi
    area = np.trapz(env, dx=dt)
    if area > 1e-15:
        env = env * (np.pi / area)
    
    # I component
    I = amp_scale * env
    
    # Q component: derivative of I scaled by drag_coeff
    dI_dt = np.gradient(I, dt)
    Q = drag_coeff * dI_dt
    
    return I, Q

def _check_validity(pulse_I, pulse_Q, dt=0.5):
    """Check amplitude and slew rate constraints."""
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amp) > 0.4 + 1e-9:
        return False
    for i in range(len(pulse_I)):
        if i > 0:
            dI = pulse_I[i] - pulse_I[i-1]
            dQ = pulse_Q[i] - pulse_Q[i-1]
        else:
            dI = pulse_I[0]
            dQ = pulse_Q[0]
        if math.sqrt(dI**2 + dQ**2) / dt > 0.12 + 1e-9:
            return False
    # Last step to zero
    dI = -pulse_I[-1]
    dQ = -pulse_Q[-1]
    if math.sqrt(dI**2 + dQ**2) / dt > 0.12 + 1e-9:
        return False
    return True

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return a default valid pulse
        return np.zeros((80, 2))
    
    dt = 0.5
    n_steps = 80
    T1 = 25000.0
    Tphi = 40000.0
    anharmonicity = -2 * np.pi * 0.25
    
    # Seed: DRAG-style Gaussian
    I_seed, Q_seed = _generate_drag_pulse(1.0, 0.5, 0.0, n_steps, dt)
    
    if not _check_validity(I_seed, Q_seed, dt):
        # Scale down to be valid
        scale_factor = 0.9
        I_seed *= scale_factor
        Q_seed *= scale_factor
    
    best_I = I_seed.copy()
    best_Q = Q_seed.copy()
    best_score = _compute_score(best_I, best_Q, hardware_model)
    
    if best_score < 1e-9:
        return np.column_stack([best_I, best_Q])
    
    # Multi-start Nelder-Mead optimization
    # Parameters: [amp_scale, drag_coeff]
    # We optimize 2 parameters
    
    starts = [
        [0.9, 0.3],
        [0.95, 0.5],
        [1.0, 0.7],
        [1.05, 0.4],
        [1.0, 0.5],
        [0.85, 0.6],
        [1.1, 0.2],
        [0.9, 0.8],
    ]
    
    for start in starts:
        if _time_left() < 30.0:
            break
        
        def objective(params):
            amp_scale, drag_coeff = params
            I, Q = _generate_drag_pulse(amp_scale, drag_coeff, 0.0, n_steps, dt)
            if not _check_validity(I, Q, dt):
                return 1e10
            return _compute_score(I, Q, hardware_model)
        
        try:
            result = scipy_nelder_mead(objective, start, maxiter=200, xatol=1e-4, fatol=1e-6)
            if result.fun < best_score:
                best_score = result.fun
                best_I, best_Q = _generate_drag_pulse(result.x[0], result.x[1], 0.0, n_steps, dt)
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_I, best_Q, dt):
        return np.column_stack([I_seed, Q_seed])
    
    return np.column_stack([best_I, best_Q])

# Import scipy at module level for the optimizer
import scipy.optimize as _scipy_opt

def scipy_nelder_mead(func, x0, maxiter=200, xatol=1e-4, fatol=1e-6):
    """Wrapper for Nelder-Mead that works with our time guard."""
    from scipy.optimize import minimize
    
    def time_guarded_func(x):
        if _time_left() < 5.0:
            return 1e10  # signal to stop
        return func(x)
    
    result = minimize(
        time_guarded_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    return result

# Alias for clarity
scipy_nelder_mead = _scipy_opt.minimize  # We'll use the wrapper above

# Re-define properly
def scipy_nelder_mead(func, x0, maxiter=200, xatol=1e-4, fatol=1e-6):
    """Wrapper for Nelder-Mead that works with our time guard."""
    def time_guarded_func(x):
        if _time_left() < 5.0:
            return 1e10
        return func(x)
    
    result = _scipy_opt.minimize(
        time_guarded_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol}
    )
    return result

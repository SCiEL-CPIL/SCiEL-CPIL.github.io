"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_DEADLINE = 300.0  # seconds

# ---- Fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25  # anharmonicity
_AMP_BOUND = 0.4
_SLEW_BOUND = 0.12
_DET_BOUNDS = 2.0 * np.pi * 0.002
_SCALE_BOUNDS = (0.97, 1.03)
_T1 = 25000.0
_TPHI = 40000.0

# 3-level transmon operators
# a = diag(sqrt([1, 2]), 1) as stated
_A = np.diag([1.0, np.sqrt(2.0), 1.0]).astype(complex)
_ADAG = _A.conj().T
_N_OP = _ADAG @ _A  # number operator

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _N_OP * np.sqrt(2.0 / _TPHI)
_L1DAG = _L1.conj().T
_L2DAG = _L2.conj().T

# Precompute Liouvillian superoperators for the 9 cases
_DET_GRID = np.array([-_DET_BOUNDS, 0.0, _DET_BOUNDS])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])

# Target: X gate (pi rotation) on the qubit subspace
# X = |0><1| + |1><0|
_X_TARGET = np.zeros((3, 3), dtype=complex)
_X_TARGET[0, 1] = 1.0
_X_TARGET[1, 0] = 1.0


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian matrix for one step given detuning, scale, I, Q."""
    # Hamiltonian
    # H = detuning * n + 0.5 * anh * n*(n-1) + 0.5 * scale * (I*(a + a^dag) + i*Q*(a - a^dag))
    H = (detuning * _N_OP + 0.5 * _ANH * (_N_OP @ (_N_OP - np.eye(3))) +
         0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG)))
    
    # Liouvillian: d/dt rho = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # In vectorized form: L = -i (H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5 (L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T))
    # Actually, let's use the standard superoperator form:
    # vec(d rho/dt) = L_vec * vec(rho)
    # where L_vec = -i (H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5 (L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T))
    
    I3 = np.eye(3, dtype=complex)
    H_vec = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    L12 = _L1.conj().T @ _L1
    L22 = _L2.conj().T @ _L2
    
    diss = (np.kron(_L1, _L1.conj()) - 0.5 * (np.kron(L12, I3) + np.kron(I3, L12.T)) +
            np.kron(_L2, _L2.conj()) - 0.5 * (np.kron(L22, I3) + np.kron(I3, L22.T)))
    
    return H_vec + diss


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix (vectorized)."""
    # Initial state: |0><0|
    psi0 = np.zeros(3, dtype=complex)
    psi0[0] = 1.0
    rho0 = np.outer(psi0, psi0.conj())
    rho_vec = rho0.flatten().copy()
    
    for i in range(_N):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
        # Exponentiate: rho_{k+1} = exp(L * dt) * rho_k
        Ldt = L * _DT
        rho_vec = expm(Ldt) @ rho_vec
    
    return rho_vec.reshape(3, 3)


def _fidelity(rho, target):
    """Compute fidelity F = <psi_target|rho|psi_target> for pure target."""
    # target is a 3x3 matrix, treat as |psi><psi|
    # F = Tr(rho @ target) for pure target
    return np.real(np.trace(rho @ target))


def _compute_score(pulse):
    """Compute the 9-case infidelity score."""
    # Check constraints
    for i in range(_N):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > _AMP_BOUND + 1e-9:
            return None  # invalid
    
    # Slew rate constraints
    prev = np.array([0.0, 0.0])
    for i in range(_N):
        curr = pulse[i]
        slew = np.linalg.norm(curr - prev) / _DT
        if slew > _SLEW_BOUND + 1e-9:
            return None
        prev = curr
    # Last step to zero
    slew = np.linalg.norm(-prev) / _DT
    if slew > _SLEW_BOUND + 1e-9:
        return None
    
    fidelities = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            rho_final = _simulate_pulse(pulse, det, scale)
            # Fidelity to X gate: mean over 5 initial states
            f_vals = []
            # |0>
            psi0 = np.zeros(3, dtype=complex); psi0[0] = 1.0
            # |1>
            psi1 = np.zeros(3, dtype=complex); psi1[1] = 1.0
            # (|0>+|1>)/sqrt(2)
            psi_plus = (psi0 + psi1) / np.sqrt(2)
            # (|0>-|1>)/sqrt(2)
            psi_minus = (psi0 - psi1) / np.sqrt(2)
            # (|0>+i|1>)/sqrt(2)
            psi_plus_i = (psi0 + 1j * psi1) / np.sqrt(2)
            # (|0>-i|1>)/sqrt(2)
            psi_minus_i = (psi0 - 1j * psi1) / np.sqrt(2)
            
            for psi in [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]:
                # After applying pulse, state is rho_final @ |psi><psi|
                # Fidelity to X|psi>
                target_state = _X_TARGET @ psi
                target_rho = np.outer(target_state, target_state.conj())
                f = np.real(np.trace(rho_final @ target_rho))
                f_vals.append(f)
            
            mean_f = np.mean(f_vals)
            infid = 1.0 - mean_f
            fidelities.append(infid)
    
    fidelities = np.array(fidelities)
    return np.mean(fidelities) + np.max(fidelities)


def _make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse."""
    pulse = np.zeros((_N, 2))
    t = np.arange(_N) * _DT
    # Gaussian with sigma = 40/6 ns
    sigma = 40.0 / 6.0
    # Area = pi
    # Gaussian: A * exp(-t^2 / (2*sigma^2)), integral = A * sigma * sqrt(2*pi) = pi
    A = pi_val / (sigma * np.sqrt(2.0 * np.pi))
    pi_val = np.pi
    A = pi_val / (sigma * np.sqrt(2.0 * np.pi))
    
    # Offset so that edges are 0
    # Actually, the task says "offset to 0 at the edges"
    # Let's center the Gaussian
    t_center = _T / 2.0
    gauss = A * np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
    
    # Ensure amplitude constraint
    max_amp = np.max(np.abs(gauss))
    if max_amp > _AMP_BOUND:
        gauss *= _AMP_BOUND / max_amp
    
    pulse[:, 0] = gauss  # I component
    pulse[:, 1] = 0.0    # Q component
    
    return pulse


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((_N, 2))
    
    best_pulse = _make_seed_pulse()
    best_score = _compute_score(best_pulse)
    
    if best_score is None:
        return best_pulse
    
    # Try to improve with simple gradient-free optimization
    # Use Nelder-Mead on a reduced parameterization
    if _time_remaining() < 10.0:
        return best_pulse
    
    # Parameterize: scale factor, DRAG Q component amplitude, detuning compensation
    # Simple local search
    
    def _score_from_params(params):
        scale = params[0]
        drag_amp = params[1]
        # Modify the seed pulse
        p = best_pulse.copy()
        p[:, 0] *= scale
        # Add Q component (DRAG)
        # DRAG Q is proportional to derivative of I
        I = p[:, 0]
        dI = np.zeros(_N)
        for i in range(1, _N):
            dI[i] = (I[i] - I[i-1]) / _DT
        p[:, 1] = drag_amp * dI
        
        # Enforce constraints
        for i in range(_N):
            amp = np.sqrt(p[i, 0]**2 + p[i, 1]**2)
            if amp > _AMP_BOUND:
                factor = _AMP_BOUND / amp
                p[i] *= factor
        
        # Check slew
        valid = True
        prev = np.array([0.0, 0.0])
        for i in range(_N):
            slew = np.linalg.norm(p[i] - prev) / _DT
            if slew > _SLEW_BOUND:
                valid = False
                break
            prev = p[i]
        if np.linalg.norm(-prev) / _DT > _SLEW_BOUND:
            valid = False
        
        if not valid:
            return 1e10
        
        s = _compute_score(p)
        return s if s is not None else 1e10
    
    # Nelder-Mead
    from scipy.optimize import minimize
    
    x0 = np.array([1.0, 0.0])
    
    try:
        result = minimize(_score_from_params, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_score:
            params = result.x
            scale = params[0]
            drag_amp = params[1]
            p = best_pulse.copy()
            p[:, 0] *= scale
            I = p[:, 0]
            dI = np.zeros(_N)
            for i in range(1, _N):
                dI[i] = (I[i] - I[i-1]) / _DT
            p[:, 1] = drag_amp * dI
            
            # Enforce constraints
            for i in range(_N):
                amp = np.sqrt(p[i, 0]**2 + p[i, 1]**2)
                if amp > _AMP_BOUND:
                    factor = _AMP_BOUND / amp
                    p[i] *= factor
            
            # Verify validity
            valid = True
            prev = np.array([0.0, 0.0])
            for i in range(_N):
                slew = np.linalg.norm(p[i] - prev) / _DT
                if slew > _AMP_BOUND + 1e-9:
                    valid = False
                    break
                prev = p[i]
            if np.linalg.norm(-prev) / _DT > _SLEW_BOUND + 1e-9:
                valid = False
            
            if valid:
                s = _compute_score(p)
                if s is not None and s < best_score:
                    best_pulse = p
                    best_score = s
    except Exception:
        pass
    
    if _time_remaining() < 10.0:
        return best_pulse
    
    # Try a few more parameterizations
    # Different sigma values
    for sigma_mult in [0.8, 1.2, 1.5]:
        if _time_remaining() < 15.0:
            break
        
        sigma = (40.0 / 6.0) * sigma_mult
        t = np.arange(_N) * _DT
        t_center = _T / 2.0
        A = np.pi / (sigma * np.sqrt(2.0 * np.pi))
        gauss = A * np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
        
        max_amp = np.max(np.abs(gauss))
        if max_amp > _AMP_BOUND:
            gauss *= _AMP_BOUND / max_amp
        
        p_test = np.zeros((_N, 2))
        p_test[:, 0] = gauss
        
        s_test = _compute_score(p_test)
        if s_test is not None and s_test < best_score:
            best_pulse = p_test
            best_score = s_test
    
    return best_pulse
# EVOLVE-BLOCK-END

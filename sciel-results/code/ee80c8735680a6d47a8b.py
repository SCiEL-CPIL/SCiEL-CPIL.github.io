"""Robust control pulses for a three-level transmon with DRAG correction and local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_GATE = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
TPHI = 40000.0  # ns

def _make_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian matrix for a given step."""
    # a = diag(sqrt([1,2]), 1)
    a = np.diag([np.sqrt(1.0), np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a  # number operator: diag(0, 1, 2)
    
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3)))
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H

def _liouvillian_step(detuning, scale, I, Q, dt):
    """Compute the Liouvillian superoperator for one time step and return the propagator."""
    H = _make_hamiltonian(detuning, scale, I, Q)
    
    # Lindblad operators
    a = np.diag([np.sqrt(1.0), np.sqrt(2.0), 1.0])
    n = a.conj().T @ a
    
    L1 = a / np.sqrt(T1)  # amplitude damping
    L2 = n * np.sqrt(2.0 / TPHI)  # pure dephasing (number operator)
    
    # Liouvillian in vectorized form: L = -i(H⊗I - I⊗H^T) + sum_k (L_k⊗L_k^* - 0.5{L_k^†L_k, .})
    # For the propagator, we need to exponentiate the Liouvillian acting on the density matrix.
    # Vectorize: |ρ> = vec(ρ), then d|ρ>/dt = L_super @ |ρ>
    # L_super = -i(H ⊗ I - I ⊗ H^T) + sum_k [L_k ⊗ L_k^* - 0.5 (L_k^† L_k ⊗ I + I ⊗ (L_k^† L_k)^T)]
    
    I3 = np.eye(3)
    L_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    for L in [L1, L2]:
        L_dag_L = L.conj().T @ L
        L_super += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag_L, I3) + np.kron(I3, L_dag_L.T))
    
    # Exponentiate
    U = expm(L_super * dt)
    return U

def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix as a 9x9 propagator-like object."""
    # pulse is (80, 2) array of [I, Q]
    # Start with identity propagator (vectorized 9x9)
    rho_vec = np.zeros(9, dtype=complex)
    # Actually, we need to track the evolution of the density matrix.
    # Let's use the propagator approach: compute the full propagator P such that rho_final = P @ rho_initial
    
    # Initialize propagator as identity (vectorized)
    P = np.eye(9, dtype=complex)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        U_step = _liouvillian_step(detuning, scale, I_val, Q_val, DT)
        P = U_step @ P
    
    return P

def _compute_fidelities(P, detuning, scale):
    """Compute mean fidelity for the 5 test states."""
    # Test states: |0>, |1>, (|0>±|1>)/sqrt(2), (|0>±i|1>)/sqrt(2)
    # |0> = [1,0,0], |1> = [0,1,0]
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi in states:
        rho0 = np.outer(psi, psi.conj())
        rho0_vec = rho0.flatten()
        rho1_vec = P @ rho0_vec
        rho1 = rho1_vec.reshape(3, 3)
        
        # Fidelity: F = <psi|rho1|psi> for pure state input
        # But rho1 may not be pure. Use Uhlmann fidelity: F = (Tr(sqrt(sqrt(rho0) rho1 sqrt(rho0))))^2
        # Since rho0 is pure: F = <psi|rho1|psi>
        F = np.real(np.vdot(psi, rho1 @ psi))
        F = np.clip(F, 0, 1)
        fidelities.append(F)
    
    return np.mean(fidelities)

def _score_pulse(pulse):
    """Compute the full score: mean + max of infidelities over 9 cases."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            P = _simulate_pulse(pulse, det, sc)
            fid = _compute_fidelities(P, det, sc)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def _check_validity(pulse):
    """Check amplitude and slew rate constraints."""
    # Amplitude bound
    mags = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.max(mags) > AMP_BOUND + 1e-9:
        return False
    
    # Slew rate bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero to step 1
    diff0 = np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2)
    if diff0 / DT > SLEW_BOUND + 1e-9:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.sqrt((pulse[i+1, 0] - pulse[i, 0])**2 + (pulse[i+1, 1] - pulse[i, 1])**2)
        if diff / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    diff_last = np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2)
    if diff_last / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def _generate_drag_pulse(beta, sigma_factor, area_scale):
    """Generate a DRAG pulse with Gaussian envelope and derivative correction."""
    t = np.arange(N_STEPS) * DT  # time points
    t_center = T_GATE / 2.0
    sigma = (T_GATE / 6.0) * sigma_factor
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Normalize area to pi (integral of gauss * dt ≈ pi)
    gauss_area = np.sum(gauss) * DT
    gauss_norm = gauss * (np.pi / gauss_area) * area_scale
    
    # Derivative of Gaussian (for DRAG term)
    dgauss = -((t - t_center) / sigma**2) * gauss
    dgauss_area = np.sum(dgauss) * DT
    # The derivative term should have zero net area, but we scale it by beta
    dgauss_norm = dgauss * (np.pi / gauss_area) * beta
    
    # I = gaussian, Q = derivative term (DRAG)
    I = gauss_norm
    Q = dgauss_norm
    
    pulse = np.column_stack([I, Q])
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = 330.0 - 30.0  # 300s limit, keep 30s margin
    
    # Generate initial DRAG pulse
    # Optimal beta for DRAG: beta = -anharmonicity * tau_gauss^2 (approximately)
    # More precisely, for a Gaussian with sigma, beta_opt ≈ -Δ * sigma^2 where Δ is anharmonicity
    sigma_base = T_GATE / 6.0
    beta_opt = -ANHARMONICITY * sigma_base**2 * 2.0  # factor of 2 from convention
    
    # Start with the analytically-informed DRAG pulse
    pulse_best = _generate_drag_pulse(beta_opt, 1.0, 1.0)
    
    # Check validity and get initial score
    if not _check_validity(pulse_best):
        # Scale down to meet constraints
        mags = np.sqrt(pulse_best[:, 0]**2 + pulse_best[:, 1]**2)
        max_mag = np.max(mags)
        if max_mag > AMP_BOUND:
            pulse_best *= AMP_BOUND / max_mag
        # Check slew
        diffs = np.diff(pulse_best, axis=0)
        max_diff = np.max(np.sqrt(np.sum(diffs**2, axis=1)))
        if max_diff / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / max_diff
            pulse_best *= scale
    
    score_best = _score_pulse(pulse_best)
    
    # Local optimization using Nelder-Mead
    def objective(params):
        beta, sigma_f, area_s = params
        p = _generate_drag_pulse(beta, sigma_f, area_s)
        if not _check_validity(p):
            return 1e6
        return _score_pulse(p)
    
    x0 = np.array([beta_opt, 1.0, 1.0])
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:  # Need at least 60s for optimization
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                     'adaptive': True})
            if result.success or result.fun < score_best:
                beta, sigma_f, area_s = result.x
                pulse_candidate = _generate_drag_pulse(beta, sigma_f, area_s)
                if _check_validity(pulse_candidate):
                    score_candidate = _score_pulse(pulse_candidate)
                    if score_candidate < score_best:
                        pulse_best = pulse_candidate
                        score_best = score_candidate
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(pulse_best):
        # Fallback to a simple valid pulse
        mags = np.sqrt(pulse_best[:, 0]**2 + pulse_best[:, 1]**2)
        max_mag = np.max(mags)
        if max_mag > AMP_BOUND:
            pulse_best *= AMP_BOUND / max_mag
        diffs = np.diff(pulse_best, axis=0)
        max_diff = np.max(np.sqrt(np.sum(diffs**2, axis=1))) if N_STEPS > 1 else 0
        if max_diff / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / max_diff
            pulse_best *= scale
        # Also check boundary slew
        if np.sqrt(pulse_best[0,0]**2 + pulse_best[0,1]**2) / DT > SLEW_BOUND:
            pulse_best[0] *= SLEW_BOUND * DT / np.sqrt(pulse_best[0,0]**2 + pulse_best[0,1]**2)
        if np.sqrt(pulse_best[-1,0]**2 + pulse_best[-1,1]**2) / DT > SLEW_BOUND:
            pulse_best[-1] *= SLEW_BOUND * DT / np.sqrt(pulse_best[-1,0]**2 + pulse_best[-1,1]**2)
    
    return pulse_best
# EVOLVE-BLOCK-END

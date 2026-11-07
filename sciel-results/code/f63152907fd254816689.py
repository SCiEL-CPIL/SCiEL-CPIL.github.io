"""
Robust DRAG pulse optimization for a three-level transmon.
Strategy:
  - Build the 3-level Hamiltonian / Liouvillian exactly as specified.
  - Use a DRAG-style Gaussian seed (area pi, sigma = 40/6 ns) with a derivative
    component (beta) to cancel leakage to |2>.
  - Optimize (amp, sigma, beta, q_scale) via Nelder-Mead minimizing the mean
    infidelity over the 9 hardware variations.
  - Enforce amplitude and slew-rate constraints (including zero-to-step-1 and
    last-step-to-zero transitions).
  - Time-budget guard: return best valid pulse before 300 s.
"""

import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 300.0

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
ANHARMONICITY = -2.0 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2.0 * np.pi * 0.002, 2.0 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
DT = 0.5
N_STEPS = 80
GATE_DURATION = 40.0  # ns

# Precompute
_A = np.array([[0.0, np.sqrt(1.0), 0.0],
               [0.0, 0.0, np.sqrt(2.0)],
               [0.0, 0.0, 0.0]], dtype=complex)
_A_DAG = _A.conj().T
_N = _A_DAG @ _A  # diag(0,1,2)

# Lindblad operators
_L1 = _A / np.sqrt(T1)
_L2 = _N * np.sqrt(2.0 / TPHI)

# Initial states for fidelity evaluation
_INIT_STATES = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [(1, 1)] / np.sqrt(2),
    [(1, -1)] / np.sqrt(2),
    [(1, 1j)] / np.sqrt(2),
    [(1, -1j)] / np.sqrt(2),
], dtype=complex)

# X gate target (in the computational subspace; |2> stays |2>)
# X swaps |0> and |1>
_X_GATE = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=complex)

# 9 hardware cases: detuning x scale
_DETUNINGS = np.array([-2.0 * np.pi * 0.002, 0.0, 2.0 * np.pi * 0.002])
_SCALES = np.array([0.97, 1.0, 1.03])


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one time step.
    
    H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    Lindblad: L1 = a/sqrt(T1), L2 = n*sqrt(2/Tphi)
    
    The Liouvillian acts on the vectorized density matrix (row-major).
    dρ/dt = -i[H, ρ] + sum_k (L_k ρ L_k† - 0.5{L_k† L_k, ρ})
    """
    n_nm1 = _N @ (_N - np.eye(3))
    H = (detuning * _N
         + 0.5 * ANHARMONICITY * n_nm1
         + 0.5 * scale * (I * (_A + _A_DAG) + 1j * Q * (_A - _A_DAG)))
    
    # Build superoperator
    # Vectorize: |ρ>> = vec(ρ) where vec stacks columns (Fortran order) for
    # standard Kronecker form, but we'll use row-major (C order) consistently.
    # For C-order vec: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
    # Actually, let's use the standard: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
    # where vec stacks columns. Let me use column-stacking (Fortran).
    
    # -i[H,ρ] = -i(H ρ - ρ H)
    # vec(H ρ) = (I ⊗ H) vec(ρ)  [column stacking]
    # vec(ρ H) = (H^T ⊗ I) vec(ρ)
    # So -i[H,ρ] -> -(i)(I⊗H - H^T⊗I) vec(ρ)
    
    dim = 3
    I3 = np.eye(dim, dtype=complex)
    
    # Dissipator for one Lindblad L:
    # L ρ L† -> vec: (L†^T ⊗ L) vec(ρ) = (conj(L) ⊗ L) vec(ρ)
    # -0.5(L†L ρ + ρ L†L) -> -0.5((I⊗L†L) + (L†L^T ⊗ I)) vec(ρ)
    
    def _lindblad_superop(L):
        Ldag_L = L.conj().T @ L
        term1 = np.conj(L) @ L  # L†^T ⊗ L  ->  (L^*) ⊗ L
        term2 = np.kron(I3, Ldag_L) + np.kron(Ldag_L.T, I3)
        return term1 - 0.5 * term2
    
    superop = (-1j) * (np.kron(I3, H) - np.kron(H.T, I3))
    superop += _lindblad_superop(_L1)
    superop += _lindblad_superop(_L2)
    
    return superop


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final process matrix (3x3x3x3).
    
    pulse: (80, 2) array of (I, Q)
    Returns: the 9x9 superoperator mapping initial ρ to final ρ (column-stacked).
    """
    # We track the process matrix by evolving each column of the identity
    # (i.e., each basis element of the density matrix space).
    # Actually, we just need to compute fidelity for the 6 initial states.
    # But for efficiency, let's evolve the 9 basis states of the superoperator.
    
    dim = 3
    # Process matrix P: P[i,j] = <i| U ρ_j U† >  ... 
    # Actually we need the full process including dissipation.
    # Let's just evolve each of the 9 basis density matrices.
    
    # Basis states for the 9-dim density matrix space (column-stacked)
    basis = np.zeros((9, 9), dtype=complex)
    for i in range(3):
        for j in range(3):
            basis[i*3+j, i*3+j] = 1.0  # |i><j| -> vec
    
    # Evolve all 9 simultaneously
    rho_vecs = basis.copy()  # (9, 9) each row is a vec(ρ)
    
    for step in range(N_STEPS):
        I_val = pulse[step, 0]
        Q_val = pulse[step, 1]
        S = _build_liouvillian(detuning, scale, I_val, Q_val)
        # expm(S * dt) applied to each column
        # For efficiency, compute expm once and multiply
        # Actually S is 9x9, so expm is cheap
        from scipy.linalg import expm
        U_step = expm(S * DT)
        rho_vecs = U_step @ rho_vecs
    
    return rho_vecs  # (9, 9): columns are final vec(ρ) for each initial basis


def _compute_fidelities(rho_vecs):
    """Compute mean fidelity to X gate for the 6 initial states."""
    # Target: X_gate applied to each initial state
    # Fidelity = <ψ_target| ρ_final |ψ_target> for pure states
    # But we have mixed states due to dissipation, so use:
    # F = <ψ_t| ρ |ψ_t> for pure target
    
    total_fidelity = 0.0
    n_states = len(_INIT_STATES)
    
    for k in range(n_states):
        psi_init = _INIT_STATES[k]
        psi_target = _X_GATE @ psi_init
        psi_target = psi_target / np.linalg.norm(psi_target)
        
        # Find which column corresponds to this initial state
        # psi_init is a pure state |i><i| for k=0,1 or superpositions
        # We need to map the initial state to the basis index
        # Actually, the 6 initial states are specific pure states.
        # Let me just simulate each one separately for clarity.
        
        # vec(psi_init psi_init†)
        rho_init = np.outer(psi_init, psi_init.conj())
        rho_init_vec = rho_init.flatten(order='F')  # column stacking
        
        # Find which column of rho_vecs matches this
        # Actually, let me just compute directly
        # The final state for this initial state is:
        # We need to apply the process to rho_init_vec
        # But rho_vecs was computed for the 9 basis states |i><j|
        # rho_init_vec = sum_{ij} psi_i psi_j* |i><j|
        # So final = sum_{ij} psi_i psi_j* rho_vecs[:, i*3+j]
        
        final_vec = np.zeros(9, dtype=complex)
        for i in range(3):
            for j in range(3):
                coeff = psi_init[i] * psi_init[j].conj()
                final_vec += coeff * rho_vecs[:, i*3+j]
        
        # Fidelity with pure target
        fidelity = np.real(np.vdot(psi_target, final_vec).real)
        # Wait, for pure state |ψ>, fidelity with ρ is <ψ|ρ|ψ>
        # vec(ρ) in column-stacking: <ψ|ρ|ψ> = sum_{ij} ψ_i* ρ_{ij} ψ_j
        # = sum_{ij} (ψ* ⊗ ψ)_{ij} ρ_{ij}
        # Let me just reconstruct ρ and compute
        rho_final = final_vec.reshape(3, 3, order='F')
        fidelity = np.real(np.vdot(psi_target, rho_final @ psi_target))
        
        total_fidelity += fidelity
    
    return total_fidelity / n_states


def _score_pulse(pulse):
    """Compute the score: mean + max infidelity over 9 hardware cases."""
    infidelities = []
    for det in _DETUNINGS:
        for scale in _SCALES:
            rho_vecs = _simulate_pulse(pulse, det, scale)
            fid = _compute_fidelities(rho_vecs)
            infidelities.append(1.0 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _check_validity(pulse):
    """Check amplitude and slew-rate constraints."""
    # Amplitude bound
    mags = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(mags > AMP_BOUND + 1e-9):
        return False
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    # Last step to zero
    if np.linalg.norm(prev - np.array([0.0, 0.0])) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _enforce_constraints(pulse):
    """Clip pulse to satisfy amplitude and slew-rate constraints."""
    pulse = pulse.copy()
    
    # Enforce amplitude
    mags = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    for i in range(N_STEPS):
        if mags[i] > AMP_BOUND:
            scale = AMP_BOUND / mags[i]
            pulse[i] *= scale
    
    # Enforce slew rate (iteratively)
    for _ in range(10):
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            diff = pulse[i] - prev
            norm = np.linalg.norm(diff)
            max_step = SLEW_BOUND * DT
            if norm > max_step:
                pulse[i] = prev + diff / norm * max_step
            prev = pulse[i]
        # Check last to zero
        diff = np.array([0.0, 0.0]) - pulse[-1]
        norm = np.linalg.norm(diff)
        max_step = SLEW_BOUND * DT
        if norm > max_step:
            # Need to reduce last step
            pulse[-1] = np.array([0.0, 0.0]) - diff / norm * max_step
        # Re-enforce amplitude
        mags = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        for i in range(N_STEPS):
            if mags[i] > AMP_BOUND:
                scale = AMP_BOUND / mags[i]
                pulse[i] *= scale
    
    return pulse


def _generate_drag_pulse(amp, sigma, beta, q_scale):
    """Generate a DRAG-style Gaussian pulse.
    
    amp: peak amplitude
    sigma: Gaussian width (ns)
    beta: DRAG beta parameter (Q/I ratio for derivative)
    q_scale: additional Q scaling
    
    Area should be pi for a pi pulse.
    Gaussian: A * exp(-t^2/(2*sigma^2))
    Area = A * sigma * sqrt(2*pi) = pi  =>  A = 1/(sigma * sqrt(2/pi)) = sqrt(pi/2)/sigma
    """
    t = np.linspace(-GATE_DURATION/2 + DT/2, GATE_DURATION/2 - DT/2, N_STEPS)
    
    # Gaussian envelope
    gauss = np.exp(-t**2 / (2.0 * sigma**2))
    
    # Normalize area to pi
    area = np.trapz(gauss, t)
    if area < 1e-10:
        area = 1.0
    gauss = gauss * (np.pi / area)
    
    # I component: Gaussian
    I = amp * gauss
    
    # Q component: DRAG derivative + scaling
    # Derivative of Gaussian: -t/sigma^2 * gauss
    dgauss = -t / sigma**2 * gauss
    # Normalize derivative area similarly
    darea = np.trapz(np.abs(dgauss), t)
    if darea < 1e-10:
        darea = 1.0
    dgauss = dgauss * (np.pi / darea)
    
    Q = amp * beta * dgauss + amp * q_scale * gauss * 0.0  # q_scale adds to I-like
    
    pulse = np.column_stack([I, Q])
    
    # Offset so edges are zero (Gaussian already decays, but ensure)
    # The Gaussian is naturally ~0 at edges for reasonable sigma
    
    return pulse


def _optimize():
    """Run Nelder-Mead optimization over DRAG parameters."""
    best_score = np.inf
    best_pulse = None
    
    # Initial parameter guesses: (amp, sigma, beta, q_scale)
    # Seed: sigma = 40/6 ≈ 6.67 ns
    # For a pi pulse with duration 40 ns, sigma ~ 6.67
    # Peak amplitude for area pi: A = pi / (sigma * sqrt(2*pi)) ≈ 3.14/(6.67*2.507) ≈ 0.189
    # But we need to stay within amp bound 0.4
    
    starting_points = [
        (0.19, 6.67, 0.0, 0.0),    # Seed DRAG
        (0.25, 6.0, 0.1, 0.0),
        (0.30, 5.5, 0.2, 0.0),
        (0.35, 5.0, 0.3, 0.0),
        (0.40, 4.5, 0.5, 0.0),
        (0.20, 7.0, 0.05, 0.0),
        (0.25, 6.5, 0.15, 0.0),
        (0.30, 6.0, 0.25, 0.0),
    ]
    
    bounds = [(0.05, AMP_BOUND), (3.0, 15.0), (-1.0, 1.0), (-0.5, 0.5)]
    
    def _objective(params):
        amp, sigma, beta, q_scale = params
        if time.time() - _T0 > _DEADLINE - 30:
            raise TimeoutError()
        
        pulse = _generate_drag_pulse(amp, sigma, beta, q_scale)
        pulse = _enforce_constraints(pulse)
        
        if not _check_validity(pulse):
            return 1e10
        
        score = _score_pulse(pulse)
        return score
    
    for start in starting_points:
        if time.time() - _T0 > _DEADLINE - 60:
            break
        
        try:
            result = minimize(_objective, start, method='Nelder-Mead',
                             bounds=bounds,
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6,
                                      'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _generate_drag_pulse(*result.x)
                best_pulse = _enforce_constraints(best_pulse)
        except (TimeoutError, Exception):
            continue
    
    # If no valid pulse found, return the seed
    if best_pulse is None:
        best_pulse = _generate_drag_pulse(0.19, 6.67, 0.0, 0.0)
        best_pulse = _enforce_constraints(best_pulse)
    
    return best_pulse


def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return a zero pulse for invalid targets
        return np.zeros((80, 2))
    
    # Time budget: use up to 280 seconds for optimization
    t_start = time.time()
    time_budget = 280.0  # Leave 20s margin
    
    try:
        best_pulse = _optimize()
    except Exception:
        # Fallback to seed pulse
        best_pulse = _generate_drag_pulse(0.19, 6.67, 0.0, 0.0)
        best_pulse = _enforce_constraints(best_pulse)
    
    # Final validity check
    if not _check_validity(best_pulse):
        best_pulse = _enforce_constraints(best_pulse)
        if not _check_validity(best_pulse):
            # Last resort: very small pulse
            best_pulse = np.zeros((80, 2))
    
    return best_pulse

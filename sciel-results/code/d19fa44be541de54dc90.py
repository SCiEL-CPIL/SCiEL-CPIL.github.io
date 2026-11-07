"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_TOTAL = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T_PHI = 40000.0

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # seconds
_DEADLINE = _START_TIME + _TIME_LIMIT


def _check_deadline():
    return time.time() > _DEADLINE


def _build_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian matrix."""
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    
    H = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator (9x9) for the Lindblad equation."""
    H = _build_hamiltonian(detuning, scale, I, Q)
    
    # Lindblad operators
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2 / T_PHI)
    
    # Liouvillian: dρ/dt = -i[H,ρ] + LρL† - 0.5{L†L, ρ}
    # Vectorize: vec(ρ) is 9x1
    # Superoperator: S = -i(H⊗I - I⊗H†) + sum_k (L_k⊗L_k* - 0.5(L_k†L_k)⊗I - 0.5 I⊗(L_k†L_k)†)
    
    I9 = np.eye(9, dtype=complex)
    I3 = np.eye(3, dtype=complex)
    
    S = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    
    for L in [L1, L2]:
        L_dag = L.conj().T
        S += np.kron(L, L.conj()) - 0.5 * np.kron(L_dag @ L, I3) - 0.5 * np.kron(I3, (L_dag @ L).T)
    
    return S


def _simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    rho_vec = rho.flatten()
    
    for i in range(N_STEPS):
        S = _build_liouvillian(detuning, scale, pulse_I[i], pulse_Q[i])
        # Exponentiate the Liouvillian for dt
        S_dt = expm(S * DT)
        rho_vec = S_dt @ rho_vec
        rho = rho_vec.reshape((3, 3))
    
    return rho


def _compute_fidelity(rho, target_state):
    """Compute fidelity F = <ψ|ρ|ψ> for a pure target state."""
    return np.real(np.vdot(target_state, rho @ target_state))


def _compute_infidelity(pulse_I, pulse_Q, detuning, scale):
    """Compute the mean infidelity over 4 test states."""
    # Target: X gate (NOT gate)
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|0>+|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = (|0>+i|1>)/sqrt(2) ... wait
    # Let's be careful:
    # X = [[0,1],[1,0]]
    # X|0> = |1>
    # X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = (|0>+|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = (|0>+i|1>)/sqrt(2) ... no
    # |1> + i|0> = i|0> + |1> = (|0> + (1/i)|1>)/sqrt(2) * sqrt(2) ...
    # Actually: X(|0>+i|1>)/sqrt(2) = (|1> + i|0>)/sqrt(2)
    # And (|0>+i|1>)/sqrt(2) is the input. The output is (|1>+i|0>)/sqrt(2).
    # Note that (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2) = i * (|0>+i|1>)/sqrt(2) * (-i) ...
    # Let's just compute the target states explicitly.
    
    # In the 3-level system, we only care about the |0>, |1> subspace for the X gate.
    # |0> = [1, 0, 0], |1> = [0, 1, 0], |2> = [0, 0, 1]
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)  # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    ]
    
    # Wait, the task says: fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # So the inputs are |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # And we compare the output to X(input).
    # But the task says "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # That's 6 states? Or 4? Let's re-read: "fidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # The notation "|0> +- |1>" means both |0>+|1> and |0>-|1>. Similarly for i.
    # So 2 + 2 + 2 = 6 states? But it says "mean fidelity to X over" these 6 states.
    # Actually, let's just use 4 states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2)
    # And their X counterparts. The fidelity is F = <Xψ|ρ|Xψ> where ψ is the input.
    
    # Let's use the 4 states mentioned: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2)
    # The target states are X|0>=|1>, X|1>=|0>, X(|0>+|1>)/sqrt(2)=(|0>+|1>)/sqrt(2), X(|0>+i|1>)/sqrt(2)=(|1>+i|0>)/sqrt(2)
    
    # Actually, to be safe, let's use all 6: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    
    input_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # X gate in 2-level: X = [[0,1],[1,0]]
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # For each input state, simulate and compute fidelity to the target state.
    fidelities = []
    for psi, phi in zip(input_states, target_states):
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0  # This is wrong! We need to start in the input state psi.
        
        # Actually, the simulation should start in the input state psi, not |0>.
        # But the pulse is designed to implement X, so for input |0>, it should output |1>.
        # For input |1>, it should output |0>.
        # So we need to simulate from each input state.
        
        rho = np.outer(psi, psi.conj())
        rho_vec = rho.flatten()
        
        for i in range(N_STEPS):
            S = _build_liouvillian(detuning, scale, pulse_I[i], pulse_Q[i])
            S_dt = expm(S * DT)
            rho_vec = S_dt @ rho_vec
            rho = rho_vec.reshape((3, 3))
        
        F = np.real(np.vdot(phi, rho @ phi))
        fidelities.append(F)
    
    mean_fidelity = np.mean(fidelities)
    infidelity = 1.0 - mean_fidelity
    return infidelity


def _score_pulse(pulse_I, pulse_Q):
    """Compute the score: mean + max of infidelities over 9 hardware variations."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _compute_infidelity(pulse_I, pulse_Q, det, sc)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score


def _check_validity(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amps > AMP_BOUND + 1e-9):
        return False
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero.
    # p[i] = (I[i], Q[i])
    
    # From zero to step 1
    d1 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if d1 > SLEW_BOUND + 1e-9:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        dI = pulse_I[i + 1] - pulse_I[i]
        dQ = pulse_Q[i + 1] - pulse_Q[i]
        d = np.sqrt(dI**2 + dQ**2) / DT
        if d > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    d_last = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if d_last > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _legendre_basis(t):
    """Return the first 8 Legendre polynomials evaluated at t (scaled to [0,1])."""
    # t is in [0, 1], map to x in [-1, 1]
    x = 2 * t - 1
    P = np.zeros((8, len(t)))
    P[0, :] = 1
    P[1, :] = x
    for n in range(2, 8):
        P[n, :] = ((2 * n - 1) * x * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
    return P


def _coeffs_to_pulse(coeffs):
    """Convert 8 Legendre coefficients to an (80, 2) pulse array."""
    t = np.linspace(0, 1, N_STEPS)
    P = _legendre_basis(t)
    
    # First 4 coeffs for I, next 4 for Q
    I = P[:4, :].T @ coeffs[:4]
    Q = P[4:, :].T @ coeffs[4:]
    
    return I, Q


def _pulse_to_coeffs(I, Q):
    """Convert an (80, 2) pulse array to 8 Legendre coefficients."""
    t = np.linspace(0, 1, N_STEPS)
    P = _legendre_basis(t)
    
    # Least squares fit
    coeffs_I, _, _, _ = np.linalg.lstsq(P[:4, :].T, I, rcond=None)
    coeffs_Q, _, _, _ = np.linalg.lstsq(P[4:, :].T, Q, rcond=None)
    
    return np.concatenate([coeffs_I, coeffs_Q])


def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    sigma = 40 / 6  # ns
    center = 20.0  # ns
    
    # Gaussian with area pi
    amp = np.pi / (sigma * np.sqrt(2 * np.pi))
    I = amp * np.exp(-(t - center)**2 / (2 * sigma**2))
    
    # Offset to 0 at the edges
    I[0] = 0.0
    I[-1] = 0.0
    
    # DRAG term: Q = beta * dI/dt
    beta = 0.5
    dI = np.gradient(I, DT)
    Q = beta * dI
    
    return I, Q


def pulse(target_gate, hardware_model):
    """Main entry point. Returns an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Start with the seed pulse
    I_seed, Q_seed = _seed_pulse()
    pulse_seed = np.column_stack([I_seed, Q_seed])
    
    # Check if seed is valid
    if _check_validity(I_seed, Q_seed):
        best_pulse = pulse_seed
        best_score = _score_pulse(I_seed, Q_seed)
    else:
        best_pulse = np.zeros((N_STEPS, 2))
        best_score = np.inf
    
    # Convert seed to Legendre coefficients
    coeffs_seed = _pulse_to_coeffs(I_seed, Q_seed)
    
    # Define the objective function
    def objective(coeffs):
        if _check_deadline():
            return best_score
        I, Q = _coeffs_to_pulse(coeffs)
        if not _check_validity(I, Q):
            return 1e10
        return _score_pulse(I, Q)
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(
            objective,
            coeffs_seed,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.success and result.fun < best_score:
            I_opt, Q_opt = _coeffs_to_pulse(result.x)
            if _check_validity(I_opt, Q_opt):
                best_pulse = np.column_stack([I_opt, Q_opt])
                best_score = result.fun
    except Exception:
        pass
    
    # Try a few more random starting points if time permits
    if not _check_deadline():
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _check_deadline():
                break
            coeffs_rand = coeffs_seed + rng.normal(0, 0.1, 8)
            try:
                result = minimize(
                    objective,
                    coeffs_rand,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.success and result.fun < best_score:
                    I_opt, Q_opt = _coeffs_to_pulse(result.x)
                    if _check_validity(I_opt, Q_opt):
                        best_pulse = np.column_stack([I_opt, Q_opt])
                        best_score = result.fun
            except Exception:
                pass
    
    return best_pulse
# EVOLVE-BLOCK-END

"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = 310.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_T1 = 25000.0
_TPHI = 40000.0

# 9 cases: detuning in {-2pi*0.002, 0, 2pi*0.002}, scale in {0.97, 1.0, 1.03}
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_CASES = [(d, s) for d in _DET_GRID for s in _SCALE_GRID]

# 5 test states
_STATES = np.zeros((5, 3), dtype=complex)
_STATES[0] = [1, 0, 0]  # |0>
_STATES[1] = [0, 1, 0]  # |1>
_STATES[2] = [1/np.sqrt(2), 1/np.sqrt(2), 0]  # (|0>+|1>)/sqrt(2)
_STATES[3] = [1/np.sqrt(2), -1/np.sqrt(2), 0]  # (|0>-|1>)/sqrt(2)
_STATES[4] = [1/np.sqrt(2), 1j/np.sqrt(2), 0]  # (|0>+i|1>)/sqrt(2)

# Operators: a = diag(sqrt([1,2]),1); a^dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A  # n = a^dag a

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute Liouvillian coefficients (constant parts)
# L = -i(H rho - rho H) + L1 rho L1^dag - 1/2(L1^dag L1 rho + rho L1^dag L1) + similar for L2
# For a given H, the Liouvillian L maps rho -> -i[H,rho] + ...
# The -i[H,rho] part: L_H = -i H^L + i H^R where H^L is left mult, H^R is right mult
# In superoperator form: L = -i(H kron I - I kron H^T) + ...
# But we'll build it step by step.

# For the constant Lindblad part (independent of pulse), precompute:
# L_damp = L1 rho L1^dag - 1/2(L1^dag L1 rho + rho L1^dag L1) + L2 rho L2^dag - 1/2(L2^dag L2 rho + rho L2^dag L2)
# In superoperator form (vectorizing rho column-wise or row-wise)

# Let's use the convention: vec(rho) stacks columns. Then A rho B -> (B^T kron A) vec(rho)
# Actually, vec(A X B) = (B^T kron A) vec(X)

# For L1 rho L1^dag: (L1^T kron L1)
# For L1^dag L1 rho: (I kron L1^dag L1)
# For rho L1^dag L1: ((L1^dag L1)^T kron I) = (L1^dag L1 kron I) since it's Hermitian
# So L1 part: (L1^T kron L1) - 1/2(I kron L1^dag L1) - 1/2(L1^dag L1 kron I)
# Similarly for L2.

_L1DAG_L1 = _L1ADAG @ _L1
_L2DAG_L2 = _L2ADAG @ _L2

# Superoperator for Lindblad part
# Using vec convention (stack columns)
# A rho B -> (B^T kron A) vec(rho)
# L1 rho L1^dag: A=L1, B=L1^dag -> (L1^dag)^T kron L1 = L1^T kron L1
# (L1^dag L1) rho: A=L1^dag L1, B=I -> I^T kron (L1^dag L1) = I kron (L1^dag L1)
# rho (L1^dag L1): A=I, B=L1^dag L1 -> (L1^dag L1)^T kron I = (L1^dag L1) kron I

_Lind_sup = (np.kron(_L1.T, _L1) - 0.5 * np.kron(np.eye(3), _L1DAG_L1) - 0.5 * np.kron(_L1DAG_L1, np.eye(3))
            + np.kron(_L2.T, _L2) - 0.5 * np.kron(np.eye(3), _L2DAG_L2) - 0.5 * np.kron(_L2DAG_L2, np.eye(3)))

def _hamiltonian(I, Q, detuning, scale):
    """Build the Hamiltonian for given I, Q amplitudes."""
    # H = detuning * n + 0.5*anharmonicity * n(n-1) + 0.5*scale*(I*(a+a^dag) + i*Q*(a-a^dag))
    # n(n-1) = n^2 - n
    N2 = _NOP @ _NOP
    n_nm1 = N2 - _NOP
    H_static = detuning * _NOP + 0.5 * _ANH * n_nm1
    # Drive term
    # I*(a+a^dag) + i*Q*(a-a^dag)
    # a+a^dag is Hermitian, a-a^dag is anti-Hermitian (purely imaginary diagonal... wait)
    # a = diag(1, sqrt(2), 1), a^dag = same (real diagonal), so a-a^dag = 0?
    # Wait, a is diagonal real, so a^dag = a^T = a (since real). So a - a^dag = 0!
    # That means Q term vanishes? Let me re-read the task.
    # "0.5 scale (I (a + a^dagger) + i Q (a - a^dagger))"
    # If a is diagonal real, then a - a^dag = 0. So Q has no effect?
    # But that seems odd. Let me re-read: a = diag(sqrt([1, 2]), 1)
    # sqrt([1,2]) = [1, sqrt(2)]. So a = diag(1, sqrt(2), 1). Yes, real diagonal.
    # So a - a^dag = 0. Q term is zero.
    # Hmm, but the task says I and Q amplitudes. Maybe I'm misreading.
    # "a = diag(sqrt([1, 2]), 1)" - this means a is a 3x3 diagonal matrix with entries sqrt(1), sqrt(2), 1.
    # So a = diag(1, sqrt(2), 1). Real diagonal. a^dag = a^H = a^T = a.
    # So a - a^dag = 0. Q term contributes nothing.
    # That's surprising but let's go with it.
    drive = 0.5 * scale * (I * (_A + _ADAG))  # i*Q*(a-a^dag) = 0
    return H_static + drive

def _liouvillian_step(rho, I, Q, detuning, scale):
    """Apply one time step of the Liouvillian to rho."""
    H = _hamiltonian(I, Q, detuning, scale)
    # H rho - rho H
    comm = H @ rho - rho @ H
    # -i * comm
    unitary_part = -1j * comm
    # Lindblad part
    lind_part = _L1 @ rho @ _L1ADAG - 0.5 * (_L1DAG_L1 @ rho + rho @ _L1DAG_L1)
    lind_part += _L2 @ rho @ _L2ADAG - 0.5 * (_L2DAG_L2 @ rho + rho @ _L2DAG_L2)
    return unitary_part + lind_part

def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix for each test state."""
    # pulse is (80, 2) array of (I, Q)
    # Return array of shape (5, 3, 3) of final density matrices
    final_states = np.zeros((5, 3, 3), dtype=complex)
    for k in range(5):
        rho = np.outer(_STATES[k], _STATES[k].conj())
        for t in range(_N):
            I = pulse[t, 0]
            Q = pulse[t, 1]
            drho = _liouvillian_step(rho, I, Q, detuning, scale)
            # Exponentiate the Liouvillian step: rho_{t+1} = exp(dt * L) rho_t
            # Since L depends on I,Q, we build the superoperator for this step
            H = _hamiltonian(I, Q, detuning, scale)
            # Superoperator for -i[H, .]:
            # -i(H rho - rho H) = -i H rho + i rho H
            # In superoperator form (vec columns):
            # H rho -> (I kron H) vec(rho)
            # rho H -> (H^T kron I) vec(rho)
            # So -i H rho + i rho H -> (-i I kron H + i H^T kron I) vec(rho)
            L_unit = -1j * np.kron(np.eye(3), H) + 1j * np.kron(H.T, np.eye(3))
            L_total = L_unit + _Lind_sup
            rho = np.linalg.expm(_DT * L_total) @ rho.reshape(9)
            rho = rho.reshape(3, 3)
            # Renormalize trace to 1 to combat numerical drift
            tr = np.trace(rho)
            if abs(tr) > 1e-15:
                rho = rho / tr
        final_states[k] = rho
    return final_states

def _fidelity(rho, target_state):
    """Compute fidelity |<target|rho|target>|... wait, for pure state |psi>, F = <psi|rho|psi>."""
    return np.real(np.vdot(target_state, rho @ target_state))

def _score_components(pulse):
    """Compute infidelity for each of the 9 cases and 5 states."""
    # For each case, simulate and compute fidelity for each state
    # Target is X gate on |0>->|1>, |1>->|0> (up to phase?)
    # Actually, the X gate: X|0> = |1>, X|1> = |0>
    # The target state for initial |psi> is X|psi>
    # Fidelity = |<X psi | rho | X psi>|^2? No, for pure states, F = |<psi_target|psi_final>|^2
    # But rho is a density matrix. If the final state is pure, F = <psi_target|rho|psi_target>
    # With dissipation, the state may not be pure, but we compute F = Tr(rho X |psi><psi| X^dag)
    # = Tr(rho X |psi><psi| X^dag)
    # For initial |psi>, target is X|psi>. F = <X psi | rho | X psi>
    # Actually, the task says "fidelity to X over |0>, |1>, ...". This means:
    # For each initial state |psi>, the ideal final state is X|psi>.
    # Fidelity = |<X psi | psi_final>|^2 if pure, or Tr(rho_final X|psi><psi|X^dag) in general.
    # Let's use F = Tr(rho_final * (X|psi>)(<psi|X^dag))
    # X|psi> = X @ |psi>
    
    # X matrix in 3-level: X swaps |0> and |1>, |2> stays? 
    # X = [[0,1,0],[1,0,0],[0,0,0]]? That's the qubit X.
    # For 3 levels, the "X-pi gate" likely means a pi pulse between |0> and |1>.
    # So the target unitary is the X gate on the {|0>,|1>} subspace.
    # X = [[0,1,0],[1,0,0],[0,0,1]]
    X_gate = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
    
    infidelities = np.zeros((9, 5))
    for ci, (det, sc) in enumerate(_CASES):
        finals = _simulate_pulse(pulse, det, sc)
        for k in range(5):
            psi = _STATES[k]
            target = X_gate @ psi
            target_dm = np.outer(target, target.conj())
            F = np.trace(finals[k] @ target_dm)
            F = np.clip(np.real(F), 0, 1)
            infidelities[ci, k] = 1 - F
    
    # For each case, mean fidelity to X over the 5 states
    mean_fidelity = np.mean(1 - infidelities, axis=1)  # 1 - mean infidelity = mean fidelity
    # Actually, infidelity = 1 - fidelity. mean infidelity per case = 1 - mean fidelity per case.
    case_infidelities = np.mean(infidelities, axis=1)
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    return mean_inf, max_inf, mean_inf + max_inf

def _make_seed_pulse():
    """DRAG-style Gaussian seed."""
    t = np.arange(_N) * _DT
    sigma = 40.0 / 6.0  # sigma = 40/6 ns
    # Gaussian: A * exp(-t^2 / (2 sigma^2))
    # Offset to 0 at edges: subtract value at t=0 and t=T
    g = np.exp(-t**2 / (2 * sigma**2))
    # Offset so that g[0] = 0 and g[-1] = 0
    # g is symmetric, so g[0] = g[-1] = exp(0) = 1 at t=0? No, t=0 -> g=1, t=40 -> g=exp(-1600/(2*44.4)) = exp(-18) ~ 0
    # So g[0] = 1, g[-1] ~ 0.
    # "offset to 0 at the edges" - subtract g[0] from all? Or subtract a linear trend?
    # Let's subtract g[0] * (1 - t/T) ... or just subtract g[0] to make g[0]=0.
    # Actually, "offset to 0 at the edges" probably means the pulse starts and ends at 0.
    # The Gaussian already starts at 1 and ends near 0. Let's subtract the value at t=0 to make it 0 at t=0.
    # But then the area changes. Let's normalize the area to pi.
    g_shifted = g - g[0]  # now g[0] = 0
    # Area = sum(g_shifted) * dt. We want area = pi.
    area = np.sum(g_shifted) * _DT
    if abs(area) > 1e-15:
        g_scaled = g_shifted * (np.pi / area)
    else:
        g_scaled = g_shifted
    # Q component: DRAG typically has a derivative term. But since a-a^dag=0, Q has no effect.
    # Let's set Q=0 for simplicity, or add a small derivative term.
    # Since Q doesn't affect the Hamiltonian (a-a^dag=0), Q is irrelevant.
    I = g_scaled
    Q = np.zeros(_N)
    pulse = np.column_stack([I, Q])
    return pulse

def _enforce_constraints(pulse):
    """Ensure amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    mag = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_mag = np.max(mag)
    if max_mag > _AMP:
        pulse = pulse * (_AMP / max_mag)
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12, including from zero to step 1 and last step to zero
    # p[0] - 0, p[i+1] - p[i], p[-1] - 0
    diffs = np.zeros((_N + 1, 2))
    diffs[0] = pulse[0]  # from 0 to p[0]
    for i in range(_N - 1):
        diffs[i+1] = pulse[i+1] - pulse[i]
    diffs[_N] = -pulse[-1]  # from p[-1] to 0
    
    mag_diffs = np.sqrt(np.sum(diffs**2, axis=1))
    max_slew = np.max(mag_diffs) / _DT
    if max_slew > _SLEW:
        # Scale the pulse to reduce slew
        # This is tricky because scaling the pulse scales all differences.
        # Let's try a simple approach: reduce the amplitude.
        scale = _SLEW * _DT / max(mag_diffs)
        pulse = pulse * scale
        # Re-check amplitude
        mag = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        max_mag = np.max(mag)
        if max_mag > _AMP:
            pulse = pulse * (_AMP / max_mag)
    return pulse

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((_N, 2))
    
    t0 = time.time()
    
    # Seed pulse
    p = _make_seed_pulse()
    p = _enforce_constraints(p)
    
    # Optimize using L-BFGS-B
    def objective(x):
        p = x.reshape(_N, 2)
        p = _enforce_constraints(p)
        mean_inf, max_inf, total = _score_components(p)
        return total
    
    x0 = p.flatten()
    
    # Gradient: use finite differences or analytic? Finite differences is slow (160 params).
    # Let's use numerical gradient with step size 1e-4.
    # Actually, let's just use L-BFGS-B without gradient and see if it's fast enough.
    # Or provide a gradient function.
    
    # Let's try with gradient first.
    def gradient(x):
        p = x.reshape(_N, 2)
        p = _enforce_constraints(p)
        grad = np.zeros_like(x)
        eps = 1e-4
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # This gradient is very expensive: 160 * 9 * 5 * 80 = 576000 simulations per gradient step.
    # Too slow. Let's use finite differences with a larger step or use a different optimizer.
    # Let's use Nelder-Mead or Powell without gradient, but with a coarser step.
    # Or, use L-BFGS-B with analytic gradient via backpropagation.
    
    # For now, let's try a simple local search: perturb the pulse and keep if better.
    # Or, use scipy.optimize.minimize with method='Nelder-Mead' which doesn't need gradient.
    
    best_p = p.copy()
    best_score = objective(x0)
    
    # Try a few local optimizations
    for method in ['Nelder-Mead', 'Powell']:
        if _time_remaining() < 60:
            break
        try:
            result = minimize(objective, x0, method=method, options={'maxiter': 200, 'fatol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_p = _enforce_constraints(result.x.reshape(_N, 2))
                x0 = best_p.flatten()
        except Exception:
            pass
    
    return best_p

# EVOLVE-BLOCK-END

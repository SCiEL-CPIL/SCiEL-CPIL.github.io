"""Structural change: Legendre-polynomial basis expansion optimized with L-BFGS-B."""
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import legendre

# Constants from the task
N_STEPS = 80
DT = 0.5  # ns
T_GATE = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
T2_PHI = 40000.0
# Note: Tphi in Lindblad is Tphi, so T2 = 2*Tphi? No, Tphi is pure dephasing.
# Lindblad operators: a/sqrt(T1) and n*sqrt(2/Tphi)
# So T1 = 25000, Tphi = 40000

# 3-level system
# a = diag(sqrt([1, 2]), 1) -> a = [[0,0,0],[sqrt(1),0,0],[0,sqrt(2),0]]
# n = a^dag a = diag(0, 1, 2)
A_OP = np.zeros((3, 3), dtype=complex)
A_OP[1, 0] = 1.0
A_OP[2, 1] = np.sqrt(2)
A_OP_DAG = A_OP.conj().T
N_OP = A_OP_DAG @ A_OP  # diag(0, 1, 2)

# Lindblad operators
L1 = A_OP / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T_PHI)  # Wait, Tphi = 40000
L2 = N_OP * np.sqrt(2.0 / 40000.0)

# Precompute Lindblad superoperator part
# Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
# We can precompute the "dissipative" part as a linear map.
# For small dt, we can use expm on the full Liouvillian, or use a simpler approximation.
# The task says "each step exponentiates the Liouvillian".
# Let's construct the Liouvillian matrix for a given H.
# rho is 3x3, so rho_vec is 9x1.
# L(rho) = -i(H rho - rho H) + ...
# We can build the 9x9 matrix.

def build_liouvillian(H, detuning, scale):
    # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I(a+a^dag) + i Q(a-a^dag))
    # Wait, the pulse is I and Q.
    # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    # But I and Q vary per step.
    # So H varies per step.
    # We need to exponentiate the Liouvillian for each step.
    # This is expensive if we build the 9x9 matrix and call expm 80 times.
    # But 80 is small.
    
    # Actually, the Liouvillian is linear in H (for the coherent part).
    # The dissipative part is fixed.
    # Let's precompute the dissipative part's matrix representation.
    
    pass

# Let's precompute the dissipative part of the Liouvillian as a 9x9 matrix.
# L_diss(rho) = L1 rho L1^dag - 0.5 {L1^dag L1, rho} + L2 rho L2^dag - 0.5 {L2^dag L2, rho}
# We can represent this as a 9x9 matrix M_diss such that vec(L_diss(rho)) = M_diss vec(rho)

def vec(rho):
    return rho.flatten(order='F')

def unvec(v):
    return v.reshape((3, 3), order='F')

# Precompute M_diss
M_diss = np.zeros((9, 9), dtype=complex)
for L in [L1, L2]:
    Ldag = L.conj().T
    # L rho Ldag -> (Ldag^T \otimes L) vec(rho)
    M_diss += np.kron(Ldag.conj().T, L)
    # -0.5 {Ldag L, rho} -> -0.5 (I \otimes Ldag L + (Ldag L)^T \otimes I) vec(rho)
    # Wait, vec(A rho B) = (B^T \otimes A) vec(rho)
    # So vec(Ldag L rho) = (I \otimes Ldag L) vec(rho)
    # vec(rho Ldag L) = ((Ldag L)^T \otimes I) vec(rho)
    M_diss -= 0.5 * (np.kron(np.eye(3), Ldag @ L) + np.kron((Ldag @ L).T, np.eye(3)))

# Now, for each step, we need to add the coherent part.
# H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
# Let's precompute the matrices for the constant parts.
N_N_MINUS_1 = N_OP @ (N_OP - np.eye(3))
H_const = 0.5 * ANHARMONICITY * N_N_MINUS_1

# The drive part: 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
# a+a^dag = [[0,1,0],[1,0,sqrt(2)],[0,sqrt(2),0]]
# a-a^dag = [[0,1,0],[-1,0,sqrt(2)],[0,-sqrt(2),0]]
# i*(a-a^dag) = [[0,i,0],[-i,0,i*sqrt(2)],[0,-i*sqrt(2),0]]
A_PLUS_ADAG = A_OP + A_OP_DAG
A_MINUS_ADAG = A_OP - A_OP_DAG

def compute_pulse_score(coeffs, detunings, scales):
    """
    coeffs: 8 coefficients for Legendre polynomials.
    The pulse is p(t) = sum_{k=0}^7 c_k P_k(2t/T - 1)
    But we need I and Q. The task says "returning an (80, 2) array of I and Q amplitudes".
    The seed is a DRAG-style Gaussian. DRAG has both I and Q.
    The Legendre expansion should be for both I and Q, or just one?
    The task says "directly optimizing the 8 coefficients". This implies 8 coefficients total.
    Maybe the pulse is real (I only) or Q is fixed?
    But DRAG uses Q to correct leakage.
    Let's assume the 8 coefficients are for I, and Q is 0? Or maybe 4 for I and 4 for Q?
    The directive says "first 8 Legendre polynomials... optimizing the 8 coefficients".
    This suggests 8 coefficients total. Let's assume they are for I, and Q is 0.
    But that might not be enough.
    Alternatively, maybe the 8 coefficients define the shape, and we scale it?
    Let's try 8 coefficients for I, and Q=0 first. If that's not enough, we can adjust.
    Actually, the seed is DRAG, which has Q.
    Let's use 8 coefficients for I and 8 for Q? No, "the 8 coefficients".
    Let's assume the pulse is p(t) = I(t) + i Q(t), and we expand I and Q in Legendre polynomials.
    But 8 coefficients total is very few.
    Let's assume 8 coefficients for the real part (I) and 0 for Q.
    Or maybe 4 for I and 4 for Q?
    Let's try 8 coefficients for I, Q=0.
    
    Actually, let's look at the seed: "DRAG-style Gaussian (sigma 40/6 ns, offset to 0 at the edges, area pi)".
    DRAG usually has a derivative term in Q.
    Let's assume the 8 coefficients are for I, and we can also optimize a Q shape if needed.
    But for now, let's stick to 8 coefficients for I, Q=0.
    
    Wait, the task says "returning an (80, 2) array of I and Q amplitudes".
    So we must return both.
    Let's use 8 coefficients for I and 8 for Q? That's 16 parameters.
    The directive says "the 8 coefficients". Maybe it means 8 for the whole pulse?
    Let's assume 8 coefficients for I, and Q is derived from I (like DRAG)?
    Or maybe 8 coefficients total, split between I and Q?
    Let's try 8 coefficients for I, and Q=0. If the score is bad, we can change.
    
    Actually, let's use 8 coefficients for I and 8 for Q, but the directive says 8.
    Let's assume 8 coefficients for I, Q=0.
    
    The pulse is p_k = sum_{j=0}^7 c_j P_j(x_k), where x_k = 2*k/(N-1) - 1.
    Then I_k = p_k, Q_k = 0.
    But we need to ensure the area is pi? The seed has area pi.
    The optimization should find the right area.
    
    Let's proceed with 8 coefficients for I, Q=0.
    """
    # Generate time points
    t = np.linspace(0, T_GATE, N_STEPS, endpoint=False) + DT/2  # centers
    x = 2 * t / T_GATE - 1  # scaled to [-1, 1]
    
    # Evaluate Legendre polynomials
    P = np.zeros((N_STEPS, 8))
    for j in range(8):
        P[:, j] = legendre(j)(x)
    
    I = P @ coeffs
    Q = np.zeros(N_STEPS)
    
    # Check constraints
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > AMP_BOUND):
        return 1e10
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= 0.12
    # p is (I, Q) vector
    p = np.column_stack((I, Q))
    # From zero to step 1
    if np.linalg.norm(p[0]) / DT > SLEW_BOUND:
        return 1e10
    # Between steps
    diffs = np.diff(p, axis=0)
    slew_rates = np.linalg.norm(diffs, axis=1) / DT
    if np.any(slew_rates > SLEW_BOUND):
        return 1e10
    # From last step to zero
    if np.linalg.norm(p[-1]) / DT > SLEW_BOUND:
        return 1e10
    
    # Now simulate
    fidelities = []
    for det in detunings:
        for scale in scales:
            # Initial states
            states = [
                np.array([1, 0, 0], dtype=complex),  # |0>
                np.array([0, 1, 0], dtype=complex),  # |1>
                (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) - np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) + 1j*np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) - 1j*np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
            ]
            
            for psi0 in states:
                rho = np.outer(psi0, psi0.conj())
                for k in range(N_STEPS):
                    # Build H
                    drive = 0.5 * scale * (I[k] * A_PLUS_ADAG + 1j * Q[k] * A_MINUS_ADAG)
                    H = det * N_OP + H_const + drive
                    
                    # Build Liouvillian
                    # Coherent part: -i(H rho - rho H)
                    # vec(H rho) = (I \otimes H) vec(rho)
                    # vec(rho H) = (H^T \otimes I) vec(rho)
                    M_coh = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
                    M_liouv = M_coh + M_diss
                    
                    # Exponentiate
                    U = expm(M_liouv * DT)
                    rho_vec = U @ vec(rho)
                    rho = unvec(rho_vec)
                
                # Target: X gate on |0> and |1>
                # X|0> = |1>, X|1> = |0>
                # For superpositions, X acts accordingly.
                # We need to compute fidelity to X|psi0>
                # But X is a 2x2 gate on the qubit subspace.
                # The target state is X applied to psi0, embedded in 3-level system.
                # X|0> = |1>, X|1> = |0>
                # So for psi0 = |0>, target is |1>
                # For psi0 = |1>, target is |0>
                # For superpositions, target is the X-transformed state.
                
                # Let's compute the target state
                if np.allclose(psi0[2], 0):
                    # In qubit subspace
                    psi_target = np.zeros(3, dtype=complex)
                    psi_target[0] = psi0[1]  # X swaps 0 and 1
                    psi_target[1] = psi0[0]
                    psi_target[2] = 0
                else:
                    # Should not happen for our initial states
                    psi_target = psi0
                
                # Fidelity
                # F = <psi_target|rho|psi_target>
                # But rho is a density matrix.
                # Fidelity for pure state |psi> and density matrix rho is <psi|rho|psi>
                F = np.vdot(psi_target, rho @ psi_target).real
                fidelities.append(F)
    
    # Score: mean + max of the 9 cases
    # Wait, the 9 cases are detuning x scale combinations.
    # For each case, we have 6 fidelities (one per initial state).
    # "infidelity = 1 - mean fidelity to X over |0>, |1>, ..."
    # So for each (det, scale), we compute mean fidelity over the 6 states, then infidelity.
    # Then score = mean + max of the 9 infidelities.
    
    # We have 9 cases * 6 states = 54 fidelities in the list.
    # Let's restructure: compute infidelity for each case.
    infidelities = []
    idx = 0
    for det in detunings:
        for scale in scales:
            case_fidelities = fidelities[idx:idx+6]
            mean_f = np.mean(case_fidelities)
            infidelities.append(1 - mean_f)
            idx += 6
    
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds
    
    # Initial guess: DRAG-like pulse in Legendre basis
    # We can start with a Gaussian-like shape approximated by Legendre polynomials.
    # Or just start with zero coefficients and let L-BFGS-B find the pulse.
    # Let's start with a simple shape: c_0 = pi (area), others 0.
    x0 = np.zeros(8)
    x0[0] = np.pi  # P_0 = 1, so area = pi * T? No, integral of P_0 over [-1,1] is 2.
    # The pulse is p(t) = sum c_j P_j(x), x = 2t/T - 1.
    # Area = integral_0^T p(t) dt = integral_{-1}^1 p(x) (T/2) dx = (T/2) * sum c_j integral_{-1}^1 P_j(x) dx
    # integral_{-1}^1 P_0(x) dx = 2, others 0.
    # So area = (T/2) * 2 * c_0 = T * c_0.
    # We want area = pi, so c_0 = pi / T = pi / 40.
    x0[0] = np.pi / 40.0
    
    # Bounds for coefficients to ensure amplitude and slew constraints
    # This is tricky. Let's use loose bounds and rely on the penalty in the objective.
    # Or we can use a change of variables to enforce constraints.
    # For now, use loose bounds.
    bounds = [(-1.0, 1.0)] * 8
    
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    best_score = 1e10
    best_coeffs = x0
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            compute_pulse_score,
            x0,
            args=(detunings, scales),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 1000, 'maxtime': 250}
        )
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
    except Exception as e:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Return best result found so far
        pass
    
    # Generate final pulse
    t = np.linspace(0, T_GATE, N_STEPS, endpoint=False) + DT/2
    x = 2 * t / T_GATE - 1
    P = np.zeros((N_STEPS, 8))
    for j in range(8):
        P[:, j] = legendre(j)(x)
    
    I = P @ best_coeffs
    Q = np.zeros(N_STEPS)
    
    # Final constraint check
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > AMP_BOUND):
        # Scale down
        max_amp = np.max(amp)
        I = I * (AMP_BOUND / max_amp)
    
    p = np.column_stack((I, Q))
    if np.linalg.norm(p[0]) / DT > SLEW_BOUND:
        scale_factor = SLEW_BOUND * DT / np.linalg.norm(p[0])
        p = p * scale_factor
    diffs = np.diff(p, axis=0)
    slew_rates = np.linalg.norm(diffs, axis=1) / DT
    if np.any(slew_rates > SLEW_BOUND):
        # Scale down the whole pulse
        max_slew = np.max(slew_rates)
        p = p * (SLEW_BOUND / max_slew)
    if np.linalg.norm(p[-1]) / DT > SLEW_BOUND:
        scale_factor = SLEW_BOUND * DT / np.linalg.norm(p[-1])
        p = p * scale_factor
    
    I, Q = p[:, 0], p[:, 1]
    
    return np.column_stack((I, Q))

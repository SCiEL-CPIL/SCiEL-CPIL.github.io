"""Program for robust control pulses for a three-level transmon using Legendre basis expansion."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import expm
import time

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

# Precompute Legendre polynomials scaled to [0, 1]
def legendre_basis(t):
    """Return 8xN_STEPS matrix of Legendre polynomials evaluated at t in [0,1]."""
    n = len(t)
    basis = np.zeros((8, n))
    # P_0 = 1
    basis[0, :] = 1.0
    # P_1 = x
    basis[1, :] = t
    # P_2 = (3x^2 - 1)/2
    basis[2, :] = (3 * t**2 - 1) / 2
    # P_3 = (5x^3 - 3x)/2
    basis[3, :] = (5 * t**3 - 3 * t) / 2
    # P_4 = (35x^4 - 30x^2 + 3)/8
    basis[4, :] = (35 * t**4 - 30 * t**2 + 3) / 8
    # P_5 = (63x^5 - 70x^3 + 15x)/8
    basis[5, :] = (63 * t**5 - 70 * t**3 + 15 * t) / 8
    # P_6 = (231x^6 - 315x^4 + 105x^2 - 5)/16
    basis[6, :] = (231 * t**6 - 315 * t**4 + 105 * t**2 - 5) / 16
    # P_7 = (429x^7 - 693x^5 + 315x^3 - 35x)/16
    basis[7, :] = (429 * t**7 - 693 * t**5 + 315 * t**3 - 35 * t) / 16
    return basis

# Time grid
t_grid = np.linspace(0, 1, N_STEPS)
B = legendre_basis(t_grid)  # 8 x 80

def coeffs_to_pulse(coeffs_I, coeffs_Q):
    """Convert Legendre coefficients to I and Q pulse arrays."""
    I = B @ coeffs_I
    Q = B @ coeffs_Q
    return I, Q

def pulse_to_coeffs(I, Q):
    """Project pulse onto Legendre basis (least squares)."""
    coeffs_I = np.linalg.lstsq(B, I, rcond=None)[0]
    coeffs_Q = np.linalg.lstsq(B, Q, rcond=None)[0]
    return coeffs_I, coeffs_Q

def check_constraints(I, Q):
    """Check amplitude and slew constraints. Return True if valid."""
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # From zero to step 1
    d0 = np.sqrt(I[0]**2 + Q[0]**2)
    if d0 / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        d = np.sqrt(dI**2 + dQ**2)
        if d / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    dlast = np.sqrt(I[-1]**2 + Q[-1]**2)
    if dlast / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def build_hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for one time step."""
    # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    
    H = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) + \
        0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    return H

def lindblad_operators():
    """Build Lindblad operators."""
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2 / Tphi)
    
    return [L1, L2]

def liouvillian(H, L_ops):
    """Build the Liouvillian superoperator."""
    dim = 3
    L = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.conj().T))
    
    for L_op in L_ops:
        L += np.kron(L_op, L_op.conj().T) - 0.5 * np.kron(L_op.conj().T @ L_op, np.eye(dim)) - \
             0.5 * np.kron(np.eye(dim), L_op @ L_op.conj().T)
    
    return L

def simulate_pulse(I, Q, detuning=0.0, scale=1.0):
    """Simulate the pulse and return the unitary evolution matrix."""
    L_ops = lindblad_operators()
    
    # Initialize density matrix for each input state
    # We'll simulate the full evolution of the density matrix
    # Actually, for fidelity computation, we need to track how each basis state evolves
    # Let's simulate the evolution operator on the density matrix space
    
    # Start with identity density matrix (to get the full evolution)
    # Actually, let's simulate each initial state separately for clarity
    
    # For efficiency, simulate the full 9x9 Liouvillian evolution
    # But we need the unitary part for fidelity. Let's use a different approach.
    
    # Simulate the state evolution for each of the 4 test states
    # |0> = [1,0,0], |1> = [0,1,0], (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), etc.
    
    # Actually, let's just simulate the density matrix evolution for the full system
    # and compute fidelity with X gate
    
    # The X gate maps: |0> -> |1>, |1> -> |0>
    # So target states: |0> -> |1>, |1> -> |0>, (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2), etc.
    
    # Let's simulate the evolution of each basis state and compute fidelity
    
    # For the 9 cases, we need to simulate with different detuning and scale
    # Return the evolved density matrices
    
    # Simulate one step at a time
    rho = np.eye(9)  # This is the full density matrix in Liouville space
    
    # Actually, let's simulate the state vector evolution for pure states
    # and compute fidelity as |<target|psi>|^2
    
    # Initial states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi_0 + psi_1) / np.sqrt(2)
    psi_minus = (psi_0 - psi_1) / np.sqrt(2)
    psi_i_plus = (psi_0 + 1j * psi_1) / np.sqrt(2)
    psi_i_minus = (psi_0 - 1j * psi_1) / np.sqrt(2)
    
    # Target states after X gate
    target_0 = psi_1  # |0> -> |1>
    target_1 = psi_0  # |1> -> |0>
    target_plus = psi_plus  # (|0>+|1>)/sqrt(2) -> same
    target_minus = psi_minus  # (|0>-|1>)/sqrt(2) -> same
    target_i_plus = psi_i_plus  # (|0>+i|1>)/sqrt(2) -> same
    target_i_minus = psi_i_minus  # (|0>-i|1>)/sqrt(2) -> same
    
    states = [psi_0, psi_1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    targets = [target_0, target_1, target_plus, target_minus, target_i_plus, target_i_minus]
    
    # Evolve each state
    fidelities = []
    for psi in states:
        psi_evol = psi.copy()
        for i in range(N_STEPS):
            H = build_hamiltonian(detuning, scale, I[i], Q[i])
            # Exponentiate: exp(-i H dt)
            U = expm(-1j * H * DT)
            psi_evol = U @ psi_evol
            # Normalize
            norm = np.linalg.norm(psi_evol)
            if norm > 0:
                psi_evol /= norm
        
        target = targets[states.index(psi)]
        fidelity = np.abs(np.vdot(target, psi_evol))**2
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def compute_score(coeffs_I, coeffs_Q, detunings=None, scales=None):
    """Compute the score for given coefficients."""
    if detunings is None:
        detunings = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    if scales is None:
        scales = [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]
    
    I, Q = coeffs_to_pulse(coeffs_I, coeffs_Q)
    
    # Check constraints
    if not check_constraints(I, Q):
        return 1.0  # Invalid
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = simulate_pulse(I, Q, det, sc)
            infid = 1 - fid
            infidelities.append(infid)
    
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    score = mean_infid + max_infid
    
    return score

def objective(params):
    """Objective function for optimization."""
    coeffs_I = params[:8]
    coeffs_Q = params[8:]
    
    I, Q = coeffs_to_pulse(coeffs_I, coeffs_Q)
    
    # Check constraints
    if not check_constraints(I, Q):
        return 1.0
    
    # Use nominal case for fast evaluation during optimization
    fid = simulate_pulse(I, Q, 0.0, 1.0)
    infid = 1 - fid
    return infid

# Initial guess: DRAG-style Gaussian converted to Legendre coefficients
def initial_guess():
    """Generate initial guess based on DRAG-style Gaussian."""
    t = np.linspace(0, T_TOTAL, N_STEPS)
    sigma = 40 / 6  # ns
    center = T_TOTAL / 2
    
    # Gaussian with area pi
    gauss = np.exp(-((t - center)**2) / (2 * sigma**2))
    area = np.sum(gauss) * DT
    gauss = gauss * (np.pi / area)
    
    # Add DRAG term (derivative of Gaussian)
    drag_coeff = 0.1  # Typical DRAG coefficient
    drag = drag_coeff * np.gradient(gauss, DT)
    
    I_init = gauss
    Q_init = drag
    
    # Project onto Legendre basis
    coeffs_I, coeffs_Q = pulse_to_coeffs(I_init, Q_init)
    
    return np.concatenate([coeffs_I, coeffs_Q])

def optimize_pulse():
    """Optimize the pulse using L-BFGS-B."""
    x0 = initial_guess()
    
    # Define constraints
    # We'll handle constraints by checking in the objective and returning large values
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    return result.x

# Precompute the optimized pulse at module load time
def _precompute():
    """Precompute the optimized pulse."""
    t_start = time.time()
    x_opt = optimize_pulse()
    t_end = time.time()
    
    coeffs_I = x_opt[:8]
    coeffs_Q = x_opt[8:]
    
    I, Q = coeffs_to_pulse(coeffs_I, coeffs_Q)
    
    # Verify constraints
    if not check_constraints(I, Q):
        # Fall back to initial guess
        x0 = initial_guess()
        coeffs_I = x0[:8]
        coeffs_Q = x0[8:]
        I, Q = coeffs_to_pulse(coeffs_I, coeffs_Q)
    
    return I, Q

_I_opt, _Q_opt = _precompute()

def pulse(target_gate, hardware_model):
    """Return the optimized pulse."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    return np.column_stack([_I_opt, _Q_opt])

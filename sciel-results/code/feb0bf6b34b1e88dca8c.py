"""Robust control pulses for a three-level transmon using DRAG-based direct optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0  # Tphi

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Build ladder operators for 3-level system
def build_operators():
    """Build a, a_dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

def hamiltonian(detuning, scale, I, Q):
    """Build the Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    drive_term = 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    H += drive_term
    return H

def liouvillian_step(rho, H, dt):
    """Apply one step of the Liouvillian evolution using matrix exponential."""
    U = expm(-1j * H * dt)
    rho = U @ rho @ U.conj().T
    
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        rho = rho - 0.5 * dt * (LdagL @ rho + rho @ LdagL)
        rho = rho + dt * L @ rho @ L.conj().T
    
    return rho

def fidelity_with_X(rho):
    """Compute mean fidelity to X gate over the 6 test states."""
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    fidelities = []
    for psi in states:
        psi_target = np.zeros(3, dtype=complex)
        psi_target[0] = psi[1]
        psi_target[1] = psi[0]
        psi_target[2] = psi[2]
        
        psi_final = rho @ psi
        overlap = np.vdot(psi_target, psi_final)
        fid = np.abs(overlap) ** 2
        fidelities.append(fid)
    
    return np.mean(fidelities)

def evolve_pulse(pulse, detuning, scale):
    """Evolve the system with the given pulse and return final density matrix."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        H = hamiltonian(detuning, scale, I[i], Q[i])
        rho = liouvillian_step(rho, H, DT)
    
    return rho

def compute_score(pulse):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            rho = evolve_pulse(pulse, det, sc)
            fid = fidelity_with_X(rho)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def check_validity(pulse):
    """Check if the pulse satisfies amplitude and slew constraints."""
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    slew0 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
    
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_last > SLEW_BOUND + 1e-10:
        return False
    
    return True

def generate_drag_pulse(sigma, area, drag_coeff, drag_sigma, offset):
    """Generate a DRAG pulse with Gaussian envelope and derivative correction.
    
    Args:
        sigma: Standard deviation of main Gaussian (ns)
        area: Total area of the pulse (should be ~pi for X-pi gate)
        drag_coeff: Coefficient for DRAG correction
        drag_sigma: Standard deviation for DRAG correction (ns)
        offset: Time offset for centering (ns)
    """
    t = np.arange(N_STEPS) * DT - TOTAL_TIME / 2.0 + offset
    
    # Main Gaussian envelope
    gauss = np.exp(-0.5 * (t / sigma) ** 2)
    
    # Scale to achieve desired area
    gauss_area = np.sum(gauss) * DT
    I = area * gauss / gauss_area
    
    # DRAG correction: proportional to derivative of Gaussian
    # d/dt exp(-t^2/(2*drag_sigma^2)) = -(t/drag_sigma^2) * exp(...)
    drag_gauss = np.exp(-0.5 * (t / drag_sigma) ** 2)
    Q = drag_coeff * (t / (drag_sigma ** 2)) * drag_gauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce constraints
    pulse = _enforce_constraints(pulse)
    
    return pulse

def _enforce_constraints(pulse):
    """Enforce amplitude and slew constraints with iterative scaling."""
    pulse = pulse.copy()
    
    for _ in range(10):  # Iterate to converge
        # Amplitude bound
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > AMPLITUDE_BOUND:
            pulse *= AMPLITUDE_BOUND / max_amp
        
        # Slew from 0 to first step
        slew0 = np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2) / DT
        if slew0 > SLEW_BOUND:
            pulse *= SLEW_BOUND * DT / np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2)
        
        # Slew between steps
        d_pulse = np.diff(pulse, axis=0)
        slews = np.sqrt(d_pulse[:, 0]**2 + d_pulse[:, 1]**2) / DT
        max_slew = np.max(slews)
        if max_slew > SLEW_BOUND:
            pulse *= SLEW_BOUND / max_slew
        
        # Slew from last step to 0
        slew_last = np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2) / DT
        if slew_last > SLEW_BOUND:
            pulse *= SLEW_BOUND * DT / np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2)
        
        # Check if all constraints satisfied
        amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
        if np.max(amps) <= AMPLITUDE_BOUND + 1e-10:
            slew0 = np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2) / DT
            if slew0 <= SLEW_BOUND + 1e-10:
                d_pulse = np.diff(pulse, axis=0)
                slews = np.sqrt(d_pulse[:, 0]**2 + d_pulse[:, 1]**2) / DT
                if np.max(slews) <= SLEW_BOUND + 1e-10:
                    slew_last = np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2) / DT
                    if slew_last <= SLEW_BOUND + 1e-10:
                        break
    
    return pulse

def optimize_pulse_direct():
    """Optimize pulse parameters using DRAG theory and numerical refinement."""
    def objective(params):
        sigma, area, drag_coeff, drag_sigma, offset = params
        pulse = generate_drag_pulse(sigma, area, drag_coeff, drag_sigma, offset)
        if not check_validity(pulse):
            return 1e6
        return compute_score(pulse)
    
    # Bounds for parameters
    # sigma: width of Gaussian. 40ns total. 3-12ns is reasonable.
    # area: integral of I. ~ pi for pi-pulse.
    # drag_coeff: usually small.
    # drag_sigma: width of derivative Gaussian.
    # offset: time shift.
    bounds = [
        (4.0, 10.0),      # sigma
        (2.5, 3.5),       # area
        (-0.2, 0.2),      # drag_coeff
        (3.0, 8.0),       # drag_sigma
        (-2.0, 2.0),      # offset
    ]
    
    # Use Nelder-Mead with restarts for robust optimization
    # Keep runtime reasonable by limiting iterations
    result = minimize(objective, np.array([6.0, np.pi, 0.0, 5.0, 0.0]), 
                      method='Nelder-Mead', bounds=bounds,
                      options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
    
    best_params = result.x
    best_score = result.fun
    
    # Try multiple restarts with different initial conditions
    rng = np.random.default_rng(42)
    for _ in range(10):
        x_rand = np.array([
            rng.uniform(4.0, 10.0),  # sigma
            rng.uniform(2.5, 3.5),   # area
            rng.uniform(-0.2, 0.2),  # drag_coeff
            rng.uniform(3.0, 8.0),   # drag_sigma
            rng.uniform(-2.0, 2.0)   # offset
        ])
        
        res_rand = minimize(objective, x_rand, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
        if res_rand.fun < best_score:
            best_score = res_rand.fun
            best_params = res_rand.x
    
    return generate_drag_pulse(*best_params)

# Generate the optimized pulse at module load time
_optimized_pulse = optimize_pulse_direct()

def pulse(target_gate, hardware_model):
    """Return the optimized pulse for the given target gate and hardware model."""
    global _optimized_pulse
    return _optimized_pulse.copy()
# EVOLVE-BLOCK-END

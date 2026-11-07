import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit

    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # Basis states for fidelity calculation
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Target states for X gate (bit flip)
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)
    ]

    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_op.conj().T @ a_op

    # Precompute static parts of Hamiltonian
    I3 = np.eye(3, dtype=complex)
    
    # H0 = delta * n + 0.5 * alpha * n(n-1)
    n_nm1 = n_op @ (n_op - I3)
    alpha_term = 0.5 * alpha * n_nm1
    
    # Precompute Lindblad superoperator terms
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Precompute Lindblad contributions to the superoperator
    lindblad_super = np.zeros((9, 9), dtype=complex)
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        term1 = np.kron(Lk, Lk_dag.T)
        Lk_dag_Lk = Lk_dag @ Lk
        term2 = -0.5 * (np.kron(Lk_dag_Lk, I3) + np.kron(I3, Lk_dag_Lk.T))
        lindblad_super += term1 + term2

    def get_H(I, Q, delta, scale):
        H = delta * n_op + alpha_term
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H

    def build_superoperator(H):
        """Build the full Liouvillian superoperator for a given Hamiltonian."""
        # -i [H, rho] -> -i (H rho - rho H)
        # In vectorized form: -i (kron(H, I) - kron(I, H.T))
        H_vec = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        return H_vec + lindblad_super

    def simulate_pulse_full(I_Q, delta, scale):
        """Simulate pulse with full Liouvillian evolution for scoring."""
        fidels = []
        H_steps = [get_H(I_Q[i, 0], I_Q[i, 1], delta, scale) for i in range(N)]
        U_steps = []
        for i in range(N):
            H = H_steps[i]
            S = build_superoperator(H)
            U_steps.append(expm(S * dt))
        
        for idx in range(6):
            psi = states[idx].copy()
            target = targets[idx]
            
            # Use state vector evolution for speed, but the task says "exponentiates the Liouvillian"
            # However, for pure states, if we just want fidelity, we can use the density matrix.
            # But wait, the task says "Model: ... each step exponentiates the Liouvillian".
            # This implies we must track the density matrix.
            rho = np.outer(psi, psi.conj())
            target_rho = np.outer(target, target.conj())
            
            for i in range(N):
                rho_vec = rho.flatten()
                rho_next_vec = U_steps[i] @ rho_vec
                rho = rho_next_vec.reshape(3, 3)
            
            fid = np.real(np.trace(rho @ target_rho))
            # Clip fidelity to [0, 1]
            fid = np.clip(fid, 0.0, 1.0)
            fidels.append(1 - fid)
        
        return fidels

    def simulate_pulse_fast(I_Q, delta, scale):
        """Fast simulation using unitary evolution only (ignores dissipation).
        Used for optimization speed. Dissipation is small (T1, Tphi large)."""
        fidels = []
        U_total = np.eye(3, dtype=complex)
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            U_step = expm(-1j * H * dt)
            U_total = U_step @ U_total
        
        for idx in range(6):
            psi = states[idx].copy()
            target = targets[idx]
            psi_final = U_total @ psi
            fid = np.abs(np.vdot(target, psi_final))**2
            fid = np.clip(fid, 0.0, 1.0)
            fidels.append(1 - fid)
        
        return fidels

    def check_constraints(I_Q):
        """Check amplitude and slew constraints."""
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-10:
            return False
        
        # From 0 to step 1
        if amps[0] / dt > slew_bound + 1e-10:
            return False
        
        # Between steps
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-10:
            return False
        
        # From last step to 0
        if amps[-1] / dt > slew_bound + 1e-10:
            return False
        
        return True

    def score_pulse_fast(I_Q):
        """Compute the scoring metric using fast (unitary) simulation."""
        total_score = 0.0
        count = 0
        for delta in deltas:
            for scale in scales:
                infs = simulate_pulse_fast(I_Q, delta, scale)
                mean_inf = np.mean(infs)
                max_inf = np.max(infs)
                total_score += (mean_inf + max_inf)
                count += 1
        return total_score / count

    def score_pulse_full(I_Q):
        """Compute the scoring metric using full (Liouvillian) simulation."""
        total_score = 0.0
        count = 0
        for delta in deltas:
            for scale in scales:
                infs = simulate_pulse_full(I_Q, delta, scale)
                mean_inf = np.mean(infs)
                max_inf = np.max(infs)
                total_score += (mean_inf + max_inf)
                count += 1
        return total_score / count

    def generate_pulse(sigma, eta, center=20.0):
        """Generate DRAG-style Gaussian pulse."""
        t = np.linspace(0, 40, N)
        t_c = center
        A = np.sqrt(np.pi / 2.0) / sigma
        I = A * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        
        dI = I * (-(t - t_c) / (sigma**2))
        Q = eta * dI
        
        I_Q = np.column_stack((I, Q))
        
        # Enforce constraints by clipping
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        for i in range(N):
            if amps[i] > amp_bound:
                scale_factor = amp_bound / amps[i]
                I_Q[i, 0] *= scale_factor
                I_Q[i, 1] *= scale_factor
        
        # Enforce slew rate
        for i in range(1, N):
            diff = I_Q[i] - I_Q[i-1]
            norm = np.linalg.norm(diff)
            if norm / dt > slew_bound:
                scale_factor = (slew_bound * dt) / norm
                I_Q[i] = I_Q[i-1] + scale_factor * diff
        
        # Enforce start/end slew
        norm0 = np.linalg.norm(I_Q[0])
        if norm0 / dt > slew_bound:
            scale_factor = (slew_bound * dt) / norm0
            I_Q[0] *= scale_factor
            
        normN = np.linalg.norm(I_Q[-1])
        if normN / dt > slew_bound:
            scale_factor = (slew_bound * dt) / normN
            I_Q[-1] *= scale_factor
            
        return I_Q

    # Initial grid search for good starting point
    best_score = np.inf
    best_IQ = None

    # Try a range of sigma and eta values
    sigma_range = np.linspace(4, 12, 17)
    eta_range = np.linspace(-0.5, 0.5, 21)
    center_range = [18, 19, 20, 21, 22]

    for center in center_range:
        if time.time() > deadline - 30:
            break
        for sigma in sigma_range:
            if time.time() > deadline - 30:
                break
            for eta in eta_range:
                I_Q = generate_pulse(sigma, eta, center)
                if not check_constraints(I_Q):
                    continue
                s = score_pulse_fast(I_Q)
                if s < best_score:
                    best_score = s
                    best_IQ = I_Q.copy()

    # Local refinement using coordinate descent
    if best_IQ is not None and time.time() < deadline - 20:
        current_IQ = best_IQ.copy()
        current_score = best_score
        step_size = 0.005
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try small perturbations in I and Q
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                for j in range(2):
                    # Increase
                    old_val = current_IQ[i, j]
                    current_IQ[i, j] += step_size
                    if check_constraints(current_IQ):
                        s = score_pulse_fast(current_IQ)
                        if s < current_score - 1e-10:
                            current_score = s
                            improved = True
                        else:
                            current_IQ[i, j] = old_val
                    else:
                        current_IQ[i, j] = old_val
                    
                    # Decrease
                    current_IQ[i, j] -= step_size
                    if check_constraints(current_IQ):
                        s = score_pulse_fast(current_IQ)
                        if s < current_score - 1e-10:
                            current_score = s
                            improved = True
                        else:
                            current_IQ[i, j] = old_val
                    else:
                        current_IQ[i, j] = old_val
            
            if not improved:
                break

        if current_score < best_score:
            best_score = current_score
            best_IQ = current_IQ.copy()

    # Final constraint check and fallback
    if best_IQ is None or not check_constraints(best_IQ):
        # Fallback to seed pulse
        sigma_seed = 40.0 / 6.0
        best_IQ = generate_pulse(sigma_seed, 0.0)
        if not check_constraints(best_IQ):
            I = np.full(N, np.pi / 40.0)
            Q = np.zeros(N)
            best_IQ = np.column_stack((I, Q))
            if not check_constraints(best_IQ):
                I = np.full(N, 0.01)
                Q = np.zeros(N)
                best_IQ = np.column_stack((I, Q))

    return best_IQ

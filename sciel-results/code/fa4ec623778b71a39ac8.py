"""Robust X_pi control pulse for a three-level transmon using adjoint-based gradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    N = 80  # number of steps
    total_time = N * dt  # 40 ns
    
    # Build the 3-level Hamiltonian components
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator diag(0, 1, 2)
    I3 = np.eye(3)
    
    # Precompute Liouvillian components
    # H = detuning*n + 0.5*anharm*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    # dH/dI = 0.5*scale*(a + a_dag)
    # dH/dQ = 0.5*scale*i*(a - a_dag)
    # dH/d(detuning) = n
    # dH/d(scale) = 0.5*(I*(a+a_dag) + i*Q*(a-a_dag))
    
    def build_hamiltonian(I, Q, detuning, scale):
        """Build the Hamiltonian for given I, Q, detuning, scale."""
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - I3)
        H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return H
    
    def build_liouvillian_from_H(H):
        """Build the Liouvillian superoperator from Hamiltonian H."""
        L1 = a / np.sqrt(T1)
        L2 = n_op * np.sqrt(2.0 / Tphi)
        
        H_left = np.kron(I3, H)
        H_right = np.kron(H.T, I3)
        
        L1_dag_L1 = L1.conj().T @ L1
        L2_dag_L2 = L2.conj().T @ L2
        
        L1_term = np.kron(L1, L1.T)
        L2_term = np.kron(L2, L2.T)
        
        L1_diss_left = np.kron(I3, L1_dag_L1)
        L1_diss_right = np.kron(L1_dag_L1.T, I3)
        
        L2_diss_left = np.kron(I3, L2_dag_L2)
        L2_diss_right = np.kron(L2_dag_L2.T, I3)
        
        Liouvillian = -1j * (H_left - H_right) + L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                        + L2_term - 0.5 * (L2_diss_left + L2_diss_right)
        
        return Liouvillian
    
    def simulate_pulse_forward(pulse_IQ, detuning, scale):
        """
        Simulate the pulse forward and return the final density matrix.
        Also returns intermediate states for adjoint computation.
        """
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        
        states = [rho.copy()]
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            H = build_hamiltonian(I_val, Q_val, detuning, scale)
            L = build_liouvillian_from_H(H)
            
            U = expm(dt * L)
            rho_vec = rho.flatten()
            rho_vec_new = U @ rho_vec
            rho = rho_vec_new.reshape(3, 3)
            states.append(rho.copy())
        
        return states
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        psi = target_state.flatten()
        return np.real(np.conj(psi) @ rho @ psi)
    
    def compute_infidelity_and_states(pulse_IQ, detunings, scales, states_test):
        """
        Compute infidelity over all cases.
        Returns (infidelities_array, all_states) where all_states is a list of state lists for each case.
        """
        all_infidelities = []
        all_states = []
        
        for det in detunings:
            for scale in scales:
                states = simulate_pulse_forward(pulse_IQ, det, scale)
                rho_final = states[-1]
                
                case_infs = []
                for state in states_test:
                    # Apply X_pi to the state: -i * swap |0> and |1>
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    
                    fid = compute_fidelity(rho_final, target)
                    case_infs.append(1.0 - fid)
                
                all_infidelities.extend(case_infs)
                all_states.append(states)
        
        return np.array(all_infidelities), all_states
    
    def compute_gradient_adjoint(pulse_IQ, detunings, scales, states_test, all_states):
        """
        Compute gradient of the score (mean + max infidelity) with respect to pulse parameters
        using adjoint method.
        """
        N_cases = len(all_states)
        # all_states[k] is a list of N+1 states for case k
        # all_infidelities[k*6:(k+1)*6] are the 6 infidelities for case k
        
        # Compute which case-state has max infidelity
        # infidelities is flattened: [case0_state0, case0_state1, ..., case8_state5]
        # Each case has 6 states
        infidelities_flat = []
        for k in range(N_cases):
            states_k = all_states[k]
            rho_final = states_k[-1]
            for state in states_test:
                target = np.zeros(3, dtype=complex)
                target[0] = -1j * state[1]
                target[1] = -1j * state[0]
                target[2] = state[2]
                fid = compute_fidelity(rho_final, target)
                infidelities_flat.append(1.0 - fid)
        
        infidelities_flat = np.array(infidelities_flat)
        mean_inf = np.mean(infidelities_flat)
        max_idx = np.argmax(infidelities_flat)
        max_inf = infidelities_flat[max_idx]
        
        # Gradient of score = mean + max
        # d(score)/d(infid_j) = 1/len(infids) + delta_{j,max_idx}
        weights = np.ones(len(infidelities_flat)) / len(infidelities_flat)
        weights[max_idx] += 1.0
        
        # Now compute gradient with respect to pulse parameters
        # For each case k, we have states_k[0..N]
        # The adjoint propagator lambda_k[i] is the gradient of the final cost with respect to state at time i
        # For the Liouvillian evolution: rho_{i+1} = expm(dt*L_i) @ rho_i
        # d(rho_{i+1}) = expm(dt*L_i) @ d(rho_i) + dt * dL_i @ rho_i (first order)
        # But for exact gradient, we need to be more careful.
        
        # Actually, let's use the chain rule through the density matrix evolution.
        # The cost for case k is C_k = sum_j w_{k,j} * (1 - F_j(rho_final_k))
        # where F_j = <psi_j|rho|psi_j>
        # dC_k/d(rho_final) = -sum_j w_{k,j} * |psi_j><psi_j| (in vectorized form)
        
        # For the adjoint: lambda_k[N] = dC_k/d(rho_N)
        # lambda_k[i] = expm(dt * L_i^dag) @ lambda_k[i+1]  (for the linear part)
        # But the Hamiltonian depends on pulse parameters, so we need:
        # dC_k/d(pulse[i]) = dt * <lambda_k[i+1], dL_i/d(pulse[i]) @ rho_i>
        
        # Let's implement this properly.
        # rho is a 3x3 matrix, we work in vectorized form (9-dim)
        
        # Precompute dL/dI, dL/dQ for each step
        # dL/dI = -1j*(kron(I3, dH/dI) - kron((dH/dI).T, I3))
        # where dH/dI = 0.5*scale*(a+a_dag)
        
        grad_I = np.zeros(N)
        grad_Q = np.zeros(N)
        
        for k in range(N_cases):
            states_k = all_states[k]
            det = detunings[k // 3]
            scale = scales[k % 3]
            
            # Compute the weight vector for this case
            # Case k has states_test indices: k*6 to k*6+5
            case_weights = weights[k*6:(k+1)*6]
            
            # Compute lambda[N] = dC/d(rho_N) in vectorized form
            # C_k = sum_j case_weights[j] * (1 - <psi_j|rho_N|psi_j>)
            # dC_k/d(rho_N) = -sum_j case_weights[j] * |psi_j><psi_j|
            # In vectorized form: vec(dC/d(rho)) = -sum_j case_weights[j] * kron(|psi_j><psi_j|^T, I) ... 
            # Actually, for pure state fidelity F = <psi|rho|psi>, dF/d(rho) = |psi><psi|
            # So dC/d(rho) = -sum_j w_j |psi_j><psi_j|
            
            lambda_vec = np.zeros(9, dtype=complex)
            for j in range(6):
                state = states_test[j]
                psi = state.flatten()
                # |psi><psi| is a 3x3 matrix
                psi_psi_dag = np.outer(psi, psi.conj())
                # vec of psi_psi_dag
                lambda_vec -= case_weights[j] * psi_psi_dag.flatten()
            
            # Now propagate adjoint backward
            # lambda[i] = (dL_i/dt)^dag @ lambda[i+1]  ... wait
            # rho_{i+1} = expm(dt*L_i) @ rho_i
            # d(rho_{i+1}) = expm(dt*L_i) @ d(rho_i)
            # So dC/d(rho_i) = expm(dt*L_i)^dag @ dC/d(rho_{i+1})
            # But L_i depends on pulse[i], so there's also a direct term.
            
            # Let's define lambda[i] = dC/d(rho_i) as a 9-dim vector
            # lambda[N] is computed above
            # lambda[i] = (expm(dt*L_i))^dag @ lambda[i+1]
            # dC/d(pulse[i]) = dt * (lambda[i+1] @ dL_i/d(pulse[i]) @ vec(rho_i))  ... 
            # Actually, d(rho_{i+1})/d(pulse[i]) = dt * dL_i/d(pulse[i]) @ rho_i (first order, since rho_i doesn't depend on pulse[i])
            # So dC/d(pulse[i]) = lambda[i+1]^T @ d(rho_{i+1})/d(pulse[i]) = dt * lambda[i+1]^T @ (dL_i/d(pulse[i]) @ vec(rho_i))
            
            # Let's implement step by step
            lambda_current = lambda_vec.copy()
            
            for i in range(N - 1, -1, -1):
                I_val = pulse_IQ[i, 0]
                Q_val = pulse_IQ[i, 1]
                
                H = build_hamiltonian(I_val, Q_val, det, scale)
                L = build_liouvillian_from_H(H)
                
                # Compute dL/dI and dL/dQ
                dH_dI = 0.5 * scale * (a + a_dag)
                dH_dQ = 0.5 * scale * 1j * (a - a_dag)
                
                # dL/dI = -1j*(kron(I3, dH_dI) - kron(dH_dI.T, I3))
                dL_dI = -1j * (np.kron(I3, dH_dI) - np.kron(dH_dI.T, I3))
                dL_dQ = -1j * (np.kron(I3, dH_dQ) - np.kron(dH_dQ.T, I3))
                
                # rho_i in vectorized form
                rho_i_vec = states_k[i].flatten()
                
                # dC/d(pulse_I[i]) = dt * lambda_current^T @ (dL_dI @ rho_i_vec)
                # lambda_current is dC/d(rho_{i+1}), and d(rho_{i+1})/d(I_i) = dt * dL_dI @ rho_i
                # So dC/d(I_i) = dt * lambda_current @ (dL_dI @ rho_i_vec)
                
                dLdI_rho = dL_dI @ rho_i_vec
                dLdQ_rho = dL_dQ @ rho_i_vec
                
                grad_I[i] += dt * np.real(np.conj(lambda_current) @ dLdI_rho)
                grad_Q[i] += dt * np.real(np.conj(lambda_current) @ dLdQ_rho)
                
                # Propagate adjoint: lambda[i] = expm(dt*L)^dag @ lambda[i+1]
                U = expm(dt * L)
                lambda_current = (U.conj().T) @ lambda_current
        
        return np.column_stack([grad_I, grad_Q])
    
    def check_validity(pulse_IQ):
        """Check if the pulse satisfies all constraints."""
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-12:
                return False
            prev = curr
        
        diff = np.linalg.norm(prev - np.zeros(2))
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True
    
    # Test states
    states_test = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>-i|1>)/sqrt(2)
    ]
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Create initial DRAG pulse
    def make_drag_pulse(alpha=0.0):
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        I = gaussian.copy()
        
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        
        pulse_IQ = np.column_stack([I, Q])
        
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                scale_factor = amplitude_bound / max_amp
                pulse_IQ *= scale_factor
        
        return pulse_IQ
    
    # Start with DRAG pulse
    best_pulse = make_drag_pulse(0.0)
    
    # Compute initial score
    infid, all_states = compute_infidelity_and_states(best_pulse, detunings, scales, states_test)
    best_score = np.mean(infid) + np.max(infid)
    
    start_time = time.time()
    time_limit = 300.0
    
    # Adjoint-based gradient descent
    for iteration in range(50):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        
        # Compute gradient using adjoint method
        grad = compute_gradient_adjoint(best_pulse, detunings, scales, states_test, all_states)
        
        # Gradient descent step
        learning_rate = 0.01
        new_pulse = best_pulse - learning_rate * grad
        
        # Project to feasible region
        new_pulse[:, 0] = np.clip(new_pulse[:, 0], -amplitude_bound, amplitude_bound)
        new_pulse[:, 1] = np.clip(new_pulse[:, 1], -amplitude_bound, amplitude_bound)
        
        # Check validity
        if check_validity(new_pulse):
            new_infid, _ = compute_infidelity_and_states(new_pulse, detunings, scales, states_test)
            new_score = np.mean(new_infid) + np.max(new_infid)
            
            if new_score < best_score:
                best_score = new_score
                best_pulse = new_pulse.copy()
                # Recompute all_states for next iteration
                _, all_states = compute_infidelity_and_states(best_pulse, detunings, scales, states_test)
            else:
                # Reduce learning rate
                learning_rate *= 0.5
        else:
            learning_rate *= 0.5
    
    return best_pulse
# EVOLVE-BLOCK-END

"""Rewritten program for robust control pulses for a three-level transmon."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Constants
    dt = 0.5  # ns
    N_steps = 80
    T_total = N_steps * dt  # 40 ns
    
    # Hardware parameters from contract
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # Simulation parameters
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Time points (center of each step)
    t = np.arange(N_steps) * dt + dt / 2.0
    t_center = T_total / 2.0

    def generate_pulse(params):
        amp, drag, sigma = params
        # Gaussian envelope
        I = amp * np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
        # DRAG component
        dI_dt = np.gradient(I, dt)
        Q = -drag * dI_dt
        
        # Enforce amplitude bound
        max_amp = np.sqrt(I**2 + Q**2)
        if np.max(max_amp) > amp_bound:
            scale_factor = amp_bound / np.max(max_amp)
            I *= scale_factor
            Q *= scale_factor
            
        # Enforce slew rate bound
        I_ext = np.concatenate([[0.0], I, [0.0]])
        Q_ext = np.concatenate([[0.0], Q, [0.0]])
        dI = np.diff(I_ext) / dt
        dQ = np.diff(Q_ext) / dt
        slew = np.sqrt(dI**2 + dQ**2)
        
        if np.max(slew) > slew_bound:
            scale_factor = slew_bound / np.max(slew)
            I *= scale_factor
            Q *= scale_factor
            
        # Re-check amplitude after slew scaling
        max_amp = np.sqrt(I**2 + Q**2)
        if np.max(max_amp) > amp_bound:
            scale_factor = amp_bound / np.max(max_amp)
            I *= scale_factor
            Q *= scale_factor
            
        return np.column_stack((I, Q))

    def simulate_single_case(I, Q, detuning, scale):
        # 3-level system
        # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
        # a_dag = a.T
        # n = a_dag @ a = diag(0, 1, 2)
        # H = detuning * n + 0.5 * anharmonicity * n * (n-1) + 0.5 * scale * (I * (a + a_dag) + i * Q * (a - a_dag))
        
        # Precompute matrices
        sqrt1 = np.sqrt(1.0)
        sqrt2 = np.sqrt(2.0)
        
        a = np.array([
            [0, sqrt1, 0],
            [0, 0, sqrt2],
            [0, 0, 0]
        ], dtype=complex)
        
        a_dag = a.conj().T
        n = a_dag @ a # diag(0, 1, 2)
        
        # Lindblad operators
        # L1 = a / sqrt(T1)
        # L2 = n * sqrt(2 / Tphi)
        # Note: n is diagonal, so L2 is diagonal
        L1 = a / np.sqrt(T1)
        L2 = n * np.sqrt(2.0 / Tphi)
        
        # Liouvillian superoperator L
        # d rho / dt = -i [H, rho] + sum (L rho L^dag - 0.5 {L^dag L, rho})
        # We can construct the superoperator matrix or just step it.
        # Stepping is easier for variable H(t).
        
        # Initial states
        # |0> = [1, 0, 0]
        # |1> = [0, 1, 0]
        # |+> = (|0>+|1>)/sqrt(2)
        # |-> = (|0>-|1>)/sqrt(2)
        # |+i> = (|0>+i|1>)/sqrt(2)
        # |-i> = (|0>-i|1>)/sqrt(2)
        
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1], dtype=complex) / np.sqrt(2),
            np.array([1, -1], dtype=complex) / np.sqrt(2),
            np.array([1, 1j], dtype=complex) / np.sqrt(2),
            np.array([1, -1j], dtype=complex) / np.sqrt(2)
        ]
        
        # Target is X gate: |0> -> |1>, |1> -> |0>
        # So we check fidelity of final state to X(initial_state)
        # X|0> = |1>, X|1> = |0>, X|+> = |+>, X|-> = |->, X|+i> = |-i>, X|-i> = |+i>
        # Wait, X maps |0> to |1> and |1> to |0>.
        # X |0> = |1>
        # X |1> = |0>
        # X (|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = |+>
        # X (|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -|->
        # Fidelity is |<psi_target|psi_final>|^2, so phase doesn't matter.
        
        target_states = [
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 0, 0], dtype=complex),
            np.array([1, 1], dtype=complex) / np.sqrt(2),
            np.array([1, -1], dtype=complex) / np.sqrt(2), # X|-> = -|->, fidelity same
            np.array([1, -1j], dtype=complex) / np.sqrt(2), # X|+i> = |-i>
            np.array([1, 1j], dtype=complex) / np.sqrt(2)   # X|-i> = |+i>
        ]
        
        fidelities = []
        
        for psi0, psi_target in zip(states, target_states):
            # Density matrix rho = |psi><psi|
            rho = np.outer(psi0, psi0.conj())
            
            for step in range(N_steps):
                I_val = I[step]
                Q_val = Q[step]
                
                # Hamiltonian
                # H = detuning * n + 0.5 * anharmonicity * n * (n-1) + 0.5 * scale * (I * (a + a_dag) + i * Q * (a - a_dag))
                
                # n(n-1) for diag(0,1,2) is diag(0, 0, 2)
                n_nm1 = np.diag([0, 0, 2])
                
                H = (detuning * n + 
                     0.5 * anharmonicity * n_nm1 + 
                     0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag)))
                
                # Liouvillian step
                # d rho = -i [H, rho] * dt
                # + (L1 rho L1^dag - 0.5 {L1^dag L1, rho}) * dt
                # + (L2 rho L2^dag - 0.5 {L2^dag L2, rho}) * dt
                
                H_rho = H @ rho
                rho_H = rho @ H
                comm = H_rho - rho_H
                
                L1_rho_L1dag = L1 @ rho @ L1.conj().T
                L1dag_L1 = L1.conj().T @ L1
                anti_comm_1 = 0.5 * (L1dag_L1 @ rho + rho @ L1dag_L1)
                damp_1 = L1_rho_L1dag - anti_comm_1
                
                L2_rho_L2dag = L2 @ rho @ L2.conj().T
                L2dag_L2 = L2.conj().T @ L2
                anti_comm_2 = 0.5 * (L2dag_L2 @ rho + rho @ L2dag_L2)
                damp_2 = L2_rho_L2dag - anti_comm_2
                
                drho = (-1j * comm + damp_1 + damp_2) * dt
                rho = rho + drho
                
                # Renormalize trace to 1 to prevent numerical drift
                tr = np.trace(rho)
                if abs(tr) > 1e-12:
                    rho = rho / tr
                    
            # Fidelity
            # F = |<psi_target|psi_final>|^2
            # We have rho_final. We can compute <psi_target| rho_final |psi_target>
            # Or extract psi_final. Since we start with pure states and use Lindblads, it becomes mixed.
            # Fidelity for pure target and mixed state rho is F = <psi_target| rho |psi_target>
            
            fid = np.real(np.vdot(psi_target, rho @ psi_target))
            # Clamp to [0,1]
            fid = np.clip(fid, 0.0, 1.0)
            fidelities.append(fid)
            
        return np.mean(fidelities)

    def objective(params):
        I, Q = generate_pulse(params)
        
        infidelities = []
        for d in detunings:
            for s in scales:
                fid = simulate_single_case(I, Q, d, s)
                infid = 1.0 - fid
                infidelities.append(infid)
                
        mean_infid = np.mean(infidelities)
        max_infid = np.max(infidelities)
        
        return mean_infid + max_infid

    # Initial guess
    sigma_init = T_total / 6.0
    amp_init = np.sqrt(np.pi / 2.0) / sigma_init
    drag_init = 0.1
    
    x0 = [amp_init, drag_init, sigma_init]
    
    # Bounds
    # amp: 0 to 0.5 (bound is 0.4, but Q adds to it, so keep amp lower)
    # drag: -0.5 to 0.5
    # sigma: 2 to 10
    
    bounds = [(0.0, 0.5), (-0.5, 0.5), (2.0, 10.0)]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    optimal_params = result.x
    I, Q = generate_pulse(optimal_params)
    
    return np.column_stack((I, Q))

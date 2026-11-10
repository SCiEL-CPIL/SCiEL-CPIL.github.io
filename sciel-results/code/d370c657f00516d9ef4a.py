import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes.
    Strategy: Gradient-based optimization of pulse parameters to minimize
    worst-case infidelity across 9 hardware corners, starting from DRAG seed.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    # Constants
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters (nominal for operator construction, corners for scoring)
    # Note: The Hamiltonian structure is fixed, but detuning and scale vary in scoring.
    # The pulse I, Q is what we optimize.
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a # diag(0,1,2)
    
    # Anharmonicity
    chi = -2 * math.pi * 0.25
    
    # Initial DRAG Seed
    t = np.arange(n_steps) * dt
    # Gaussian pulse
    sigma = 40.0 / 6.0
    # Area pi means integral of amplitude dt = pi.
    # For a Gaussian A * exp(-(t-tc)^2 / (2*sigma^2)), integral is A * sigma * sqrt(2*pi).
    # We want area pi. So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma.
    # However, DRAG often has a derivative term. 
    # Let's stick to a standard DRAG shape: I = A_gauss, Q = -k * dI/dt
    # The seed in the prompt says: "DRAG-style Gaussian (sigma 40/6 ns, offset to 0 at edges, area pi)".
    # It implies the I component is the main carrier.
    
    tc = T / 2.0
    gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
    # Normalize to area pi
    # Discrete sum: sum(A_i * dt) = pi => A_i = pi / (dt * sum(gauss))
    area_sum = np.sum(gauss) * dt
    A = pi_val / area_sum if (pi_val := math.pi) else 1.0
    I_seed = A * gauss
    
    # DRAG term: Q is proportional to -dI/dt to cancel leakage.
    # dI/dt approx
    dI_dt = np.gradient(I_seed, dt)
    # Standard DRAG scaling k = -chi / (2 * pi) * ... wait, usually Q = - (chi / 2pi) * dI/dt / (2pi*freq)?
    # In the rotating frame defined by the task (0.5 * scale * (I(a+adag) + iQ(a-adag))),
    # the standard DRAG condition to suppress 0->2 leakage is Q = - (chi / 2 * pi) * (1/2) * dI/dt ?
    # Let's look at the Hamiltonian: H = ... + 0.5 * scale * (I (a+adag) + i Q (a-adag))
    # In rotating wave approximation, this drives 0->1 and 1->2.
    # The DRAG term is typically Q = - (chi / 2) * (1 / (2*pi*Omega)) * dI/dt ? No, simpler:
    # For a transmon, the DRAG pulse is Q(t) = - (chi / 2) * (1 / (2*pi * Omega_1)) * dI/dt ?
    # Actually, the standard result is Q(t) = - (chi / 2) * (1 / (2*pi * f_1)) * dI/dt is for specific conventions.
    # Let's use the common empirical DRAG: Q = - (chi / (2 * pi)) * dI/dt / (2 * pi) ?
    # Let's assume the seed program `quantum_seed` does this correctly. 
    # I will replicate a robust DRAG seed.
    # From literature: Q(t) = - (chi / 2) * (1 / (2*pi * Omega)) * dI/dt is often cited but depends on units.
    # Given H has 0.5 * scale * ..., the coupling is g = 0.5 * scale * I.
    # The DRAG amplitude is typically Q = - (chi / 2) * (1 / (2 * pi * Omega_1)) * dI/dt ?
    # Let's use the dimensionless form often found in Qiskit: Q = - (chi / 2) * (1 / (2*pi)) * dI/dt / (2*pi) ?
    # Actually, let's just use the seed's likely logic:
    # Q = - (chi / (2 * pi)) * dI_dt / (2 * pi) ? 
    # Let's try: Q = - (chi / 2) * dI_dt / (2 * pi)
    # Wait, the task says "seed is a DRAG-style Gaussian... area pi".
    # I will use a generic DRAG: Q = - (chi / 2) * dI_dt / (2 * pi)
    # Note: chi is in rad/ns? No, chi is -2*pi*0.25 GHz? 
    # Task says: anharmonicity -2 pi 0.25. This usually means -0.25 GHz in angular freq? 
    # "detuning bounds +-2 pi 0.002" -> 0.002 GHz?
    # If frequencies are in GHz, and time in ns, then 2*pi*f is rad/ns.
    # chi = -2*pi*0.25 rad/ns.
    # dI/dt is (rad/ns)/ns = rad/ns^2.
    # Q needs to be rad/ns.
    # Q = - (chi / 2) * (1 / (2*pi)) * dI_dt ? 
    # Let's check units. chi [rad/ns]. dI/dt [rad/ns^2]. 
    # If Q = k * dI/dt, k must be [ns].
    # Standard DRAG: Q = - (chi / 2) * (1 / Omega) * dI/dt ? 
    # Let's assume the seed works. I will start with I_seed and Q_seed = - (chi / 2) * dI_dt / (2 * pi)
    # Actually, a safer bet is to let the optimizer find the Q shape, initialized with a standard DRAG.
    
    Q_seed = - (chi / 2.0) * dI_dt / (2.0 * math.pi)
    
    # Constraint check for seed
    # Slew: ||p[i+1]-p[i]||/dt <= 0.12
    # Amp: sqrt(I^2+Q^2) <= 0.4
    
    def project_pulse(I, Q):
        """Project I, Q to satisfy amplitude and slew constraints."""
        I = I.copy()
        Q = Q.copy()
        
        # Amplitude constraint: sqrt(I^2+Q^2) <= 0.4
        # Scale down if violated
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > 0.4:
            scale = 0.4 / max_amp
            I *= scale
            Q *= scale
            amps *= scale
        
        # Slew constraint: ||p[i+1]-p[i]||/dt <= 0.12
        # This is a convex constraint on the sequence.
        # We can do a simple iterative projection or use a solver.
        # For speed, we do a simple forward-backward smoothing if violated.
        # Let's check violations.
        max_slew = 0.0
        for i in range(n_steps - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = math.sqrt(dI**2 + dQ**2)
            slew = dist / dt
            if slew > 0.12:
                max_slew = max(max_slew, slew)
        
        # Also check boundary: p[0] from 0, p[-1] to 0
        dI0 = I[0]
        dQ0 = Q[0]
        slew0 = math.sqrt(dI0**2 + dQ0**2) / dt
        if slew0 > 0.12:
             max_slew = max(max_slew, slew0)
        dIe = I[-1]
        dQe = Q[-1]
        slewe = math.sqrt(dIe**2 + dQe**2) / dt
        if slewe > 0.12:
             max_slew = max(max_slew, slewe)
        
        if max_slew > 0.12:
            # Simple heuristic: scale the whole pulse down? No, that hurts area.
            # Better: smooth the pulse.
            # We can apply a low-pass filter or simple averaging.
            # For now, if slew is violated, we scale down the pulse to meet slew.
            # This is a valid (though suboptimal) projection.
            scale = 0.12 / max_slew
            I *= scale
            Q *= scale
            # Re-check amplitude
            amps = np.sqrt(I**2 + Q**2)
            max_amp = np.max(amps)
            if max_amp > 0.4:
                scale2 = 0.4 / max_amp
                I *= scale2
                Q *= scale2
                
        return I, Q

    def simulate_pulse(I, Q, detuning, scale):
        """
        Simulate the pulse and return fidelity for 5 test states.
        Returns mean fidelity and max infidelity.
        """
        # Build Liouvillian
        # H = detuning * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I(a+adag) + iQ(a-adag))
        # Note: I and Q vary in time.
        
        # We need to evolve the density matrix.
        # rho(t+dt) = exp(L * dt) rho(t)
        # L = -i [H, .] + dissipators
        
        # To speed up, we pre-compute the static parts of H.
        # H_static = detuning * n + 0.5 * chi * n(n-1)
        # H_drive = 0.5 * scale * (I_t (a+adag) + i Q_t (a-adag))
        
        n_nm1 = n_op @ (n_op - np.eye(3))
        H_static = detuning * n_op + 0.5 * chi * n_nm1
        
        # Dissipators
        # a / sqrt(T1)
        L1 = a / math.sqrt(25000.0)
        # n sqrt(2/Tphi)
        L2 = n_op * math.sqrt(2.0 / 40000.0)
        
        # Pre-compute Lindblad superoperator for dissipators
        # D[L] rho = L rho Ldag - 0.5 (Ldag L rho + rho Ldag L)
        def lindblad_superop(L):
            LdagL = L.conj().T @ L
            # Superop matrix M such that vec(rho_new) = M vec(rho)
            # M = -i [H, .] + ...
            # For dissipator:
            # Term 1: L rho Ldag -> vec(L rho Ldag) = (Ldag^T ⊗ L) vec(rho)
            # Term 2: 0.5 Ldag L rho -> 0.5 (I ⊗ LdagL) vec(rho)
            # Term 3: 0.5 rho Ldag L -> 0.5 ((LdagL)^T ⊗ I) vec(rho)
            
            LdagT = L.conj().T.T # L^T
            M = np.kron(LdagT, L) - 0.5 * np.kron(np.eye(3), LdagL) - 0.5 * np.kron(LdagL.T, np.eye(3))
            return M
        
        M_diss = lindblad_superop(L1) + lindblad_superop(L2)
        
        # Hamiltonian Superop
        # -i [H, rho] -> -i (H rho - rho H)
        # vec(H rho) = (I ⊗ H) vec(rho)
        # vec(rho H) = (H^T ⊗ I) vec(rho)
        # M_H = -i (I ⊗ H - H^T ⊗ I)
        
        # Test states
        psi_0 = np.array([1, 0, 0], dtype=complex)
        psi_1 = np.array([0, 1, 0], dtype=complex)
        psi_p = (psi_0 + psi_1) / math.sqrt(2)
        psi_m = (psi_0 - psi_1) / math.sqrt(2)
        psi_ip = (psi_0 + 1j*psi_1) / math.sqrt(2)
        psi_im = (psi_0 - 1j*psi_1) / math.sqrt(2)
        
        test_states = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
        
        fidelities = []
        
        for psi in test_states:
            rho = np.outer(psi, psi.conj())
            rho_vec = rho.flatten()
            
            for t_idx in range(n_steps):
                I_t = I[t_idx]
                Q_t = Q[t_idx]
                
                # H_drive for this step
                H_drive = 0.5 * scale * (I_t * (a + a_dag) + 1j * Q_t * (a - a_dag))
                H_total = H_static + H_drive
                
                M_H = -1j * (np.kron(np.eye(3), H_total) - np.kron(H_total.T, np.eye(3)))
                M_total = M_H + M_diss
                
                # Exponentiate
                # For small dt, we can use expm
                M_exp = sla.expm(M_total * dt)
                rho_vec = M_exp @ rho_vec
                
            rho_final = rho_vec.reshape(3, 3)
            
            # Target state for X gate
            # X |0> = |1>, X |1> = |0>
            # X (|0> +/- |1>)/sqrt2 = (+/- |1> + |0>)/sqrt2
            # X (|0> +/- i |1>)/sqrt2 = (+/- i |1> + |0>)/sqrt2
            
            if np.allclose(psi, psi_0):
                psi_target = psi_1
            elif np.allclose(psi, psi_1):
                psi_target = psi_0
            elif np.allclose(psi, psi_p):
                psi_target = (psi_1 + psi_0) / math.sqrt(2)
            elif np.allclose(psi, psi_m):
                psi_target = (-psi_1 + psi_0) / math.sqrt(2)
            elif np.allclose(psi, psi_ip):
                psi_target = (1j*psi_1 + psi_0) / math.sqrt(2)
            else:
                psi_target = (-1j*psi_1 + psi_0) / math.sqrt(2)
            
            rho_target = np.outer(psi_target, psi_target.conj())
            fidelity = np.real(np.trace(rho_target.conj().T @ rho_final))
            # Clip to [0,1]
            fidelity = max(0.0, min(1.0, fidelity))
            fidelities.append(fidelity)
        
        mean_fid = np.mean(fidelities)
        max_infid = 1.0 - np.min(fidelities)
        return mean_fid, max_infid

    def objective(params):
        """
        Objective function: mean + max infidelity across 9 corners.
        """
        I = params[:n_steps]
        Q = params[n_steps:]
        
        I, Q = project_pulse(I, Q)
        
        total_score = 0.0
        max_infid_all = 0.0
        mean_fid_all = 0.0
        
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        count = 0
        for det in detunings:
            for sc in scales:
                mean_fid, max_infid = simulate_pulse(I, Q, det, sc)
                infid = 1.0 - mean_fid
                total_score += infid
                if max_infid > max_infid_all:
                    max_infid_all = max_infid
                count += 1
        
        mean_infid = total_score / count
        score = mean_infid + max_infid_all
        return score

    # Initial guess
    I_init, Q_init = project_pulse(I_seed, Q_seed)
    x0 = np.concatenate([I_init, Q_init])
    
    # Optimization
    # Use L-BFGS-B for bounded variables if we had bounds, but we project.
    # We can just use BFGS or L-BFGS-B with loose bounds and rely on projection.
    # However, projection is non-smooth. 
    # Alternative: Optimize in a parameter space that guarantees constraints?
    # E.g., I(t) = A(t) * cos(phase), Q(t) = A(t) * sin(phase) with A < 0.4 and smooth A.
    # For now, use L-BFGS-B with bounds [-0.4, 0.4] for I and Q, and hope projection handles slew.
    # Actually, slew constraint is hard. 
    # Let's use a simpler approach: Optimize the area and shape parameters of a DRAG pulse.
    # Variables: [A, k_drag, sigma, tc]
    # This reduces dimensionality from 160 to 4, making optimization much faster and more robust.
    
    def parametric_pulse(params):
        A, k_drag, sigma, tc = params
        t = np.arange(n_steps) * dt
        gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
        # Normalize area to pi? No, A is amplitude. 
        # We want area ~ pi. 
        # Let's fix the area by scaling A.
        # Sum(gauss * dt) = area_gauss.
        # A = pi / area_gauss
        area_gauss = np.sum(gauss) * dt
        if area_gauss == 0:
            return np.zeros(n_steps), np.zeros(n_steps)
        I = (math.pi / area_gauss) * gauss
        dI_dt = np.gradient(I, dt)
        Q = -k_drag * dI_dt
        return I, Q

    def objective_param(params):
        I, Q = parametric_pulse(params)
        I, Q = project_pulse(I, Q)
        
        total_score = 0.0
        max_infid_all = 0.0
        
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        count = 0
        for det in detunings:
            for sc in scales:
                mean_fid, max_infid = simulate_pulse(I, Q, det, sc)
                infid = 1.0 - mean_fid
                total_score += infid
                if max_infid > max_infid_all:
                    max_infid_all = max_infid
                count += 1
        
        mean_infid = total_score / count
        score = mean_infid + max_infid_all
        return score

    # Initial params for parametric pulse
    # A is fixed by area.
    # k_drag: standard is chi / (2*pi) ? 
    # Let's guess k_drag = 0.1
    # sigma = 40/6
    # tc = 20
    x0_param = np.array([1.0, 0.1, 40.0/6.0, 20.0])
    
    # Bounds
    # k_drag can be negative
    bounds = [(0.0, 1.0), (-0.5, 0.5), (5.0, 10.0), (10.0, 30.0)]
    
    best_score = float('inf')
    best_params = x0_param
    best_I = I_seed
    best_Q = Q_seed
    
    # Evaluate initial
    s = objective_param(x0_param)
    if s < best_score:
        best_score = s
        best_params = x0_param
        I_init, Q_init = parametric_pulse(x0_param)
        best_I, best_Q = project_pulse(I_init, Q_init)
        
    # Optimize
    try:
        result = opt.minimize(objective_param, x0_param, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'maxfun': 500})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            I_opt, Q_opt = parametric_pulse(best_params)
            best_I, best_Q = project_pulse(I_opt, Q_opt)
    except Exception as e:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        I_final, Q_final = project_pulse(best_I, best_Q)
        return np.column_stack((I_final, Q_final))
    
    # Refine with full pulse optimization if time permits?
    # Given the 300s limit and 9 simulations per eval, 200 iters * 9 sims * expm(9x9) is likely too slow for full pulse.
    # Parametric is safer.
    
    I_final, Q_final = project_pulse(best_I, best_Q)
    return np.column_stack((I_final, Q_final))

# Helper for pi
import math
pi_val = math.pi

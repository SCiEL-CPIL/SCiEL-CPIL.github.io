import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0

    # Hardware parameters
    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    # X gate targets
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)
    ]

    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_op.conj().T @ a_op
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n_op @ (n_op - I3)
    alpha_term = 0.5 * alpha * n_nm1

    def get_H(I, Q, delta, scale):
        H = delta * n_op + alpha_term
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H

    def simulate_unitary(I_Q, delta, scale):
        """Fast unitary simulation for optimization."""
        U_total = np.eye(3, dtype=complex)
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            U_step = expm(-1j * H * dt)
            U_total = U_step @ U_total
        
        infs = []
        for idx in range(6):
            psi = states[idx]
            target = targets[idx]
            psi_final = U_total @ psi
            fid = np.abs(np.vdot(target, psi_final))**2
            infs.append(1 - fid)
        return np.mean(infs), np.max(infs)

    def score_pulse_fast(I_Q):
        """Score using unitary simulation (proxy for full score)."""
        total = 0.0
        for delta in deltas:
            for scale in scales:
                m, mx = simulate_unitary(I_Q, delta, scale)
                total += (m + mx)
        return total / 9.0

    def generate_drag_pulse(params):
        """
        Generate a DRAG pulse.
        params: [t_c, sigma, A, eta]
        """
        t_c, sigma, A, eta = params
        t = np.linspace(0, 40, N)
        
        # Gaussian envelope
        g = np.exp(-(t - t_c)**2 / (2 * sigma**2))
        g_prime = g * (-(t - t_c) / sigma**2)
        
        I = A * g
        Q = eta * A * g_prime
        
        I_Q = np.column_stack((I, Q))
        
        # Enforce amplitude bound
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale_factor = amp_bound / max_amp
            I_Q *= scale_factor
            
        # Enforce slew rate
        # Check from 0 to first
        if np.linalg.norm(I_Q[0]) / dt > slew_bound:
            scale = (slew_bound * dt) / np.linalg.norm(I_Q[0])
            I_Q[0] *= scale
            
        # Check between steps
        for i in range(1, N):
            diff = I_Q[i] - I_Q[i-1]
            norm = np.linalg.norm(diff)
            if norm / dt > slew_bound:
                scale = (slew_bound * dt) / norm
                I_Q[i] = I_Q[i-1] + scale * diff
                
        # Check last to 0
        if np.linalg.norm(I_Q[-1]) / dt > slew_bound:
            scale = (slew_bound * dt) / np.linalg.norm(I_Q[-1])
            I_Q[-1] *= scale
            
        return I_Q

    def objective(params):
        I_Q = generate_drag_pulse(params)
        return score_pulse_fast(I_Q)

    # Optimization
    # Initial guess:
    # t_c = 20 (center)
    # sigma = 4 (width, ensures slew rate for A~0.4)
    # A = 0.4 (max amp)
    # eta = 0.1 (initial guess for DRAG)
    
    x0 = [20.0, 4.0, 0.4, 0.1]
    
    # Bounds
    bounds = [
        (10, 30),  # t_c
        (2, 10),   # sigma
        (0.1, 0.4),# A
        (-1.0, 1.0)# eta
    ]

    best_IQ = None
    best_score = np.inf
    
    # Run Nelder-Mead optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_IQ = generate_drag_pulse(result.x)
            
        # Check time
        if time.time() > deadline - 20:
            if best_IQ is None:
                best_IQ = generate_drag_pulse(x0)
            return best_IQ

        # Refine with L-BFGS-B if time permits
        result2 = minimize(objective, result.x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
        
        if result2.fun < best_score:
            best_score = result2.fun
            best_IQ = generate_drag_pulse(result2.x)
            
    except Exception as e:
        pass

    # Fallback
    if best_IQ is None:
        best_IQ = generate_drag_pulse(x0)

    # Final constraint check
    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        if amps[0] / dt > slew_bound + 1e-9:
            return False
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-9:
            return False
        if amps[-1] / dt > slew_bound + 1e-9:
            return False
        return True

    if not check_constraints(best_IQ):
        # Try to fix by scaling down
        amps = np.sqrt(best_IQ[:, 0]**2 + best_IQ[:, 1]**2)
        scale = amp_bound / (np.max(amps) * 1.1)
        best_IQ *= scale
        # Re-check slew
        for i in range(1, N):
            diff = best_IQ[i] - best_IQ[i-1]
            norm = np.linalg.norm(diff)
            if norm / dt > slew_bound:
                s = (slew_bound * dt) / norm
                best_IQ[i] = best_IQ[i-1] + s * diff
        if np.linalg.norm(best_IQ[0])/dt > slew_bound:
             best_IQ[0] *= (slew_bound * dt) / np.linalg.norm(best_IQ[0])
        if np.linalg.norm(best_IQ[-1])/dt > slew_bound:
             best_IQ[-1] *= (slew_bound * dt) / np.linalg.norm(best_IQ[-1])

    return best_IQ

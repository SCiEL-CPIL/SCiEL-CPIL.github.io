"""
Strategy: Exact state-vector simulation with analytical gradients and multi-start L-BFGS-B.
Instead of derivative-free Nelder-Mead, we use L-BFGS-B which leverages the Hessian
approximation to converge much faster per iteration. We compute exact analytical
gradients using the chain rule on the state vector evolution. This speedup allows
for a significantly larger number of diverse random restarts (multi-start), which is
crucial for escaping local minima in the non-convex QAOA landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys
from scipy.linalg import expm

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build the problem Hamiltonian C (MaxCut) and mixing Hamiltonian B (sum X).
    C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    B = sum_{i=0}^{n-1} X_i
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    # Helper to apply operator on qubit k
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
        
    # Build C
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    # Build B
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _expected_cut_value_and_grad(n, edges, angles, C, B):
    """
    Compute the expected cut value and its analytical gradient.
    angles: (p, 2) array of (gamma, beta)
    
    Returns:
        value: float
        grad: (p, 2) array
    """
    if len(edges) == 0:
        return 0.0, np.zeros_like(angles)
        
    dim = 2**n
    p = len(angles)
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Store states at each step for backward pass (or just recompute forward for simplicity in memory, 
    # but storing is faster for gradients. For n=10, dim=1024, storing 2*p+1 states is fine)
    states = [psi]
    
    # Forward pass
    for i in range(p):
        gamma, beta = angles[i]
        
        # U_C = exp(-i gamma C)
        U_C = expm(-1j * gamma * C)
        psi = U_C @ psi
        states.append(psi)
        
        # U_B = exp(-i beta B)
        U_B = expm(-1j * beta * B)
        psi = U_B @ psi
        states.append(psi)
        
    # Final state
    psi_final = states[-1]
    
    # Value: <psi| C |psi>
    val = np.vdot(psi_final, C @ psi_final).real
    
    # Gradient computation
    # We want d/dgamma <psi| C |psi> and d/dbeta <psi| C |psi>
    # Using the chain rule / adjoint method.
    # Let |psi_k> be the state after layer k.
    # |psi_{k+1}> = U_B(beta_k) U_C(gamma_k) |psi_k>
    # 
    # Gradient with respect to parameters in layer k:
    # d/dgamma_k <psi_final| C |psi_final> = 2 * Re( <chi_k| dU_C/dgamma_k U_C ... |psi_0> ) ... this is complex.
    # 
    # Simpler approach for small systems: 
    # d/dgamma <psi|H|psi> = 2*Re(<psi| i [H, C] |psi>) if H and C commute? No.
    # General formula: d/dtheta <psi|O|psi> = 2*Re( <psi| (dU/dtheta U^dag) O |psi> )
    # where dU/dtheta U^dag is the generator.
    # For U = exp(-i theta H), dU/dtheta U^dag = -i H.
    # So d/dgamma <psi|C|psi> = 2*Re( <psi| (-i C) C |psi> )? No, that's for the same operator.
    #
    # Let's use the standard variational derivative:
    # d/dtheta <psi|O|psi> = 2*Re( <psi| (d|psi>/dtheta) O |psi> ) + 2*Re( <d<psi|/dtheta| O |psi> )
    # Since <d<psi|/dtheta| = (d|psi>/dtheta)^dag
    #
    # For layer k, let |psi_k> be the state BEFORE layer k.
    # |psi_{k+1}> = U_k |psi_k>
    # d|psi_{k+1}>/dgamma_k = (dU_k/dgamma_k) |psi_k> = (-i C) U_k |psi_k> = -i C |psi_{k+1}>
    # d|psi_{k+1}>/dbeta_k = (dU_k/dbeta_k) |psi_k> = (-i B) U_k |psi_k> = -i B |psi_{k+1}>
    #
    # However, the gradient of the final expectation value with respect to parameters in layer k
    # depends on the "adjoint" state.
    # Let |xi_k> be the adjoint state.
    # |xi_{p}> = C |psi_p>
    # |xi_k> = U_k^dag |xi_{k+1}>
    #
    # Then d/dgamma_k <psi|C|psi> = 2*Re( <xi_{k+1}| (-i C) |psi_k> ) ? 
    # Actually, the standard formula is:
    # d/dtheta_k <psi_p|O|psi_p> = 2*Re( <xi_k| (dU_k/dtheta_k) |psi_k> )
    # where |xi_k> is propagated backward from |xi_p> = O|psi_p> using U_j^dag.
    
    grad = np.zeros((p, 2), dtype=float)
    
    # Backward pass for adjoint states
    # xi_p = C @ psi_p
    xi = C @ psi_final
    
    for k in range(p - 1, -1, -1):
        gamma_k, beta_k = angles[k]
        psi_k = states[k]
        psi_k1 = states[k+1] # After C
        psi_k2 = states[k+2] # After B
        
        # Gradient with respect to beta_k (applied on psi_k1 to get psi_k2)
        # d/d beta_k = 2*Re( <xi_k2| (-i B) |psi_k1> )
        # Note: The adjoint at step k+1 (before B) is needed? 
        # Let's trace carefully.
        # L = <psi_p| C |psi_p>
        # psi_p = U_B(p-1) ... U_C(0) psi_0
        # 
        # Let |phi_k> be the state after k layers.
        # dL/dtheta_k = 2 Re( <chi_k| (dU_k/dtheta_k) |phi_{k-1}> )
        # where |chi_k> is the adjoint state at step k, defined by:
        # |chi_p> = C |phi_p>
        # |chi_k> = U_k^dag |chi_{k+1}>
        
        # We have states[0]...states[2p]
        # states[k] is phi_{k-1} (before layer k, 0-indexed layer k)
        # states[k+1] is after C_k
        # states[k+2] is after B_k (which is phi_k)
        
        # We need |chi> at the point where the parameter acts.
        # For beta_k: acts on phi_{k-1} (states[k]) to produce phi_k (states[k+2])? No.
        # Layer k consists of C then B.
        # U_k = U_B(beta_k) U_C(gamma_k)
        # phi_k = U_k phi_{k-1}
        
        # The adjoint |chi_k> is the state "after" layer k in the backward pass?
        # Let's define |chi_k> such that it corresponds to the state right after applying layer k in the forward pass.
        # Then |chi_{k-1}> = U_k^dag |chi_k>
        
        # Let's re-index clearly.
        # Forward:
        # psi_0
        # psi_1 = U_C(gamma_0) psi_0
        # psi_2 = U_B(beta_0) psi_1
        # psi_3 = U_C(gamma_1) psi_2
        # ...
        # psi_{2p}
        
        # Backward:
        # chi_{2p} = C psi_{2p}
        # chi_{2p-1} = U_B(beta_{p-1})^dag chi_{2p}
        # chi_{2p-2} = U_C(gamma_{p-1})^dag chi_{2p-1}
        # ...
        
        # Gradient w.r.t beta_k (which is U_B(beta_k) acting on psi_{2k+1} to get psi_{2k+2}):
        # dL/dbeta_k = 2 Re( <chi_{2k+1}| (dU_B/dbeta_k) |psi_{2k+1}> )
        # dU_B/dbeta_k = -i B
        # So we need chi at index 2k+1 (before B) and psi at index 2k+1.
        
        # Gradient w.r.t gamma_k (which is U_C(gamma_k) acting on psi_{2k} to get psi_{2k+1}):
        # dL/dgamma_k = 2 Re( <chi_{2k}| (dU_C/dgamma_k) |psi_{2k}> )
        # dU_C/dgamma_k = -i C
        # So we need chi at index 2k (before C) and psi at index 2k.

        # We currently have `xi` which is chi_{2p} initially.
        # We need to propagate it back.
        
        # In the loop k from p-1 down to 0:
        # We currently hold chi_{2k+2} in `xi`?
        # Let's check initialization.
        # xi = chi_{2p}
        
        # Iteration k = p-1:
        # We need chi_{2(p-1)+1} = chi_{2p-1} for beta.
        # chi_{2p-1} = U_B(beta_{p-1})^dag chi_{2p}
        # We have chi_{2p} in `xi`.
        # U_B = exp(-i beta B). U_B^dag = exp(i beta B).
        
        U_B_dag = expm(1j * beta_k * B)
        chi_before_B = U_B_dag @ xi  # This is chi_{2k+1}
        
        # Compute grad for beta_k
        # psi_{2k+1} is states[2k+1]
        psi_before_B = states[2*k + 1]
        
        # dU_B/dbeta = -i B
        # Term: <chi_before_B| (-i B) |psi_before_B>
        term_beta = np.vdot(chi_before_B, (-1j * B) @ psi_before_B)
        grad[k, 1] = 2.0 * term_beta.real
        
        # Now propagate back through C
        # We need chi_{2k} for gamma.
        # chi_{2k} = U_C(gamma_k)^dag chi_{2k+1}
        # chi_{2k+1} is `chi_before_B`
        
        U_C_dag = expm(1j * gamma_k * C)
        chi_before_C = U_C_dag @ chi_before_B # This is chi_{2k}
        
        # Compute grad for gamma_k
        # psi_{2k} is states[2k]
        psi_before_C = states[2*k]
        
        # dU_C/dgamma = -i C
        # Term: <chi_before_C| (-i C) |psi_before_C>
        term_gamma = np.vdot(chi_before_C, (-1j * C) @ psi_before_C)
        grad[k, 0] = 2.0 * term_gamma.real
        
        # Update xi for next iteration (k-1)
        # Next iteration needs chi_{2(k-1)+2} = chi_{2k}
        # So set xi = chi_before_C
        xi = chi_before_C

    return val, grad

def _graph_features(n, edges):
    """Extract simple graph features for initialization."""
    degrees = np.zeros(n)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    avg_deg = np.mean(degrees) if n > 0 else 0.0
    max_deg = np.max(degrees) if n > 0 else 0.0
    min_deg = np.min(degrees) if n > 0 else 0.0
    density = len(edges) / (n * (n-1) / 2) if n > 1 else 0.0
    
    return avg_deg, max_deg, min_deg, density

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Get graph features
    avg_deg, max_deg, min_deg, density = _graph_features(n, edges)
    
    # Generate diverse initializations
    candidates = []
    
    # Candidate 1: Standard fixed
    candidates.append(np.tile([0.35, 0.25], (p, 1)))
    
    # Candidate 2: Based on density and degree
    if density > 0:
        g1 = 0.5 / max(1.0, max_deg)
        b1 = 0.3 / max(1.0, max_deg)
        candidates.append(np.tile([g1, b1], (p, 1)))
    
    # Candidate 3: Smaller angles for sparse graphs
    if density < 0.4:
        candidates.append(np.tile([0.2, 0.15], (p, 1)))
    
    # Candidate 4: Larger angles for dense graphs
    if density > 0.4:
        candidates.append(np.tile([0.6, 0.4], (p, 1)))
        
    # Add many random starts (L-BFGS is fast, so we can afford more)
    np.random.seed(123)
    for _ in range(20):
        gams = np.random.uniform(0.05, 1.2, p)
        bets = np.random.uniform(0.05, 0.6, p)
        candidates.append(np.column_stack((gams, bets)))
        
    best_val = -1.0
    best_angles = candidates[0]
    
    # Define objective and gradient function for L-BFGS-B
    def objective_and_grad(x):
        if _time_remaining() < 5.0:
            return 1e9, np.zeros_like(x)
        x = np.array(x).reshape(-1, 2)
        val, grad = _expected_cut_value_and_grad(n, edges, x, C, B)
        return -val, -grad.flatten()
        
    def objective(x):
        val, _ = _expected_cut_value_and_grad(n, edges, x.reshape(-1, 2), C, B)
        return -val

    # Run L-BFGS-B from best candidate and several others
    # We will run optimization on a subset of the best initial candidates to save time
    # First, evaluate all candidates to find the best start
    for cand in candidates:
        val, _ = _expected_cut_value_and_grad(n, edges, cand, C, B)
        if val > best_val:
            best_val = val
            best_angles = cand
        if _time_remaining() < 20.0:
            break
            
    x0 = best_angles.flatten()
    
    # Optimize from the best initial point
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                       jac=True,
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < -best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
    except Exception:
        pass
    
    # If time allows, try a few more restarts from other diverse points
    # Sort remaining candidates by initial value (descending) and try top ones
    # We already have best_val. Let's try 5 other random starts if time permits.
    if _time_remaining() > 30.0:
        np.random.seed(456)
        for _ in range(5):
            if _time_remaining() < 20.0:
                break
            gams = np.random.uniform(0.1, 0.8, p)
            bets = np.random.uniform(0.1, 0.4, p)
            x0 = np.column_stack((gams, bets)).flatten()
            try:
                res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                               jac=True,
                               options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < -best_val:
                    best_angles = res.x.reshape(p, 2)
                    best_val = -res.fun
            except Exception:
                pass
                
    return best_angles

# EVOLVE-BLOCK-END

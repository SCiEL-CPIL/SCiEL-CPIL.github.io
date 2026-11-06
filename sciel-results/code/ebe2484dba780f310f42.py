"""
Strategy: Multi-start L-BFGS-B optimization with analytical gradients.
Instead of a coarse grid scan + Nelder-Mead, we use a dense set of diverse starting points
and optimize each with L-BFGS-B, which provides fast convergence using gradient information.
The gradient is computed analytically using the chain rule on the state-vector evolution,
leveraging precomputed eigendecompositions of the Hamiltonians C and B for O(n^2) cost per step
(instead of O(n^3) for matrix exponentials).
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build Hamiltonians C and B and return their eigendecompositions.
    C = sum_{(u,v) in E} 0.5 * (I - Z_u Z_v)
    B = sum_{i} X_i
    """
    if len(edges) == 0:
        dim = 2**n
        C = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        wC, VC = np.linalg.eigh(C)
        wB, VB = np.linalg.eigh(B)
        return wC, VC, wB, VB

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result

    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    wC, VC = np.linalg.eigh(C)
    wB, VB = np.linalg.eigh(B)
    
    return wC, VC, wB, VB

def _expected_cut_and_grad(n, edges, gamma1, beta1, gamma2, beta2, wC, VC, wB, VB):
    """
    Compute expected cut value and its gradient with respect to [gamma1, beta1, gamma2, beta2].
    
    Circuit: |psi> = U_B(beta2) U_C(gamma2) U_B(beta1) U_C(gamma1) |+>^n
    U_C(g) = exp(-i g C) = V_C exp(-i g wC) V_C^H
    U_B(b) = exp(-i b B) = V_B exp(-i b wB) V_B^H
    
    Value: <psi| C |psi>
    Gradient: d/dtheta <psi| C |psi> = 2 Re( <dpsi/dtheta| C |psi> )
    
    d/dgamma_k |psi> = -i C |psi_before_k>
    d/dbeta_k |psi> = -i B |psi_before_k>
    
    Where |psi_before_k> is the state just before applying the k-th gate.
    """
    dim = 2**n
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute phase factors
    phase_C1 = np.exp(-1j * gamma1 * wC)
    phase_B1 = np.exp(-1j * beta1 * wB)
    phase_C2 = np.exp(-1j * gamma2 * wC)
    phase_B2 = np.exp(-1j * beta2 * wB)
    
    # Forward pass
    # 1. Apply U_C(gamma1)
    vC_psi0 = VC.conj().T @ psi0
    vC_psi0 *= phase_C1
    psi1 = VC @ vC_psi0
    
    # 2. Apply U_B(beta1)
    vB_psi1 = VB.conj().T @ psi1
    vB_psi1 *= phase_B1
    psi2 = VB @ vB_psi1
    
    # 3. Apply U_C(gamma2)
    vC_psi2 = VC.conj().T @ psi2
    vC_psi2 *= phase_C2
    psi3 = VC @ vC_psi2
    
    # 4. Apply U_B(beta2)
    vB_psi3 = VB.conj().T @ psi3
    vB_psi3 *= phase_B2
    psi4 = VB @ vB_psi3
    
    # Value: <psi4| C |psi4>
    vC_psi4 = VC.conj().T @ psi4
    val = np.sum(np.abs(vC_psi4)**2 * wC).real
    
    # Gradients
    # d/dgamma1: -i C |psi0>
    # <dpsi/dg1| C |psi4> = (-i <psi0| C) C |psi4> = -i <psi0| C^2 |psi4>
    # Wait, C is Hermitian.
    # dpsi1/dg1 = -i C psi0
    # grad_g1 = 2 Re( (-i <psi0| C) C psi4 ) = 2 Re( -i <psi0| C^2 psi4> )
    # Actually, let's use the adjoint method or just direct differentiation.
    # d/dg1 <psi|C|psi> = <dpsi/dg1|C|psi> + <psi|C|dpsi/dg1>
    # dpsi/dg1 = -i C psi0 (propagated to the end? No, psi1 depends on g1)
    # psi1 = U_C(g1) psi0
    # psi2 = U_B(beta1) psi1
    # psi3 = U_C(g2) psi2
    # psi4 = U_B(beta2) psi3
    # dpsi4/dg1 = U_B(beta2) U_C(g2) U_B(beta1) (dpsi1/dg1)
    # dpsi1/dg1 = -i C psi0
    
    # Let's compute the intermediate states for the gradient terms.
    # term_g1 = <psi4| C |dpsi4/dg1> + c.c.
    # dpsi4/dg1 = U_B2 U_C2 U_B1 (-i C psi0)
    
    # Let's compute the vector v = U_B1 (-i C psi0)
    # C psi0:
    # vC_psi0 is already computed above (before phase). Let's recompute or store.
    # We stored vC_psi0 * phase_C1 = vC_psi1_in_B_basis? No.
    # vC_psi0 = VC^H psi0.
    # C psi0 = VC (wC * vC_psi0)
    
    # This is getting complicated to do efficiently without storing all intermediates.
    # Given n<=10, dim<=1024, storing a few vectors is cheap.
    
    # Let's re-do the forward pass storing the states in the eigenbasis where needed.
    
    # 1. psi0
    # 2. psi1 = U_C(g1) psi0
    # 3. psi2 = U_B(beta1) psi1
    # 4. psi3 = U_C(g2) psi2
    # 5. psi4 = U_B(beta2) psi3
    
    # We have psi1, psi2, psi3, psi4 from the forward pass above.
    
    # Gradient w.r.t gamma1:
    # dpsi1/dg1 = -i C psi0
    # dpsi2/dg1 = U_B(beta1) dpsi1/dg1
    # dpsi3/dg1 = U_C(gamma2) dpsi2/dg1
    # dpsi4/dg1 = U_B(beta2) dpsi3/dg1
    # grad_g1 = 2 Re( <psi4| C |dpsi4/dg1> )
    
    # dpsi1/dg1 = -i C psi0
    C_psi0 = VC @ (wC * (VC.conj().T @ psi0))
    dpsi1_dg1 = -1j * C_psi0
    
    # Propagate dpsi1_dg1 through U_B(beta1)
    vB_dpsi1 = VB.conj().T @ dpsi1_dg1
    vB_dpsi1 *= phase_B1
    dpsi2_dg1 = VB @ vB_dpsi1
    
    # Propagate dpsi2_dg1 through U_C(gamma2)
    vC_dpsi2 = VC.conj().T @ dpsi2_dg1
    vC_dpsi2 *= phase_C2
    dpsi3_dg1 = VC @ vC_dpsi2
    
    # Propagate dpsi3_dg1 through U_B(beta2)
    vB_dpsi3 = VB.conj().T @ dpsi3_dg1
    vB_dpsi3 *= phase_B2
    dpsi4_dg1 = VB @ vB_dpsi3
    
    grad_g1 = 2 * np.real(np.vdot(psi4, C @ dpsi4_dg1))
    
    # Gradient w.r.t beta1:
    # dpsi2/db1 = -i B psi1
    B_psi1 = VB @ (wB * (VB.conj().T @ psi1))
    dpsi2_db1 = -1j * B_psi1
    
    # Propagate through U_C(gamma2)
    vC_dpsi2 = VC.conj().T @ dpsi2_db1
    vC_dpsi2 *= phase_C2
    dpsi3_db1 = VC @ vC_dpsi2
    
    # Propagate through U_B(beta2)
    vB_dpsi3 = VB.conj().T @ dpsi3_db1
    vB_dpsi3 *= phase_B2
    dpsi4_db1 = VB @ vB_dpsi3
    
    grad_b1 = 2 * np.real(np.vdot(psi4, C @ dpsi4_db1))
    
    # Gradient w.r.t gamma2:
    # dpsi3/dg2 = -i C psi2
    C_psi2 = VC @ (wC * (VC.conj().T @ psi2))
    dpsi3_dg2 = -1j * C_psi2
    
    # Propagate through U_B(beta2)
    vB_dpsi3 = VB.conj().T @ dpsi3_dg2
    vB_dpsi3 *= phase_B2
    dpsi4_dg2 = VB @ vB_dpsi3
    
    grad_g2 = 2 * np.real(np.vdot(psi4, C @ dpsi4_dg2))
    
    # Gradient w.r.t beta2:
    # dpsi4/db2 = -i B psi3
    B_psi3 = VB @ (wB * (VB.conj().T @ psi3))
    dpsi4_db2 = -1j * B_psi3
    
    grad_b2 = 2 * np.real(np.vdot(psi4, C @ dpsi4_db2))
    
    grad = np.array([grad_g1, grad_b1, grad_g2, grad_b2])
    
    return val, grad

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    if len(edges) == 0:
        return np.zeros((p, 2))
    
    try:
        wC, VC, wB, VB = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    def objective(x):
        val, grad = _expected_cut_and_grad(n, edges, x[0], x[1], x[2], x[3], wC, VC, wB, VB)
        return -val, -grad
    
    best_val = -1e10
    best_angles = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Define bounds
    bounds = [(0, np.pi), (0, np.pi/2), (0, np.pi), (0, np.pi/2)]
    
    # Multi-start strategy
    # 1. Known good points
    starts = [
        [0.35, 0.25, 0.35, 0.25],
        [0.25, 0.15, 0.25, 0.15],
        [0.5, 0.3, 0.5, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.8, 0.4, 0.8, 0.4],
    ]
    
    # 2. Random starts
    np.random.seed(42)
    num_random_starts = 10
    for _ in range(num_random_starts):
        g1 = np.random.uniform(0, np.pi)
        b1 = np.random.uniform(0, np.pi/2)
        g2 = np.random.uniform(0, np.pi)
        b2 = np.random.uniform(0, np.pi/2)
        starts.append([g1, b1, g2, b2])
    
    # 3. Grid-based starts (coarse)
    g_vals = np.linspace(0.1, np.pi-0.1, 4)
    b_vals = np.linspace(0.1, np.pi/2-0.1, 3)
    for g1 in g_vals:
        for b1 in b_vals:
            for g2 in g_vals:
                for b2 in b_vals:
                    starts.append([g1, b1, g2, b2])
    
    # Sort starts by heuristic? No, just run them.
    # To be safe with time, we will break if time runs out.
    
    for start in starts:
        if _time_remaining() < 5.0:
            break
            
        start = np.array(start)
        
        try:
            res = minimize(objective, start, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_angles = res.x
        except Exception:
            continue
    
    # Final polish with higher precision if time allows
    if _time_remaining() > 10.0:
        try:
            res = minimize(objective, best_angles, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_angles = res.x
        except Exception:
            pass
    
    result = best_angles.reshape(p, 2)
    result[:, 0] = np.clip(result[:, 0], 0, np.pi)
    result[:, 1] = np.clip(result[:, 1], 0, np.pi/2)
    
    return result

# EVOLVE-BLOCK-END

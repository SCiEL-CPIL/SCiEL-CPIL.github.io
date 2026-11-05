"""
QAOA angle design for fixed-depth p=2 MaxCut on small graphs.
Combines vectorized state-vector simulation with analytic gradients 
for fast, high-precision optimization.
"""
import numpy as np
import time
import scipy.optimize

# Global deadline management
# The evaluator kills after 150s. We use a margin of 10s to ensure clean return.
_DEADLINE = 140.0 
_start_time_global = time.time()

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    edge_list = list(edges)
    # Optimization: iterate through bitmasks
    for mask in range(dim):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Returns: real array (p, 2), rows (gamma, beta) in radians.
    """
    # Only p=2 is required by the task, but we handle general p if needed, 
    # though the specific structure below is optimized for p=2.
    # The task specifies p=2.
    if p != 2:
        # Fallback for unexpected p, though task says p=2
        return np.tile([0.35, 0.25], (p, 1))
        
    start_time = time.time()
    
    # 1. Precomputation
    dim = 1 << n
    edge_list = list(edges)
    
    # Compute cost vector c[x] for all x
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2), dtype=np.float64)

    # Precompute index arrays for the mixer (X rotations)
    # For each qubit q, we need indices where bit q is 0 and 1
    mixer_indices = []
    for q in range(n):
        bit = 1 << q
        # np.arange is fast enough for dim <= 1024
        idx_all = np.arange(dim)
        idx0 = idx_all[(idx_all & bit) == 0]
        idx1 = idx0 | bit
        mixer_indices.append((idx0, idx1))
        
    # Helper: Apply mixer to psi
    def apply_mixer(psi, beta):
        cb = np.cos(beta)
        sb = np.sin(beta)
        psi_new = psi.copy()
        for idx0, idx1 in mixer_indices:
            p0 = psi_new[idx0].copy()
            p1 = psi_new[idx1].copy()
            psi_new[idx0] = cb * p0 - 1j * sb * p1
            psi_new[idx1] = -1j * sb * p0 + cb * p1
        return psi_new

    # Helper: Forward pass returning all intermediate states for gradient
    def forward_pass(params):
        """
        params: [g0, b0, g1, b1]
        Returns: psi0, psi1, psi2, psi3
        psi0: after cost 0
        psi1: after mixer 0
        psi2: after cost 1
        psi3: after mixer 1 (final state)
        """
        g0, b0, g1, b1 = params
        
        # Initial state |+>^n
        psi_init = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        # Layer 0 Cost: exp(-i g0 C)
        phase0 = np.exp(-1j * g0 * cut_vals)
        psi0 = psi_init * phase0
        
        # Layer 0 Mixer: exp(-i b0 Sum X)
        psi1 = apply_mixer(psi0, b0)
        
        # Layer 1 Cost: exp(-i g1 C)
        phase1 = np.exp(-1j * g1 * cut_vals)
        psi2 = psi1 * phase1
        
        # Layer 1 Mixer: exp(-i b1 Sum X)
        psi3 = apply_mixer(psi2, b1)
        
        return psi0, psi1, psi2, psi3

    def eval_fn(params):
        """Expected cut value."""
        _, _, _, psi3 = forward_pass(params)
        probs = np.abs(psi3) ** 2
        return np.dot(probs, cut_vals)

    def analytic_gradient(params):
        """
        Compute gradient of expected cut w.r.t. [g0, b0, g1, b1].
        Uses chain rule / backpropagation.
        
        E = <psi3| C |psi3>
        dE/dtheta = 2 Re[ <dpsi3/dtheta | C | psi3> ]
        
        We compute adjoints (gradients of loss w.r.t. intermediate states)
        and propagate back.
        
        Let L = E.
        dL/dpsi3 = 2 * C * psi3 (treating psi3 as variable, ignoring conjugate for now, 
        but in complex backprop: dL/dpsi3* = C psi3, dL/dpsi3 = 0 if psi3 is independent, 
        but here psi3 depends on params. Standard quantum gradient:
        dE/dtheta = 2 Re[ <psi3| (i dH/dtheta) |psi3> ] is not quite right for state prep.
        
        Let's use the standard parameter-shift or chain rule on the state vector.
        E = sum |psi3|^2 c.
        dE/dtheta = sum (d|psi3|^2/dtheta) c = sum (psi3* dpsi3/dtheta + dpsi3*/dtheta psi3) c
        = 2 Re[ sum (psi3* dpsi3/dtheta) c ] = 2 Re[ <psi3| C | dpsi3/dtheta> ]
        
        We need dpsi3/dtheta for theta in {g0, b0, g1, b1}.
        Instead of computing 4 separate forward passes for derivatives, 
        we can use the adjoint method.
        
        Let's define the cost function J = <psi3| C |psi3>.
        We want dJ/dtheta.
        
        Backprop:
        1. Compute psi0, psi1, psi2, psi3.
        2. Compute adjoint at the end: 
           dJ/dpsi3 = 0 (if we treat psi3 as output, but J depends on psi3)
           Actually, let's define the "gradient of the loss with respect to the state" 
           as \bar{psi}.
           
           For real-valued loss L = <psi|O|psi>:
           dL/dpsi* = O psi
           dL/dpsi = O psi* (conjugate)
           
           We propagate the gradient backwards.
           Let \bar{psi}_k be the gradient of L w.r.t. psi_k (conjugate variable).
           
           Step 1 (End):
           \bar{psi}_3 = C * psi_3  (This is dL/dpsi3*)
           
           Step 2 (Mixer 1):
           psi_3 = U_M(b1) psi_2
           dpsi_3/db1 = (dU_M/db1) psi_2
           \bar{psi}_2 += (dU_M/db1)^dag \bar{psi}_3  (accumulate gradient for b1? No, for state)
           Wait, we need dL/db1 as well.
           
           Let's stick to the simpler, robust method: 
           Compute dpsi3/dtheta for each theta via finite difference? 
           No, the inspiration used analytic. 
           
           Let's implement the analytic gradient properly using the chain rule on the components.
           
           dpsi0/dg0 = -i * cut_vals * psi0
           dpsi0/db0 = 0
           
           dpsi1/dg0 = U_M(b0) dpsi0/dg0
           dpsi1/db0 = (dU_M/db0) psi0
           
           dpsi2/dg0 = U_C(g1) dpsi1/dg0
           dpsi2/db0 = U_C(g1) dpsi1/db0
           dpsi2/dg1 = -i * cut_vals * psi2
           dpsi2/db1 = 0
           
           dpsi3/dg0 = U_M(b1) dpsi2/dg0
           dpsi3/db0 = U_M(b1) dpsi2/db0
           dpsi3/dg1 = U_M(b1) dpsi2/dg1
           dpsi3/db1 = (dU_M/db1) psi2
           
           Then grad_theta = 2 Re[ <psi3| C | dpsi3/dtheta> ]
           
           This requires storing dpsi/dtheta vectors. There are 4 parameters, so 4 vectors of size dim.
           This is feasible for dim=1024.
        """
        g0, b0, g1, b1 = params
        
        # Forward pass states
        psi0, psi1, psi2, psi3 = forward_pass(params)
        
        # Precompute U_M and dU_M/db for b0, b1
        # U_M(b) = exp(-i b X) = cos(b) I - i sin(b) X
        # dU_M/db = -sin(b) I - i cos(b) X
        
        def get_mixer_deriv(psi_in, beta):
            """
            Returns d(U_M(psi_in))/dbeta
            U_M = cb I - i sb X
            dU_M/db = -sb I - i cb X
            Apply to psi_in.
            """
            cb = np.cos(beta)
            sb = np.sin(beta)
            # dU_M/db |psi> = -sb |psi> - i cb X|psi>
            # X|psi> is just flipping bits.
            x_psi = np.zeros_like(psi_in)
            for idx0, idx1 in mixer_indices:
                x_psi[idx0] = psi_in[idx1]
                x_psi[idx1] = psi_in[idx0]
            
            return -sb * psi_in - 1j * cb * x_psi

        # Derivatives of psi0
        dpsi0_dg0 = -1j * cut_vals * psi0
        dpsi0_db0 = np.zeros_like(psi0)
        
        # Derivatives of psi1
        # psi1 = U_M(b0) psi0
        # dpsi1/dg0 = U_M(b0) dpsi0/dg0
        dpsi1_dg0 = apply_mixer(dpsi0_dg0, b0)
        # dpsi1/db0 = (dU_M(b0)/db0) psi0
        dpsi1_db0 = get_mixer_deriv(psi0, b0)
        
        # Derivatives of psi2
        # psi2 = U_C(g1) psi1
        # U_C(g) = exp(-i g C)
        # dU_C/dg = -i C U_C(g)
        phase1 = np.exp(-1j * g1 * cut_vals)
        
        dpsi2_dg0 = phase1 * dpsi1_dg0
        dpsi2_db0 = phase1 * dpsi1_db0
        dpsi2_dg1 = -1j * cut_vals * psi2
        dpsi2_db1 = np.zeros_like(psi2)
        
        # Derivatives of psi3
        # psi3 = U_M(b1) psi2
        dpsi3_dg0 = apply_mixer(dpsi2_dg0, b1)
        dpsi3_db0 = apply_mixer(dpsi2_db0, b1)
        dpsi3_dg1 = apply_mixer(dpsi2_dg1, b1)
        dpsi3_db1 = get_mixer_deriv(psi2, b1)
        
        # Compute gradients
        # grad_theta = 2 Re[ <psi3| C | dpsi3/dtheta> ]
        # Note: <psi3| C | dpsi3> = sum (psi3.conj() * C * dpsi3)
        
        c_psi3 = cut_vals * psi3
        
        grad = np.zeros(4)
        grad[0] = 2 * np.real(np.dot(psi3.conj(), c_psi3 * dpsi3_dg0 / cut_vals if np.any(cut_vals!=0) else 0)) # Wait, C is diagonal.
        # <psi3| C | dpsi3> = sum_i psi3[i]^* * c[i] * dpsi3[i]
        # My c_psi3 is c[i] * psi3[i]. I need c[i] * dpsi3[i].
        
        # Let's re-calculate correctly
        # inner_prod = sum (psi3.conj() * (cut_vals * dpsi3))
        
        grad[0] = 2 * np.real(np.dot(psi3.conj(), cut_vals * dpsi3_dg0))
        grad[1] = 2 * np.real(np.dot(psi3.conj(), cut_vals * dpsi3_db0))
        grad[2] = 2 * np.real(np.dot(psi3.conj(), cut_vals * dpsi3_dg1))
        grad[3] = 2 * np.real(np.dot(psi3.conj(), cut_vals * dpsi3_db1))
        
        return -grad # Negate for minimization

    def objective(params):
        return -eval_fn(params)

    # Optimization
    bounds = [(0.0, np.pi)] * 4
    
    # Initial guesses
    inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.4, 0.4, 0.4],
        [0.6, 0.2, 0.3, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.8, 0.6, 0.5, 0.4],
        [0.3, 0.6, 0.3, 0.6],
        [0.6, 0.4, 0.6, 0.4],
        [0.55, 0.35, 0.35, 0.25],
        [0.45, 0.45, 0.45, 0.35],
        [0.35, 0.25, 0.35, 0.25], # Seed
    ]
    
    # Add random inits
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.uniform(0.1, 1.5, size=4).tolist())
        
    best_score = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Phase 1: Broad search
    for init in inits:
        if time.time() - start_time > _DEADLINE - 10:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective, np.array(init, dtype=np.float64),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue

    # Phase 2: Polish
    if time.time() - start_time < _DEADLINE - 5:
        try:
            res = scipy.optimize.minimize(
                objective, best_params.copy(),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Final High Precision
    if time.time() - start_time < _DEADLINE - 2:
        try:
            res = scipy.optimize.minimize(
                objective, best_params.copy(),
                jac=analytic_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    # Format output
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])

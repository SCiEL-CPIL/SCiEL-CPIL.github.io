import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Builds A and B for the 6-site XXZ Heisenberg chain.
    Sites 0..5.
    A contains odd bonds: (0,1), (2,3), (4,5)
    B contains even bonds: (1,2), (3,4)
    Field is split: 0.5 * h * sum(Sz) in both A and B.
    """
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    # Spin operators are 0.5 * Pauli
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(I2)
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res
    
    def bond_term(site1, site2):
        # Sx1 Sx2 + Sy1 Sy2 + lambda Sz1 Sz2
        def two_site_op(op1, s1, op2, s2):
            ops = []
            for i in range(N):
                if i == s1:
                    ops.append(op1)
                elif i == s2:
                    ops.append(op2)
                else:
                    ops.append(I2)
            res = ops[0]
            for i in range(1, N):
                res = np.kron(res, ops[i])
            return res

        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    # Field
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def coefficients(stages):
    start_time = time.time()
    # Hard deadline: 140s to be safe within 150s limit
    deadline = start_time + 140.0
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute eigendecompositions
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Precompute exact propagators
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals_H)
        U_t = eigvecs_H @ np.diag(phase) @ eigvecs_H.conj().T
        U_exact.append(U_t)
    
    # Store references for speed
    VA = eigvecs_A
    VB = eigvecs_B
    eA = eigvals_A
    eB = eigvals_B
    dim = 64
    I = np.eye(dim, dtype=complex)
    
    def fast_exp_A(t_coeff):
        if t_coeff == 0:
            return I.copy()
        phase = np.exp(-1j * t_coeff * eA)
        return VA @ np.diag(phase) @ VA.conj().T
        
    def fast_exp_B(t_coeff):
        if t_coeff == 0:
            return I.copy()
        phase = np.exp(-1j * t_coeff * eB)
        return VB @ np.diag(phase) @ VB.conj().T

    def compute_error(a, b):
        """Compute mean spectral norm error for given a, b coefficient arrays."""
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = I.copy()
            for k in range(12):
                E_A_k = fast_exp_A(a[k] * t)
                E_B_k = fast_exp_B(b[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
                
            U_target = U_exact[i]
            diff = U_approx - U_target
            err = np.linalg.norm(diff, 2)
            err_sum += err
        return err_sum / len(times)

    def check_valid(a, b):
        if len(a) != 12 or len(b) != 12:
            return False
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return False
        if abs(np.sum(a) - 1.0) > 1e-10:
            return False
        if abs(np.sum(b) - 1.0) > 1e-10:
            return False
        return True

    # Strategy: Use high-order symmetric product formulas as starting points,
    # then refine with Nelder-Mead in the full 22D space (11 free a's, 11 free b's).
    
    # Known 4th-order symmetric coefficients for 4-stage splitting:
    # a1 = 1/(2*(2 - 2^(1/3))), a2 = 1/2 - a1
    c4 = 1.0 / (2.0 * (2.0 - 2.0**(1.0/3.0)))
    c4b = 0.5 - c4
    
    # Initialization 1: Uniform
    a_init1 = np.ones(12) / 12.0
    b_init1 = np.ones(12) / 12.0
    
    # Initialization 2: 4th-order 4-stage coefficients repeated 3 times
    a_init2 = np.array([c4/3, c4b/3, c4b/3, c4/3, c4/3, c4b/3, c4b/3, c4/3, c4/3, c4b/3, c4b/3, c4/3])
    b_init2 = a_init2.copy()
    
    # Initialization 3: Palindromic with emphasis on middle
    a_init3 = np.array([0.06, 0.09, 0.12, 0.16, 0.16, 0.12, 0.12, 0.16, 0.16, 0.12, 0.09, 0.06])
    a_init3 = a_init3 / np.sum(a_init3)
    b_init3 = a_init3.copy()
    
    # Initialization 4: Edge-weighted symmetric
    a_init4 = np.array([0.18, 0.12, 0.08, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.08, 0.12, 0.18])
    a_init4 = a_init4 / np.sum(a_init4)
    b_init4 = a_init4.copy()
    
    # Initialization 5: Strang-like with refinements
    a_init5 = np.array([0.10, 0.12, 0.14, 0.14, 0.14, 0.12, 0.12, 0.14, 0.14, 0.14, 0.12, 0.10])
    a_init5 = a_init5 / np.sum(a_init5)
    b_init5 = a_init5.copy()
    
    # Evaluate all initializations
    candidates = [(a_init1, b_init1), (a_init2, b_init2), (a_init3, b_init3), (a_init4, b_init4), (a_init5, b_init5)]
    
    best_a = None
    best_b = None
    best_val = float('inf')
    
    for a_cand, b_cand in candidates:
        if time.time() > deadline - 60:
            break
        val = compute_error(a_cand, b_cand)
        if val < best_val:
            best_val = val
            best_a = a_cand.copy()
            best_b = b_cand.copy()
    
    # Now refine with Nelder-Mead in the full space
    # Convert to 22-param representation: a[0..10], b[0..10], with a[11] and b[11] determined
    
    def objective(params):
        a_free = params[:11]
        b_free = params[11:22]
        
        a = np.array(a_free)
        a = np.append(a, 1.0 - np.sum(a_free))
        
        b = np.array(b_free)
        b = np.append(b, 1.0 - np.sum(b_free))
        
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        return compute_error(a, b)
    
    # Initialize from best candidate
    x0 = np.concatenate([best_a[:11], best_b[:11]])
    
    # Nelder-Mead implementation
    def _nelder_mead(f, x0, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-8, time_limit=None, start_time=None):
        if start_time is None:
            start_time = time.time()
        n = len(x0)
        
        # Initialize simplex with reasonable step sizes
        simplex = [x0.copy()]
        for i in range(n):
            x_new = x0.copy()
            step = 0.02  # Smaller initial step for better exploration
            if abs(x_new[i]) < 1e-6:
                x_new[i] += step
            else:
                x_new[i] += step * np.sign(x_new[i])
            simplex.append(x_new)
            
        values = [f(x) for x in simplex]
        
        for it in range(max_iter):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
                
            # Sort simplex
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Check convergence
            if abs(values[0] - values[-1]) < tol * max(1.0, abs(values[0])):
                break
                
            x_best = simplex[0]
            x_worst = simplex[-1]
            x_second_worst = simplex[-2]
            
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            f_r = f(x_r)
            
            if f_r < values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    values[-1] = f_r
            elif f_r < values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                values[-1] = f_r
            else:
                # Contraction
                x_c = x_centroid + rho * (x_worst - x_centroid)
                f_c = f(x_c)
                if f_c < values[-1]:
                    simplex[-1] = x_c
                    values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        values[i] = f(simplex[i])
                        
        order = np.argsort(values)
        return simplex[order[0]], values[order[0]]
    
    # Run Nelder-Mead with time budget
    remaining_time = deadline - time.time() - 10.0  # 10s safety margin
    
    if remaining_time > 15.0:
        t_start_opt = time.time()
        x_best_opt, val_best_opt = _nelder_mead(
            objective, x0, 
            max_iter=3000,
            time_limit=remaining_time,
            start_time=t_start_opt
        )
        
        # Convert back
        a_opt = np.array(x_best_opt[:11])
        a_opt = np.append(a_opt, 1.0 - np.sum(x_best_opt[:11]))
        b_opt = np.array(x_best_opt[11:22])
        b_opt = np.append(b_opt, 1.0 - np.sum(x_best_opt[11:22]))
        
        if check_valid(a_opt, b_opt) and val_best_opt < best_val:
            best_a = a_opt
            best_b = b_opt
            best_val = val_best_opt
    
    # Final safety check
    if not check_valid(best_a, best_b):
        # Fallback to uniform
        best_a = np.ones(12) / 12.0
        best_b = np.ones(12) / 12.0
    
    return [float(x) for x in best_a], [float(x) for x in best_b]

if __name__ == "__main__":
    # Self test
    t0 = time.time()
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
    print(f"Time taken: {time.time() - t0:.2f}s")

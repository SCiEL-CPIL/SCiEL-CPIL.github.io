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

def _nelder_mead(f, x0, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-10, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    # Initial step size: 0.05 for zeros, 5% for non-zeros
    for i in range(n):
        x_new = x0.copy()
        if x_new[i] == 0:
            x_new[i] += 0.05
        else:
            x_new[i] += 0.05 * x_new[i]
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
        if abs(values[0] - values[-1]) < tol:
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

def coefficients(stages):
    start_time = time.time()
    # Hard deadline: 145s to be safe within 150s limit
    deadline = start_time + 145.0
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute eigendecompositions
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals_H)
        U_t = eigvecs_H @ np.diag(phase) @ eigvecs_H.conj().T
        U_exact.append(U_t)
        
    def get_exp_A(t_coeff):
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_A)
        return eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
        
    def get_exp_B(t_coeff):
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_B)
        return eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T

    def objective(params):
        # params: 22 values. 11 free a's, 11 free b's.
        # a_12 = 1 - sum(a_1..a_11)
        # b_12 = 1 - sum(b_1..b_11)
        
        a_free = params[:11]
        b_free = params[11:22]
        
        a = np.array(a_free)
        a = np.append(a, 1.0 - np.sum(a_free))
        
        b = np.array(b_free)
        b = np.append(b, 1.0 - np.sum(b_free))
        
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # Order: exp(-i b_k t B) exp(-i a_k t A)
                E_A_k = get_exp_A(a[k] * t)
                E_B_k = get_exp_B(b[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
                
            U_target = U_exact[i]
            diff = U_approx - U_target
            err = np.linalg.norm(diff, 2)
            err_sum += err
            
        return err_sum / len(times)
        
    # Strategy: 
    # 1. Try a symmetric initialization (palindromic a and b)
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1] -> 6 free params for a
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1] -> 6 free params for b
    # Total 12 params.
    # This reduces the search space significantly and aligns with typical product formula symmetries.
    
    def objective_symmetric(params):
        # params: 12 values. 6 for a, 6 for b.
        x_a = params[:6]
        x_b = params[6:12]
        
        a = np.array([x_a[0], x_a[1], x_a[2], x_a[3], x_a[4], x_a[5], 
                      x_a[5], x_a[4], x_a[3], x_a[2], x_a[1], x_a[0]])
        b = np.array([x_b[0], x_b[1], x_b[2], x_b[3], x_b[4], x_b[5], 
                      x_b[5], x_b[4], x_b[3], x_b[2], x_b[1], x_b[0]])
        
        # Adjust to satisfy sum=1
        sum_a = np.sum(a)
        if sum_a != 0:
            a = a * (1.0 / sum_a)
        else:
            a = np.ones(12) / 12.0
            
        sum_b = np.sum(b)
        if sum_b != 0:
            b = b * (1.0 / sum_b)
        else:
            b = np.ones(12) / 12.0
            
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                E_A_k = get_exp_A(a[k] * t)
                E_B_k = get_exp_B(b[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
                
            U_target = U_exact[i]
            diff = U_approx - U_target
            err = np.linalg.norm(diff, 2)
            err_sum += err
            
        return err_sum / len(times)
        
    # Initial guess: uniform distribution (1/12 each)
    # In symmetric params, uniform means all x are 1/12.
    # But our mapping scales to sum 1. If all x=1, sum=12, scaled to 1/12.
    x0_sym = np.ones(12) # Will be scaled to 1/12
    
    # Run Nelder-Mead on symmetric space
    # Time budget: 60s for symmetric optimization
    t_start_sym = time.time()
    time_limit_sym = 60.0
    
    x_best_sym, val_best_sym = _nelder_mead(
        objective_symmetric, x0_sym, 
        max_iter=2000, 
        time_limit=time_limit_sym, 
        start_time=t_start_sym
    )
    
    # Convert best symmetric params to full coefficients
    x_a = x_best_sym[:6]
    x_b = x_best_sym[6:12]
    a_full = np.array([x_a[0], x_a[1], x_a[2], x_a[3], x_a[4], x_a[5], 
                       x_a[5], x_a[4], x_a[3], x_a[2], x_a[1], x_a[0]])
    b_full = np.array([x_b[0], x_b[1], x_b[2], x_b[3], x_b[4], x_b[5], 
                       x_b[5], x_b[4], x_b[3], x_b[2], x_b[1], x_b[0]])
    
    # Normalize
    sum_a = np.sum(a_full)
    if sum_a != 0:
        a_full = a_full * (1.0 / sum_a)
    else:
        a_full = np.ones(12) / 12.0
        
    sum_b = np.sum(b_full)
    if sum_b != 0:
        b_full = b_full * (1.0 / sum_b)
    else:
        b_full = np.ones(12) / 12.0
        
    best_a = a_full
    best_b = b_full
    best_val = val_best_sym
    
    # Check if we have time for asymmetric refinement
    if time.time() < deadline - 10.0:
        # Initialize asymmetric search from the symmetric solution
        # Convert full a, b to 22-param vector
        # a[11] is dependent, b[11] is dependent
        a_init = best_a[:11]
        b_init = best_b[:11]
        x0_asym = np.concatenate([a_init, b_init])
        
        t_start_asym = time.time()
        time_limit_asym = deadline - t_start_asym - 5.0 # 5s safety margin
        
        if time_limit_asym > 10.0:
            x_best_asym, val_best_asym = _nelder_mead(
                objective, x0_asym, 
                max_iter=1000, 
                time_limit=time_limit_asym, 
                start_time=t_start_asym
            )
            
            # Convert back
            a_asym = np.array(x_best_asym[:11])
            a_asym = np.append(a_asym, 1.0 - np.sum(x_best_asym[:11]))
            b_asym = np.array(x_best_asym[11:22])
            b_asym = np.append(b_asym, 1.0 - np.sum(x_best_asym[11:22]))
            
            # Check bounds
            if np.any(np.abs(a_asym) <= 50) and np.any(np.abs(b_asym) <= 50):
                if val_best_asym < best_val:
                    best_a = a_asym
                    best_b = b_asym
                    best_val = val_best_asym
    
    # Final safety check
    if np.any(np.abs(best_a) > 50) or np.any(np.abs(best_b) > 50):
        # Fallback to uniform
        best_a = np.ones(12) / 12.0
        best_b = np.ones(12) / 12.0
        
    return [float(x) for x in best_a], [float(x) for x in best_b]

if __name__ == "__main__":
    # Self test
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
    print(f"Time taken: {time.time() - time.time()}") # Just to ensure it runs

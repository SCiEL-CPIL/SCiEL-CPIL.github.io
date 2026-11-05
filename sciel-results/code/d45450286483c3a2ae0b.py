import numpy as np
import time

def _build_hamiltonians():
    """
    Builds A and B for the 6-site XXZ Heisenberg chain.
    Sites 0..5.
    Bonds:
    0: (0,1) -> Odd -> A
    1: (1,2) -> Even -> B
    2: (2,3) -> Odd -> A
    3: (3,4) -> Even -> B
    4: (4,5) -> Odd -> A
    """
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Spin operators (0.5 * Pauli)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond_term(site1, site2):
        def two_site_op(op1, s1, op2, s2):
            ops = []
            for i in range(N):
                if i == s1:
                    ops.append(op1)
                elif i == s2:
                    ops.append(op2)
                else:
                    ops.append(np.eye(2, dtype=complex))
            res = ops[0]
            for i in range(1, N):
                res = np.kron(res, ops[i])
            return res

        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    A, B = _build_hamiltonians()
    H = A + B
    dim = 64
    
    # Precompute exact propagators
    eigvals, eigvecs = np.linalg.eigh(H)
    
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals)
        U_t = eigvecs @ np.diag(phase) @ eigvecs.conj().T
        U_exact.append(U_t)
        
    # Precompute eigendecomps of A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    def get_exp_A(t_coeff):
        if abs(t_coeff) < 1e-12:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_A)
        return eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
        
    def get_exp_B(t_coeff):
        if abs(t_coeff) < 1e-12:
            return np.eye(dim, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_B)
        return eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T

    def compute_error(a, b):
        """Compute the mean spectral norm error. a and b are length-12 arrays."""
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(dim, dtype=complex)
            for k in range(12):
                E_A_k = get_exp_A(a[k] * t)
                E_B_k = get_exp_B(b[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
            diff = U_approx - U_exact[i]
            err = np.linalg.norm(diff, 2)
            err_sum += err
        return err_sum / len(times)

    def objective(params):
        """
        params: 24-element vector = [a_0..a_11, b_0..b_11]
        We enforce sum(a)=1 and sum(b)=1 via penalty.
        """
        a = params[:12]
        b = params[12:24]
        
        # Penalty for sum constraints
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 100.0 * (sum_a - 1.0)**2 + 100.0 * (sum_b - 1.0)**2
        
        # Penalty for bounds
        bound_penalty = 0.0
        for x in a:
            if x < -50:
                bound_penalty += 100.0 * (-50 - x)**2
            elif x > 50:
                bound_penalty += 100.0 * (x - 50)**2
        for x in b:
            if x < -50:
                bound_penalty += 100.0 * (-50 - x)**2
            elif x > 50:
                bound_penalty += 100.0 * (x - 50)**2
        
        total_penalty = penalty + bound_penalty
        if total_penalty > 10.0:
            return 1e10 + total_penalty
            
        return compute_error(a, b) + total_penalty

    # Differential Evolution
    def differential_evolution(objective, bounds, pop_size=40, max_iter=200, F=0.8, CR=0.9, time_limit=None, start_time=None):
        """
        Simple differential evolution.
        bounds: list of (low, high) tuples, length 24.
        """
        n = len(bounds)
        low = np.array([b[0] for b in bounds])
        high = np.array([b[1] for b in bounds])
        
        # Initialize population
        np.random.seed(42)
        pop = np.random.uniform(low, high, (pop_size, n))
        # Ensure initial population satisfies sum constraints approximately
        for i in range(pop_size):
            a_part = pop[i, :12]
            b_part = pop[i, 12:24]
            # Scale to satisfy sum=1
            if np.sum(a_part) != 0:
                pop[i, :12] = a_part / np.sum(a_part)
            else:
                pop[i, :12] = 1.0/12.0
            if np.sum(b_part) != 0:
                pop[i, 12:24] = b_part / np.sum(b_part)
            else:
                pop[i, 12:24] = 1.0/12.0
            # Clip to bounds
            pop[i] = np.clip(pop[i], low, high)
        
        fits = [objective(x) for x in pop]
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_fit = fits[best_idx]
        
        for it in range(max_iter):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            
            # Sort by fitness
            order = np.argsort(fits)
            pop = pop[order]
            fits = [fits[i] for i in order]
            
            new_pop = []
            new_fits = []
            
            for i in range(pop_size):
                # Select three distinct random indices different from i
                r1, r2, r3 = i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.randint(pop_size)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.empty(n)
                j_rand = np.random.randint(n)
                for j in range(n):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, low, high)
                
                # Evaluate
                f_trial = objective(trial)
                
                if f_trial <= fits[i]:
                    new_pop.append(trial)
                    new_fits.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_fits.append(fits[i])
            
            pop = np.array(new_pop)
            fits = new_fits
            
            # Update best
            best_idx = np.argmin(fits)
            if fits[best_idx] < best_fit:
                best_fit = fits[best_idx]
                best_x = pop[best_idx].copy()
            
            # Check convergence
            if abs(fits[0] - fits[-1]) < 1e-10:
                break
        
        return best_x, best_fit

    # Define bounds for the 24 parameters
    bounds = [(-50, 50)] * 24
    
    # Run differential evolution
    x_best, val_best = differential_evolution(
        objective, bounds, 
        pop_size=30, max_iter=100, 
        F=0.7, CR=0.85,
        time_limit=deadline - 5.0, start_time=start_time
    )
    
    # Extract a and b
    a = x_best[:12]
    b = x_best[12:24]
    
    # Enforce sum constraints exactly
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-12:
        a = a / sum_a
    else:
        a = np.full(12, 1.0/12.0)
    if abs(sum_b) > 1e-12:
        b = b / sum_b
    else:
        b = np.full(12, 1.0/12.0)
    
    # Clip to bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-enforce sum after clipping
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-12:
        a = a / sum_a
    if abs(sum_b) > 1e-12:
        b = b / sum_b
    
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    return a, b

if __name__ == "__main__":
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")

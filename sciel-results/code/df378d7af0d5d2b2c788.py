import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def _compute_error(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times):
    """Compute mean spectral norm error."""
    dim = A_eigvecs.shape[0]
    errors = []
    
    for t in times:
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            if abs(b[k]) > 1e-15:
                theta_b = b[k] * t
                exp_b = np.exp(-1j * theta_b * B_eigvals)
                U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
            
            if abs(a[k]) > 1e-15:
                theta_a = a[k] * t
                exp_a = np.exp(-1j * theta_a * A_eigvals)
                U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
        
        U_exact = U_exact_cache[t]
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)

def _suzuki_4th_order_coeffs(n_sub):
    """Generate 4th-order Suzuki-Trotter coefficients for n_sub substeps.
    Each substep uses the 4th-order scheme:
    A_{c0/2} B_{c1/2} A_{c0/2} A_{c1/2} B_{c0} A_{c1/2} A_{c0/2} B_{c1/2} A_{c0/2}
    where c0 = 1/(2 - 2^(1/3)), c1 = -2^(1/3)/(2 - 2^(1/3))
    
    This gives 9 factors per substep. For 12 stages (24 factors), we need
    a different approach. Let's use a symmetric 12-stage scheme directly.
    """
    # For a 12-stage product formula with pattern B A B A ... (12 A's and 12 B's),
    # we can use a symmetric 6th-order-like scheme or optimized coefficients.
    # Let's start with a symmetric pattern inspired by high-order splittings.
    
    # Actually, let's use the 4th-order Suzuki coefficients adapted to 12 stages.
    # The 4th-order Suzuki-Trotter has 3 A-factors and 2 B-factors per "cell" (5 factors).
    # For 12 stages, we have 12 A and 12 B factors = 24 factors total.
    # We can think of this as 4 "cells" of 6 factors each (3 A + 3 B alternating).
    
    # Let's use a different approach: define 12 coefficients for A and 12 for B
    # using a symmetric high-order splitting.
    
    # For simplicity, start with a symmetric 4th-order-like distribution.
    # The key insight: for 4th order, we need the coefficients to satisfy certain conditions.
    # Let's use the known 4th-order Suzuki-Trotter with c0, c1 and distribute over 12 stages.
    
    c0 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    c1 = -2.0 ** (1.0/3.0) / (2.0 - 2.0 ** (1.0/3.0))
    
    # Each Suzuki 4th-order cell: A_{c0/2} B_{c1/2} A_{c0/2} A_{c1/2} B_{c0} A_{c1/2} A_{c0/2} B_{c1/2} A_{c0/2}
    # That's 5 A-factors and 4 B-factors per cell (9 factors total).
    # For 12 stages, let's use a symmetric pattern with 12 A and 12 B.
    
    # Alternative: use a 6th-order symmetric scheme.
    # 6th order Suzuki-Trotter coefficients (per substep):
    # This is complex. Let's use a simpler approach: symmetric distribution.
    
    # For 12 stages, use a symmetric pattern:
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1] (symmetric)
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1] (symmetric)
    # with sum(a) = 1, sum(b) = 1
    
    # Initialize with uniform (2nd order Strang-like)
    a = [1.0/12.0] * 12
    b = [1.0/12.0] * 12
    
    return a, b

def _genetic_algorithm(f, pop_size=20, generations=50, mutation_rate=0.3, elite_size=2, timeout=None, start_time=None):
    """Genetic algorithm for optimization."""
    n = len(f.__code__.co_varnames) if hasattr(f, '__code__') else 22
    # Assume 22 parameters (2*(stages-1) for stages=12)
    n = 22
    
    # Initialize population
    population = []
    fitness = []
    
    # Start with Suzuki-inspired initial points
    base_a = [0.15, -0.05, 0.15, 0.15, -0.05, 0.15, 0.15, -0.05, 0.15, -0.05, 0.15, 0.15]
    base_b = [0.15, -0.05, 0.15, 0.15, -0.05, 0.15, 0.15, -0.05, 0.15, -0.05, 0.15, 0.15]
    
    for i in range(pop_size):
        if i == 0:
            x = [0.0] * n
        elif i == 1:
            x = [base_a[j] - 1.0/12.0 for j in range(11)] + [base_b[j] - 1.0/12.0 for j in range(11)]
        else:
            x = np.random.uniform(-0.2, 0.2, size=n)
        population.append(x)
        fitness.append(f(x))
    
    best_x = population[np.argmin(fitness)]
    best_f = min(fitness)
    
    for gen in range(generations):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                break
        
        # Sort by fitness
        indices = np.argsort(fitness)
        population = [population[i] for i in indices]
        fitness = [fitness[i] for i in indices]
        
        # Update best
        if fitness[0] < best_f:
            best_f = fitness[0]
            best_x = population[0]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(min(elite_size, pop_size)):
            new_population.append(population[i][:])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                candidates = np.random.choice(len(population), size=3, replace=False)
                best_cand = candidates[np.argmin([fitness[c] for c in candidates])]
                return population[best_cand]
            
            p1 = select()
            p2 = select()
            
            # Uniform crossover
            child = []
            for i in range(n):
                if np.random.random() < 0.5:
                    child.append(p1[i])
                else:
                    child.append(p2[i])
            
            # Mutation
            for i in range(n):
                if np.random.random() < mutation_rate:
                    child[i] += np.random.normal(0, 0.05)
            
            new_population.append(child)
        
        population = new_population
        fitness = [f(x) for x in population]
    
    return best_x, best_f

def _normalize_and_clip(a, b):
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    return a, b

def coefficients(stages):
    start_time = time.time()
    deadline = 130.0
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    
    U_exact_cache = {}
    for t in times:
        exp_h = np.exp(-1j * t * H_eigvals)
        U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T
    
    # Initial guess: symmetric 4th-order Suzuki-Trotter adapted to 12 stages
    a0, b0 = _suzuki_4th_order_coeffs(stages)
    
    best_a = a0[:]
    best_b = b0[:]
    best_err = _compute_error(best_a, best_b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return _normalize_and_clip(best_a, best_b)
    
    # Parameterize: a_i = 1/stages + x_i for i=0..stages-2, a_{stages-1} = 1 - sum(a_0..a_{stages-2})
    # Similarly for b.
    
    def _error_from_x(x):
        a = [0.0] * stages
        b = [0.0] * stages
        for i in range(stages - 1):
            a[i] = 1.0 / stages + x[i]
        a[stages - 1] = 1.0 - sum(a[:stages - 1])
        
        offset = stages - 1
        for i in range(stages - 1):
            b[i] = 1.0 / stages + x[offset + i]
        b[stages - 1] = 1.0 - sum(b[:stages - 1])
        
        return _compute_error(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
    
    n_params = 2 * (stages - 1)
    
    # Use genetic algorithm
    np.random.seed(42)
    x_best, f_best = _genetic_algorithm(_error_from_x, pop_size=20, generations=30, mutation_rate=0.25, elite_size=2, timeout=deadline, start_time=start_time)
    
    if f_best < best_err:
        a_opt = [0.0] * stages
        b_opt = [0.0] * stages
        for i in range(stages - 1):
            a_opt[i] = 1.0 / stages + x_best[i]
        a_opt[stages - 1] = 1.0 - sum(a_opt[:stages - 1])
        
        offset = stages - 1
        for i in range(stages - 1):
            b_opt[i] = 1.0 / stages + x_best[offset + i]
        b_opt[stages - 1] = 1.0 - sum(b_opt[:stages - 1])
        
        best_a = a_opt[:]
        best_b = b_opt[:]
        best_err = f_best
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return _normalize_and_clip(best_a, best_b)
    
    # Fine-tune with local search around best solution
    for _ in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Local perturbation
        x_start = x_best[:] + np.random.uniform(-0.05, 0.05, size=n_params)
        
        # Simple local search: try small steps in each direction
        best_local = x_start[:]
        best_local_f = f_best
        
        for i in range(n_params):
            for delta in [0.01, -0.01, 0.02, -0.02]:
                x_try = x_start[:]
                x_try[i] += delta
                f_try = _error_from_x(x_try)
                if f_try < best_local_f:
                    best_local_f = f_try
                    best_local = x_try
        
        if best_local_f < best_err:
            a_opt = [0.0] * stages
            b_opt = [0.0] * stages
            for i in range(stages - 1):
                a_opt[i] = 1.0 / stages + best_local[i]
            a_opt[stages - 1] = 1.0 - sum(a_opt[:stages - 1])
            
            offset = stages - 1
            for i in range(stages - 1):
                b_opt[i] = 1.0 / stages + best_local[offset + i]
            b_opt[stages - 1] = 1.0 - sum(b_opt[:stages - 1])
            
            best_a = a_opt[:]
            best_b = b_opt[:]
            best_err = best_local_f
            x_best = best_local
    
    return _normalize_and_clip(best_a, best_b)

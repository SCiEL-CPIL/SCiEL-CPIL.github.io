import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Strategy:
    1. Precompute the cut values and the exact classical optimum.
    2. Precompute the initial state vector.
    3. Define a fast simulation function that constructs the full QAOA unitary 
       matrix on-the-fly (or caches if possible, but on-the-fly is fast enough 
       for n<=10 with GEMM) and applies it to the initial state.
       Actually, constructing the unitary is O(4^n) which is 1M ops for n=10.
       Wait, for n=10, 2^10 = 1024. 1024x1024 matrix construction is cheap.
       Applying it is a matrix-vector product, which is very fast.
       
       Alternative: Precompute the C-phase diagonal vector and X-rotation matrices?
       No, the X rotation is a tensor product, so the full unitary depends on beta.
       
       Let's stick to constructing the full unitary U = U_X(beta_1) U_C(gamma_1) U_X(beta_0) U_C(gamma_0).
       For n=10, this is a 1024x1024 complex matrix.
       Construction:
       U_C is diagonal: diag(exp(-i gamma * cut_vals)).
       U_X is tensor product of Rx(beta).
       
       We can compute U_X efficiently using Kronecker products or just build it.
       For n=10, np.kron of 10 matrices of size 2x2 is fast.
       
       Then U = U_X @ U_C @ U_X @ U_C.
       Then state_final = U @ state_init.
       Then expected_cut = sum(|state_final|^2 * cut_vals).
       
       This is much faster than the tensor reshape loop.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # Initial state |+>^n
    state_init = np.ones(max_states, dtype=np.complex128)
    state_init /= np.sqrt(max_states)

    # Precompute the X-rotation matrix for a given beta? 
    # No, beta changes. But we can precompute the structure.
    # U_X(beta) = \bigotimes_{i=0}^{n-1} Rx(beta)
    # Rx(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    
    def get_u_x(beta):
        """Construct the full U_X matrix for given beta."""
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        rx = np.array([[c, s], [s, c]], dtype=np.complex128)
        # Kronecker product
        u = rx
        for _ in range(n - 1):
            u = np.kron(u, rx)
        return u

    def get_u_c(gamma):
        """Construct the full U_C matrix (diagonal) for given gamma."""
        phases = np.exp(-1j * gamma * cut_vals)
        # For efficiency in multiplication, we can just keep it as a vector 
        # and multiply, but if we want a single matrix U, we can make it diagonal.
        # However, U_C is diagonal, so U_C @ v is just v * phases.
        # It is faster to keep U_C as a vector and apply it element-wise.
        return phases

    def simulate(params):
        """
        params: [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
        Returns: expected_cut / opt_cut
        """
        gammas = params[:p]
        betas = params[p:]
        
        state = state_init.copy()
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase: state = U_C(g) @ state
            # Since U_C is diagonal, this is element-wise multiplication
            phases_c = np.exp(-1j * g * cut_vals)
            state = state * phases_c
            
            # Apply X rotation: state = U_X(b) @ state
            # Construct U_X(b)
            # For n=10, this is a 1024x1024 matrix construction.
            # Is it faster to construct U_X every time or precompute?
            # Construction involves np.kron which is O(4^n).
            # Matrix-vector product is O(4^n).
            # So total is ~2 * 4^n operations.
            # The old way was ~n * 2^n operations but with high overhead.
            # Let's try constructing U_X.
            
            c_b = np.cos(b)
            s_b = -1j * np.sin(b)
            rx = np.array([[c_b, s_b], [s_b, c_b]], dtype=np.complex128)
            
            u_x = rx
            for _ in range(n - 1):
                u_x = np.kron(u_x, rx)
                
            state = u_x @ state
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut / opt_cut

    def objective(params):
        return -simulate(params)

    # Optimization Strategy
    # Use L-BFGS-B with multiple random starts.
    # The simulation is now fast (matrix-vector product).
    # We can afford more restarts.
    
    best_value = -np.inf
    best_params = None
    
    rng = np.random.default_rng(42)
    
    # Estimate cost per restart
    # L-BFGS-B typically takes ~20-50 iterations.
    # Each iteration requires 1 objective eval + ~8 gradient evals (finite diff).
    # So ~50 * 9 = 450 simulations per restart.
    # If simulation takes ~1ms (for n=10), then 0.45s per restart.
    # We have ~115s. So ~250 restarts.
    
    # Let's dynamically adjust the number of restarts based on a quick benchmark.
    # Benchmark one simulation
    t0 = time.time()
    _ = simulate(np.array([0.5, 0.5, 0.5, 0.5]))
    sim_time = time.time() - t0
    
    # Estimate max restarts
    # Assume 50 iterations * 9 evals = 450 sims
    # Plus overhead
    est_time_per_restart = sim_time * 500 
    max_restarts = int((deadline - time.time()) / est_time_per_restart)
    max_restarts = max(10, min(max_restarts, 500)) # Cap at 500 to be safe
    
    for run in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Heuristic: gamma in [0, pi], beta in [0, pi]
        # Sometimes starting near 0 or specific values helps, but random is robust.
        g_init = rng.uniform(0, np.pi, p)
        b_init = rng.uniform(0, np.pi, p)
        x0 = np.concatenate([g_init, b_init])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if -res.fun > best_value:
                best_value = -res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = np.array([0.5, 0.5, 0.5, 0.5])

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T

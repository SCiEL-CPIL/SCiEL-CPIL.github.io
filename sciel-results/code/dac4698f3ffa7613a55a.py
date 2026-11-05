import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Strategy:
    1. Use a Fast Walsh-Hadamard Transform (FWHT) based simulation for the mixer.
       The mixer exp(-i beta sum X_i) is conjugated by the Hadamard transform to a 
       diagonal operator. This allows O(N log N) mixer application instead of 
       O(N * n) with high constants.
    2. Use Nelder-Mead optimization with multiple random restarts. This is 
       derivative-free and robust for low-dimensional problems. The speed of 
       the FWHT simulation allows for many restarts to explore the landscape.
    3. Final polish with L-BFGS-B to refine the best solution.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    
    # Compute cut values
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Precompute popcounts for mixer phases
    popcounts = np.zeros(max_states, dtype=np.int32)
    for s in range(max_states):
        popcounts[s] = bin(s).count('1')
        
    # Mixer phases factor: n - 2 * popcount(s)
    mixer_phases_factor = n - 2 * popcounts
    
    def fwht(x):
        """
        In-place Fast Walsh-Hadamard Transform.
        x is a numpy array of length 2^n.
        """
        h = 1
        N = len(x)
        while h < N:
            # Apply Hadamard transform to blocks of size 2*h
            # x[2i] = x[i] + x[i+h]
            # x[2i+1] = x[i] - x[i+h]
            # Vectorized using slicing
            # We need to be careful with in-place updates.
            # Standard FWHT:
            # for i in range(0, N, 2*h):
            #   for j in range(i, i+h):
            #     x[j], x[j+h] = x[j]+x[j+h], x[j]-x[j+h]
            
            # Vectorized version:
            # Split into even and odd parts of each block
            # This is tricky to do purely with slicing without temporary arrays 
            # for the addition/subtraction to avoid dependency issues in one line.
            # However, for N=1024, a loop over h (10 iterations) is fine.
            
            # Let's use a simple loop for the h steps, but vectorize the inner j loop.
            for i in range(0, N, 2 * h):
                # x[i : i+h] and x[i+h : i+2h]
                a = x[i : i + h].copy()
                b = x[i + h : i + 2 * h].copy()
                x[i : i + h] = a + b
                x[i + h : i + 2 * h] = a - b
            h *= 2
        return x

    def apply_mixer(state, beta):
        """
        Apply exp(-i beta sum X_i) to state using FWHT.
        """
        # 1. FWHT
        ft = state.copy()
        fwht(ft)
        
        # 2. Diagonal phase
        # e^{-i beta (n - 2 popcount(s))}
        phases = np.exp(-1j * beta * mixer_phases_factor)
        ft *= phases
        
        # 3. Inverse FWHT
        # H^2 = N I, so inverse is H / N
        fwht(ft)
        ft /= max_states
        
        return ft

    def simulate_fast(gammas, betas):
        """
        Fast simulation of QAOA circuit.
        """
        state = np.ones(max_states, dtype=np.complex128)
        state /= np.sqrt(max_states)
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase
            state *= np.exp(-1j * g * cut_vals)
            
            # Apply B mixer
            state = apply_mixer(state, b)
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut

    def objective(params):
        """
        Objective function for minimization (we minimize -ratio).
        """
        if time.time() > deadline:
            return 1e10
            
        gammas = params[:p]
        betas = params[p:]
        
        exp_cut = simulate_fast(gammas, betas)
        return -exp_cut / opt_cut

    best_params = None
    best_score = -1e10
    
    # Nelder-Mead with random restarts
    n_restarts = 0
    max_restarts = 200 # Will be limited by time
    
    while time.time() < deadline - 10: # Leave 10s for polish and safety
        # Random initial guess
        x0 = np.random.uniform(0, 2 * np.pi, 2 * p)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                # Early exit if we find a very good solution?
                # The ratio is exp_cut / opt_cut.
                # If -res.fun > 0.99, we are close to optimal.
                if -res.fun > 0.99:
                    # We might still want to try a few more to be sure, 
                    # but we can break if time is tight.
                    pass
        except Exception:
            pass
        
        n_restarts += 1
        
        # If we've done many restarts and haven't improved in a while, 
        # we might reduce the step or just continue until time runs out.
        # For now, just run until time limit.

    if best_params is None:
        best_params = np.zeros(2 * p)
        best_score = objective(best_params)

    # Final polish with L-BFGS-B
    if time.time() < deadline - 2:
        try:
            res_polish = minimize(objective, best_params, method='L-BFGS-B',
                                  bounds=[(0, 2 * np.pi)] * (2 * p),
                                  options={'maxiter': 100, 'ftol': 1e-10})
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T

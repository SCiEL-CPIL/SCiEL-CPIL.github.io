import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Strategy:
    1. Construct the dense unitary operators for the problem Hamiltonian (C) 
       and the mixer Hamiltonian (B).
    2. The QAOA circuit is a sequence of exp(-i gamma C) and exp(-i beta B).
       We can precompute the matrices for C and B.
       Note: exp(-i gamma C) is diagonal. exp(-i beta B) is a tensor product 
       of single-qubit rotations.
       
       However, for p=2 and n<=10, we can simulate the state vector directly 
       but compute the gradient analytically.
       
       Actually, a more robust "different" strategy is to use the fact that 
       the expected value <C> is a smooth function of parameters.
       We can use the parameter-shift rule to get exact gradients.
       
       Let's stick to state-vector simulation for the value, but use 
       analytic gradients for the optimizer. This is structurally different 
       from the random search + Nelder-Mead approach.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
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

    # Precompute the mixer matrix B = sum X_i
    # B is sparse, but for n<=10, we can work with state vectors directly.
    # The operation exp(-i beta B) = tensor product of Rx(beta).
    
    def get_state_and_expectation(gammas, betas):
        """
        Computes the state vector and the expected cut value.
        Returns: state (complex array), expected_cut (float)
        """
        state = np.ones(max_states, dtype=np.complex128)
        state /= np.sqrt(max_states)
        
        # Precompute phases for C? No, gamma changes.
        # But we can precompute the diagonal elements of exp(-i C)
        # D = exp(-i * cut_vals)
        # Then exp(-i gamma C) = D ** gamma ? No, exp(-i gamma C) = diag(exp(-i gamma * c_s))
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase
            # state[s] *= exp(-i g cut_vals[s])
            state *= np.exp(-1j * g * cut_vals)
            
            # Apply X rotations
            # Reshape to tensor
            t = state.reshape([2] * n)
            
            c_b = np.cos(b)
            rot = -1j * np.sin(b)
            
            for i in range(n):
                # Move axis i to front
                t = np.moveaxis(t, i, 0)
                
                t0 = t[0]
                t1 = t[1]
                
                new_t0 = c_b * t0 + rot * t1
                new_t1 = rot * t0 + c_b * t1
                
                # Create new tensor
                t_new = np.empty_like(t)
                t_new[0] = new_t0
                t_new[1] = new_t1
                t = t_new
                
                # Move axis back
                t = np.moveaxis(t, 0, i)
                
            state = t.reshape(max_states)
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return state, expected_cut

    def objective(params):
        """
        Returns negative expected cut ratio for minimization.
        Also computes gradient using parameter-shift rule.
        """
        if time.time() > deadline:
            return 1e10, None
            
        gammas = params[:p]
        betas = params[p:]
        
        # Value
        _, exp_cut = get_state_and_expectation(gammas, betas)
        val = -exp_cut / opt_cut
        
        # Gradient via Parameter Shift Rule
        # For exp(-i theta H), the derivative of <H'> is 0.5 * (<H'>_{theta+pi/2} - <H'>_{theta-pi/2})
        # Here we want d/dtheta <C>.
        # Note: The parameter-shift rule applies to the parameter in the Hamiltonian.
        # Our circuit is: U = ... exp(-i beta B) exp(-i gamma C) ...
        
        # d/dgamma_k <C> = 0.5 * ( <C>_{gamma_k + pi/2} - <C>_{gamma_k - pi/2} )
        # d/dbeta_k <C> = 0.5 * ( <C>_{beta_k + pi/2} - <C>_{beta_k - pi/2} )
        
        grad = np.zeros(2 * p)
        shift = np.pi / 2.0
        
        # Compute gradients for gammas
        for k in range(p):
            g_plus = gammas.copy()
            g_minus = gammas.copy()
            g_plus[k] += shift
            g_minus[k] -= shift
            
            _, exp_plus = get_state_and_expectation(g_plus, betas)
            _, exp_minus = get_state_and_expectation(g_minus, betas)
            
            grad[k] = 0.5 * (exp_plus - exp_minus) / opt_cut
            
        # Compute gradients for betas
        for k in range(p):
            b_plus = betas.copy()
            b_minus = betas.copy()
            b_plus[k] += shift
            b_minus[k] -= shift
            
            _, exp_plus = get_state_and_expectation(gammas, b_plus)
            _, exp_minus = get_state_and_expectation(gammas, b_minus)
            
            grad[p + k] = 0.5 * (exp_plus - exp_minus) / opt_cut
            
        return val, -grad  # Return negative gradient for minimization of negative objective

    # Optimization
    best_score = -1
    best_params = None
    
    # Use L-BFGS-B with analytic gradients
    # Multiple random restarts
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        # Heuristic: start with small values or uniform
        x0 = np.random.uniform(0, np.pi, 2 * p)
        
        try:
            res = minimize(
                objective, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception:
            pass

    # Fallback if optimization failed
    if best_params is None:
        # Try a few random evaluations
        for _ in range(10):
            if time.time() > deadline - 1:
                break
            x0 = np.random.uniform(0, np.pi, 2 * p)
            val, _ = objective(x0)
            if -val > best_score:
                best_score = -val
                best_params = x0.copy()
        
        if best_params is None:
            best_params = np.zeros(2 * p)

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T

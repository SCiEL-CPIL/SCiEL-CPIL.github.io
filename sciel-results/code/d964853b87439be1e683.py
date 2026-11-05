import numpy as np
import time

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Strategy:
    1. Precompute cut values for all 2^n states.
    2. Use a fast combinatorial simulation: 
       The QAOA circuit U = U_X(beta_{p-1}) U_C(gamma_{p-1}) ... U_X(beta_0) U_C(gamma_0) |+>^n
       Since U_C is diagonal (phase kicks) and U_X is a tensor product of single-qubit rotations,
       we can simulate the state evolution efficiently.
       
       However, the directive suggests "exact combinatorial probability aggregation".
       For n<=10, 2^n = 1024. 
       Standard state vector simulation is O(p * n * 2^n) if implemented with qubit-by-qubit gates, 
       or O(p * 4^n) if using full matrices.
       
       A more efficient representation for the specific structure of U_X (product of identical Rx) 
       and U_C (diagonal) allows us to update the state vector in O(2^n) per layer if we exploit 
       the fact that U_X is a Kronecker product. 
       
       Actually, for n=10, 1024x1024 matrix multiplication is ~1M ops. 
       Applying U_X (kron of 10 Rx) to a vector is O(n * 2^n) = 10k ops.
       Applying U_C is O(2^n) = 1k ops.
       Total per evaluation: p * 11k ops. For p=2, ~22k ops.
       This is extremely fast. We can afford thousands of evaluations.
       
       The "combinatorial probability aggregation" likely refers to calculating 
       P(s) = |<s|U|init>|^2 directly.
       
       Let's implement the fast state-vector evolution using the property that 
       U_X = \bigotimes Rx(beta).
       Instead of building the full matrix, we apply the single-qubit rotations 
       sequentially to the state vector, reshaping it to access individual qubits.
       
       Wait, the state vector is indexed by integer bitmask. 
       Applying Rx to qubit k of a state vector of size 2^n:
       We can do this in O(2^n) by iterating over blocks.
       
       Let's implement `simulate_fast(params)`:
       state = np.ones(2^n) / sqrt(2^n)
       for k in range(p):
           # Apply C phase
           state *= exp(-1j * gamma_k * cut_vals)
           # Apply X rotation
           # U_X = Rx(beta) \otimes ... \otimes Rx(beta)
           # This is equivalent to applying Rx(beta) to every qubit.
           # We can do this by iterating through qubits 0 to n-1.
           # For qubit q, the state is reshaped to (2^q, 2, 2^(n-q-1)).
           # The rotation acts on the middle dimension.
           # Rx(beta) = [[c, -is], [-is, c]]
           # v' = Rx @ v
           
           # Optimization: Precompute cos/sin.
           # For each qubit, we perform vectorized operations.
           
       This approach is O(p * n * 2^n). For n=10, p=2: 2 * 10 * 1024 = 20,480 ops.
       This is very fast.
       
       Optimization:
       Use DE (Differential Evolution) for global search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    
    # Efficient cut value calculation
    # For each edge (u, v), we add 1 to cut_vals[s] if bit u != bit v.
    # We can vectorize this using bitwise operations.
    for u, v in edges:
        # Mask for bit u
        mask_u = 1 << u
        mask_v = 1 << v
        # We need to check bits for all states at once.
        # states = np.arange(max_states)
        # bits_u = (states & mask_u) != 0
        # bits_v = (states & mask_v) != 0
        # cut_vals += (bits_u ^ bits_v).astype(float)
        
        # Even faster:
        # The pattern of differing bits repeats every 2^(max(u,v)+1).
        # But simple vectorized ops on 1024 elements are instant.
        states = np.arange(max_states)
        cut_vals += (((states >> u) ^ (states >> v)) & 1).astype(np.float64)

    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # Initial state |+>^n
    state_init = np.ones(max_states, dtype=np.complex128)
    state_init /= np.sqrt(max_states)

    # Precompute qubit reshaping indices? 
    # For n=10, we can just use np.reshape in the loop.
    
    def simulate(params):
        """
        params: [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        Returns: expected_cut / opt_cut
        """
        gammas = params[:p]
        betas = params[p:]
        
        state = state_init.copy()
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase
            state *= np.exp(-1j * g * cut_vals)
            
            # Apply X rotation: U_X = \bigotimes_{i=0}^{n-1} Rx(b)
            c_b = np.cos(b)
            s_b = -1j * np.sin(b)
            
            # Apply Rx(b) to each qubit
            # We iterate through qubits 0 to n-1
            # State shape: (2^n,)
            # To apply gate to qubit q:
            # Reshape to (2^q, 2, 2^(n-q-1))
            # Apply 2x2 matrix to the middle axis.
            
            # Vectorized approach for all qubits at once is hard due to varying shapes.
            # But n is small (<=10). Looping n times is fine.
            
            # Optimization: Precompute the 2x2 matrix
            # rx = [[c_b, s_b], [s_b, c_b]]
            
            for q in range(n):
                # Reshape state to isolate qubit q
                # Dimensions: left (2^q), qubit (2), right (2^(n-q-1))
                left_dim = 1 << q
                right_dim = 1 << (n - q - 1)
                
                # Reshape
                state_reshaped = state.reshape(left_dim, 2, right_dim)
                
                # Apply rotation
                # new_state[0] = c*old[0] + s*old[1]
                # new_state[1] = s*old[0] + c*old[1]
                
                # We need to be careful with axes.
                # state_reshaped has shape (L, 2, R)
                # We want to mix axis 1.
                
                # Copy to avoid aliasing issues during update
                # v0 = state_reshaped[:, 0, :]
                # v1 = state_reshaped[:, 1, :]
                # new_v0 = c_b * v0 + s_b * v1
                # new_v1 = s_b * v0 + c_b * v1
                
                v0 = state_reshaped[:, 0, :]
                v1 = state_reshaped[:, 1, :]
                
                new_v0 = c_b * v0 + s_b * v1
                new_v1 = s_b * v0 + c_b * v1
                
                # Update state_reshaped (which is a view of state)
                # This is tricky because new_v0 depends on old v0, v1.
                # We computed new_v0 and new_v1 from copies (v0, v1 are views, but operations create new arrays).
                # So we can safely assign.
                state_reshaped[:, 0, :] = new_v0
                state_reshaped[:, 1, :] = new_v1
                
                # No need to reshape back, state is still the same buffer.
            
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut / opt_cut

    def objective(params):
        return -simulate(params)

    # Optimization: Differential Evolution
    # Global search is better for QAOA landscapes which are multimodal.
    
    # Bounds
    # Gamma: [0, pi] (periodicity pi)
    # Beta: [0, pi] (periodicity pi, actually 2pi for Rx? Rx(b+pi) = -Rx(b) but global phase doesn't matter for probs? 
    # Rx(b) = cos(b)I - i sin(b)X. 
    # Rx(b+pi) = -cos(b)I + i sin(b)X = -Rx(b). 
    # Global phase -1 doesn't change probabilities. So period is pi? 
    # Wait, Rx(b) acts on |+> -> cos(b)|+> - i sin(b)|->.
    # If we add pi to b, we get -cos(b)|+> + i sin(b)|->. 
    # Probabilities are the same. So [0, pi] is sufficient for Beta too? 
    # Actually, standard QAOA often uses [0, pi] for both.
    
    bounds = [(0, np.pi)] * p + [(0, np.pi)] * p
    
    # DE Parameters
    # pop_size: 50
    # max_iter: 200
    # F: 0.5-0.9
    # CR: 0.5-0.9
    
    pop_size = 50
    max_iter = 200
    F = 0.8
    CR = 0.9
    
    # Initialize population
    rng = np.random.default_rng(42)
    pop = np.random.uniform(0, np.pi, size=(pop_size, 2 * p))
    
    # Evaluate initial population
    fitness = np.array([objective(ind) for ind in pop])
    
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_ind = pop[best_idx]
    
    # Benchmark time
    t0 = time.time()
    _ = objective(pop[0])
    eval_time = time.time() - t0
    
    # Estimate iterations we can afford
    # Each iteration: pop_size evaluations + overhead
    # Let's run as many iterations as possible until deadline
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Create new population
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros_like(fitness)
        
        for j in range(pop_size):
            # Select 3 distinct random individuals
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Clip to bounds
            mutant = np.clip(mutant, 0, np.pi)
            
            # Crossover
            trial = np.zeros_like(mutant)
            for k in range(2 * p):
                if rng.random() < CR or k == rng.randint(2 * p):
                    trial[k] = mutant[k]
                else:
                    trial[k] = pop[j][k]
            
            # Evaluate
            trial_fit = objective(trial)
            
            # Selection
            if trial_fit < fitness[j]:
                new_pop[j] = trial
                new_fitness[j] = trial_fit
            else:
                new_pop[j] = pop[j]
                new_fitness[j] = fitness[j]
                
            # Update global best
            if trial_fit < best_fit:
                best_fit = trial_fit
                best_ind = trial.copy()
                
        pop = new_pop
        fitness = new_fitness
        
        # Convergence check
        if best_fit > -1e-6: # Ratio close to 1.0 (max possible is 1.0? No, expected cut / opt cut <= 1.0)
            # Actually, max ratio is 1.0. 
            # If we hit 1.0, we can stop.
            # But floating point might make it 0.99999.
            if -best_fit > 0.999:
                break

    # Convert best_ind to (p, 2) array
    gammas = best_ind[:p]
    betas = best_ind[p:]
    
    return np.array([gammas, betas]).T


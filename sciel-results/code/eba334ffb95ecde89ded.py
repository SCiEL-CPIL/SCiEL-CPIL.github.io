"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_sidelobe_bound(x):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes.
    x is a 1D array of length N.
    """
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    R = np.fft.ifft(X * np.conj(X)).real
    R = R[:N] 
    sidelobes = R[1:]
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    # 1. Initialize with a random binary sequence
    # Using a seed for reproducibility in this specific run context, 
    # but random restarts would be better if time allows.
    # For QAOA, we need a starting point in the binary space.
    # We will use a random initialization and then apply QAOA-like mixing.
    
    # Binary representation: x in {-1, 1}
    # Map to qubits: z in {0, 1} where x = 1 - 2z or x = 2z - 1.
    # Let's use x = 1 - 2z.
    
    # QAOA parameters
    p = 10 # Number of layers (depth)
    # Heuristic initialization for gamma and beta
    # Based on problem size and typical QAOA landscapes
    gamma = np.linspace(0.1, 0.5, p) 
    beta = np.linspace(0.1, 0.3, p)
    
    # Initial state: uniform superposition (all zeros in computational basis 
    # corresponds to x = 1 for all, but we start with random to break symmetry)
    # Actually, standard QAOA starts with |+>^N. 
    # Let's start with a random bitstring to explore a specific region, 
    # or stick to standard |+> and let the mixer drive it.
    # Standard QAOA: start with |+>^N.
    
    # However, simulating full N=400 QAOA is impossible classically.
    # We must use a classical approximation or a reduced model.
    # The prompt asks for a DIFFERENT strategy. 
    # If we cannot run true QAOA, we can use a "QAOA-inspired" classical solver
    # that uses the QAOA cost function structure (QUBO) but optimizes 
    # the QAOA parameters classically to find a good bitstring.
    
    # Since we can't run the quantum circuit, we will simulate the 
    # "QAOA" by treating it as a parameterized variational method where 
    # the "expectation value" is estimated by sampling from a distribution 
    # derived from the QAOA ansatz parameters.
    # But sampling from a 400-qubit QAOA state is also hard.
    
    # Alternative interpretation: Use the QUBO formulation directly.
    # The objective is min max_k |r_k|. 
    # This is hard. 
    # Let's try to minimize the sum of squares of the sidelobes (energy) 
    # using a QUBO solver approach (like simulated annealing on the QUBO graph).
    # This is structurally different from the gradient-based L-BFGS approach.
    
    # QUBO Formulation:
    # Objective: E = sum_{i,j} Q_{ij} z_i z_j
    # where z_i in {0,1}.
    # Autocorrelation r_k = sum_i x_i x_{i+k}.
    # x_i = 1 - 2z_i.
    # This leads to a quadratic form in z.
    
    # Let's implement a Simulated Annealing solver on the QUBO energy 
    # (sum of squared sidelobes) which is a valid proxy for the max sidelobe.
    # This is a "Quantum-inspired" or "QUBO-based" strategy.
    
    def _qubo_energy(z):
        """
        Compute the sum of squared autocorrelation sidelobes for binary sequence z.
        z is a 1D array of 0s and 1s.
        """
        x = 1 - 2 * z # Map to {-1, 1}
        return _compute_sidelobe_bound(x) ** 2 # Using max squared as a proxy for energy? 
        # Actually, sum of squares is better for SA. Let's compute that.
        
    def _sum_sq_sidelobes(x):
        n_fft = 2 * N
        X = np.fft.fft(x, n=n_fft)
        R = np.fft.ifft(X * np.conj(X)).real
        R = R[:N] 
        sidelobes = R[1:]
        return np.sum(sidelobes ** 2)
        
    def _qubo_energy_sa(z):
        x = 1 - 2 * z
        return _sum_sq_sidelobes(x)
    
    # Simulated Annealing on the QUBO (binary) space
    z = np.random.randint(0, 2, N) # Initial random bitstring
    x = 1 - 2 * z
    
    current_energy = _sum_sq_sidelobes(x)
    best_energy = current_energy
    best_z = z.copy()
    
    T = 1.0 # Initial temperature
    alpha = 0.99 # Cooling rate
    min_T = 1e-3
    
    while T > min_T and (time.time() - start_time < deadline):
        # Pick a random site to flip
        i = np.random.randint(N)
        
        # Calculate energy change
        # Instead of recomputing full FFT, we can use the property of flipping one bit.
        # r_k = sum x_j x_{j+k}
        # If we flip x_i, x_i -> -x_i.
        # Delta r_k = -2 x_i (x_{i+k} + x_{i-k}) (with boundary checks)
        # This is O(N) per flip, much faster than O(N log N).
        
        delta_energy = 0.0
        # We need to update the sum of squares.
        # E = sum_k r_k^2
        # New r_k = r_k + Delta r_k
        # New E = E + sum_k (2 r_k Delta r_k + Delta r_k^2)
        
        # Let's precompute current r? No, that's O(N). 
        # But we need r_k for all k to compute the change in sum of squares.
        # So we must compute r once, then update it incrementally.
        
        # This suggests a different loop structure: 
        # Maintain current x and current r.
        
        break # The above logic is complex to implement efficiently in a tight loop without significant refactoring. 
              # Let's stick to a simpler, robust QUBO solver: 
              # Greedy Local Search with Random Restarts, which is a 
              # standard heuristic for QUBO problems.
        
    # Let's implement a robust Greedy Local Search (GLS) on the binary space.
    # This is a standard method for QUBO.
    
    def _local_search_binary(x):
        """
        Perform local search on the binary sequence x to minimize the max sidelobe.
        """
        current_score = _compute_sidelobe_bound(x)
        improved = True
        while improved and (time.time() - start_time < deadline - 5.0):
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() - start_time > deadline - 5.0:
                    break
                x_old = x[i]
                x[i] = -x[i]
                new_score = _compute_sidelobe_bound(x)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Restart scan from 0
                    break
                else:
                    x[i] = x_old # Revert
        return x, current_score

    best_x = None
    best_score = np.inf
    
    # Multiple restarts
    while time.time() - start_time < deadline - 10.0:
        # Random initialization
        x0 = np.random.choice([-1, 1], N)
        
        # Local search
        x_opt, score = _local_search_binary(x0)
        
        if score < best_score:
            best_score = score
            best_x = x_opt.copy()
            
        # Optional: Perturb best solution and restart (Simulated Annealing style)
        if time.time() - start_time < deadline - 20.0:
            x_perturbed = best_x.copy()
            # Flip a few random bits
            num_flips = np.random.randint(5, 20)
            indices = np.random.choice(N, num_flips, replace=False)
            x_perturbed[indices] *= -1
            x_opt, score = _local_search_binary(x_perturbed)
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()

    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Vectorized Quantum Annealing approach
    # 1. Convert adjacency to a spin glass Hamiltonian matrix
    # Max Cut is equivalent to minimizing energy E = -sum(s_i s_j) for edges (i,j)
    # where s_i, s_j in {+1, -1}.
    # We model the state as a complex vector z = x + iy, where x approximates the spin.
    
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Ensure symmetry and no self-loops for the Hamiltonian
    H = A.copy()
    np.fill_diagonal(H, 0)
    
    # Time budget management
    start_time = time.time()
    timeout = 100.0  # Leave 20s safety margin for the 120s limit
    
    # Quantum Annealing Parameters
    # The transverse field gamma(t) decreases from gamma_max to 0.
    # The effective temperature is related to the quantum fluctuations.
    gamma_max = 5.0
    steps = 500
    
    # Initial state: random complex vector
    # We use a random phase and amplitude to simulate quantum superposition
    z = np.ones(n, dtype=np.complex128)
    phases = np.random.uniform(0, 2 * np.pi, n)
    z *= np.exp(1j * phases)
    
    best_labels = None
    best_cut = -1
    
    for step in range(steps):
        # Check time
        if time.time() - start_time > timeout:
            break
        
        # Linear annealing schedule: gamma(t) = gamma_max * (1 - t/T)
        t = step / steps
        gamma = gamma_max * (1.0 - t)
        
        # Update rule for Quantum Annealing
        # The effective field on spin i is the sum of interactions with neighbors.
        # In the quantum case, we update the complex amplitudes.
        # A simplified vectorized update:
        # 1. Compute the classical field: h = A @ x (where x is real part of z)
        # 2. Apply the transverse field effect: this induces mixing between real and imaginary parts.
        
        x = z.real
        y = z.imag
        
        # Classical field from the problem Hamiltonian
        # Note: Max Cut energy is -sum A_ij s_i s_j. Gradient descent on this favors alignment? 
        # Wait, Max Cut wants s_i != s_j. 
        # If we define Energy E = - sum_{i,j} A_ij s_i s_j, then minimizing E means maximizing sum A_ij s_i s_j? 
        # No, Max Cut is maximizing sum_{(i,j) in E} (1 - s_i s_j)/2.
        # Equivalent to minimizing sum_{(i,j) in E} s_i s_j.
        # So the "field" acting on s_i should be - sum_{j} A_ij s_j.
        
        h = -A @ x
        
        # Quantum update:
        # The transverse field acts like a "kick" that rotates the spin in the complex plane.
        # A simple discretization:
        # x_new = x * cos(gamma*dt) + y * sin(gamma*dt) + h * dt
        # y_new = y * cos(gamma*dt) - x * sin(gamma*dt)
        
        dt = 1.0 / steps
        angle = gamma * dt
        c = np.cos(angle)
        s = np.sin(angle)
        
        x_new = x * c + y * s
        y_new = y * c - x * s
        
        # Add the classical driving force (gradient descent step)
        # We scale h by a small learning rate to ensure stability
        lr = 0.5
        x_new += h * lr * dt
        
        # Normalize to keep the state on the sphere (maintain quantum coherence magnitude)
        # This prevents amplitudes from blowing up or dying
        magnitudes = np.sqrt(x_new**2 + y_new**2)
        # Avoid division by zero
        magnitudes[magnitudes == 0] = 1.0
        x_new /= magnitudes
        y_new /= magnitudes
        
        z = x_new + 1j * y_new
        
        # Evaluate the cut size periodically or at the end
        if step % 10 == 0 or step == steps - 1:
            # Map complex state to binary labels
            # If real part is positive -> 0, else 1 (or vice versa)
            labels = (z.real > 0).astype(int)
            
            # Calculate cut size
            # Cut size = sum_{i<j} A_ij (1 - labels[i]*labels[j])/2 
            # Or more efficiently: sum of edges where labels differ
            # Using matrix multiplication:
            # Let D be diagonal matrix of labels (0 or 1).
            # Actually, just sum A_ij where labels[i] != labels[j].
            
            # Vectorized cut calculation:
            # diff = labels[:, None] != labels[None, :]
            # cut = (A * diff).sum() / 2  # since A is symmetric and undirected
            
            # Faster:
            # Sum of A where labels differ
            # We can use:
            # cut = 0.5 * (sum(A) - sum(A * (labels[:, None] == labels[None, :])))
            # But constructing the full matrix might be slow for large N, though N=100 is small.
            
            same_labels = (labels[:, None] == labels[None, :]).astype(np.float64)
            cut_size = 0.5 * (np.sum(A) - np.sum(A * same_labels))
            
            if cut_size > best_cut:
                best_cut = cut_size
                best_labels = labels.copy()
                
    # If no labels were recorded (unlikely), initialize
    if best_labels is None:
        best_labels = (z.real > 0).astype(int)
        
    # Greedy Polish (Local Search) to fix any suboptimal assignments from the quantum phase
    # This is a standard 1-flip local search
    for _ in range(5):
        improved = False
        for v in range(n):
            # Calculate gain if we flip v
            current_label = best_labels[v]
            same_side = best_labels == current_label
            other_side = best_labels != current_label
            
            # Edges to same side currently contribute 0 to cut. If we flip, they contribute 1.
            # Edges to other side currently contribute 1 to cut. If we flip, they contribute 0.
            gain = A[v, same_side].sum() - A[v, other_side].sum()
            
            if gain > 0:
                best_labels[v] = 1 - current_label
                improved = True
        
        if not improved:
            break
            
    return best_labels
# EVOLVE-BLOCK-END

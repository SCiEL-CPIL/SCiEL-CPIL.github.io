"""
Iterated Local Search (ILS) for Maximum Cut.
Strategy:
1. Initialize with a random configuration.
2. Apply a strong Local Search (Hill Climbing) to reach a local optimum.
3. Apply a Perturbation (kick) to the current best solution to move to a new region.
4. Repeat the Local Search on the perturbed solution.
5. Keep track of the global best solution found.
This approach avoids the inefficiency of random restarts by leveraging the structure of the search landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using Iterated Local Search.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    adj = np.array(adjacency, dtype=np.int32)
    adj = (adj + adj.T) // 2  # Ensure symmetry
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    def get_cut(s):
        """Calculate cut size for spin vector s (values -1, 1)."""
        return 0.5 * (total_degree - s @ (adj @ s))
    
    def local_search(s, max_iters=1000):
        """
        Greedy Local Search: repeatedly flip the vertex that gives the largest
        positive gain in cut size until no improvement is possible.
        Returns the improved solution and its cut value.
        """
        best_s = s.copy()
        best_cut = get_cut(best_s)
        
        # Optimization: Pre-compute adj @ s once per iteration? 
        # No, s changes, so we must recompute. But for n=100, it's fast.
        
        for _ in range(max_iters):
            # Calculate gains for all vertices
            # Gain from flipping v: s[v] * (adj @ s)[v]
            adj_s = adj @ best_s
            gains = best_s * adj_s
            
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            # Find vertex to flip
            v = np.argmax(gains)
            best_s[v] *= -1
            best_cut += max_gain
            
        return best_s, best_cut

    def perturb(s, num_flips):
        """
        Perturb the solution by flipping a random subset of vertices.
        """
        s_perturbed = s.copy()
        if num_flips > 0:
            indices = np.random.choice(n, size=num_flips, replace=False)
            s_perturbed[indices] *= -1
        return s_perturbed

    # Main ILS Loop
    rng = np.random.default_rng(42)
    
    # Initial random solution
    s_current = rng.choice([-1, 1], size=n)
    s_best, cut_best = local_search(s_current)
    
    # Adaptive perturbation strength
    # Start with a small kick, increase if stuck, decrease if improving
    kick_size = 5 
    stagnation_counter = 0
    max_stagnation = 10
    
    while time.time() < DEADLINE:
        # Check time before perturbation
        if time.time() > DEADLINE:
            break
            
        # Perturb the current best (or current state?) 
        # Standard ILS perturbs the current state, but often it's better to perturb the best 
        # if we are stuck, or the current state if we are moving.
        # Let's perturb the current state to explore locally, but keep global best.
        # Actually, perturbing the global best is a common variant (Memetic Algorithm).
        # Let's try perturbing the current best local optimum found in the previous step.
        
        s_perturbed = perturb(s_best, kick_size)
        
        # Local Search on perturbed
        s_new, cut_new = local_search(s_perturbed)
        
        if cut_new > cut_best:
            cut_best = cut_new
            s_best = s_new
            # Improvement found: reduce kick size to exploit
            kick_size = max(1, int(kick_size * 0.9))
            stagnation_counter = 0
        else:
            # No improvement: increase kick size to explore
            kick_size = min(20, int(kick_size * 1.2) + 1)
            stagnation_counter += 1
            
            # If stuck for too long, do a random restart from scratch to escape deep local minima
            if stagnation_counter > max_stagnation:
                s_random = rng.choice([-1, 1], size=n)
                s_new, cut_new = local_search(s_random)
                if cut_new > cut_best:
                    cut_best = cut_new
                    s_best = s_new
                    stagnation_counter = 0
                    kick_size = 5
                else:
                    # Even random restart didn't help, just reset stagnation and keep exploring
                    stagnation_counter = 0
                    kick_size = 10

    # Convert spin vector (-1, 1) back to labels (0, 1)
    labels = (s_best + 1) // 2
    labels = labels.astype(int)
    
    return labels
# EVOLVE-BLOCK-END

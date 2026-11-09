# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We aim for the highest n possible.
    # Known best is 17. We try 17, then 16, etc.
    # We will run a simulated annealing / local search for each n.
    
    for target_n in range(17, 8, -1):
        if time.time() > deadline - 5:
            break
            
        solution = _solve_numba_style(target_n, deadline)
        if solution is not None:
            return {'n': target_n, 'colors': solution}
            
    # Fallback to a known small solution if all fail
    # n=9 solution
    n0 = 9
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    return {'n': n0, 'colors': best_colors}

def _solve_numba_style(n, deadline):
    """
    Simulated Annealing / Local Search to minimize number of monochromatic walks of length 5.
    If count reaches 0, it implies no C6.
    """
    num_edges = n * (n - 1) // 2
    
    # Edge index mapping
    # We store colors in a flat list of size num_edges
    # Index for (i, j) with i < j:
    # idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def get_edge(idx):
        # Inverse of get_idx
        # i is such that i*(n-1) - i*(i-1)//2 <= idx < (i+1)*(n-1) - (i+1)*i//2
        # Let k = n-1
        # i*k - i(i-1)/2 <= idx
        # Approximation: i ~ sqrt(2*idx)
        i = 0
        while i < n:
            base = i * (n - 1) - i * (i - 1) // 2
            if base <= idx < base + (n - 1 - i):
                break
            i += 1
        j_offset = idx - base
        j = i + 1 + j_offset
        return i, j

    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Build adjacency matrices for each color
    # adj[c] is an n x n boolean matrix
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    
    def rebuild_adj():
        for c in range(4):
            adj[c].fill(False)
        for idx in range(num_edges):
            i, j = get_edge(idx)
            c = colors[idx]
            adj[c][i, j] = True
            adj[c][j, i] = True

    rebuild_adj()
    
    def get_penalty():
        """
        Sum of entries in A^5 for all colors.
        This counts walks of length 5.
        """
        total = 0
        for c in range(4):
            A = adj[c].astype(np.int8)
            # Compute A^5
            # A^2, A^4, A^5
            A2 = A @ A
            A4 = A2 @ A2
            A5 = A4 @ A
            total += np.sum(A5)
        return total

    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    current_penalty = get_penalty()
    best_penalty = current_penalty
    best_colors = colors[:]
    
    steps = 0
    max_steps = 50000 # Upper bound on steps, but we break on time
    
    while time.time() < deadline - 2:
        if current_penalty == 0:
            # Verify with exact C6 check just in case (though 0 walks implies 0 C6)
            # Actually, 0 walks of length 5 means no simple path of length 5, so no C6.
            # So we can return immediately.
            return best_colors
        
        # Pick a random edge to flip
        idx = random.randint(0, num_edges - 1)
        i, j = get_edge(idx)
        old_c = colors[idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        # Update adjacency matrices
        adj[old_c][i, j] = False
        adj[old_c][j, i] = False
        adj[new_c][i, j] = True
        adj[new_c][j, i] = True
        
        # Update colors
        colors[idx] = new_c
        
        # Calculate new penalty
        new_penalty = get_penalty()
        
        # Acceptance criteria
        if new_penalty < current_penalty:
            current_penalty = new_penalty
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_colors = colors[:]
        else:
            # Accept with probability exp(-delta / T)
            delta = new_penalty - current_penalty
            if delta < 100: # Avoid overflow
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_penalty = new_penalty
                else:
                    # Revert
                    colors[idx] = old_c
                    adj[new_c][i, j] = False
                    adj[new_c][j, i] = False
                    adj[old_c][i, j] = True
                    adj[old_c][j, i] = True
            
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        if steps > max_steps:
            break
            
    if best_penalty == 0:
        return best_colors
    return None

# EVOLVE-BLOCK-END

import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    def count_c4s(color_matrix):
        """Count C4s in a graph represented by a binary adjacency matrix."""
        n = color_matrix.shape[0]
        # A C4 exists iff there exist two distinct vertices u, v with 
        # at least 2 common neighbors.
        # For each pair (u, v), count common neighbors.
        # If common_neighbors >= 2, then there is at least one C4 involving u, v.
        # Actually, the number of C4s is sum over all pairs (u,v) of C(common_neighbors, 2)
        # But we just need to detect if ANY C4 exists for validation.
        # For optimization, counting total C4s is a good objective.
        
        # Efficient way: 
        # Let A be the adjacency matrix.
        # Common neighbors of u and v is (A @ A)[u, v].
        # Number of C4s = sum_{u < v} C((A @ A)[u, v], 2)
        
        A = color_matrix
        # A @ A gives common neighbors (plus self-loops if we had them, but A is 0-diagonal)
        # Actually, (A @ A)[i, j] = sum_k A[i, k] A[k, j]. Since A is symmetric 0-1 with 0 diagonal,
        # this counts common neighbors.
        common = A @ A
        # We only care about i < j
        # C(x, 2) = x * (x - 1) / 2
        # Sum over i < j
        total_c4s = 0
        for i in range(n):
            for j in range(i + 1, n):
                x = common[i, j]
                if x >= 2:
                    total_c4s += x * (x - 1) // 2
        return total_c4s

    def count_c4s_fast(A):
        """Faster count using numpy."""
        common = A @ A
        # Extract upper triangle excluding diagonal
        idx = np.triu_indices_from(common, 1)
        vals = common[idx]
        # C(x, 2) = x*(x-1)/2
        c4s = (vals * (vals - 1) // 2)
        return int(c4s.sum())

    def local_search(n, time_limit):
        """
        Local search with simulated annealing.
        Objective: sum of C4s in each color class.
        """
        # Initialize colors: 0..4 for each edge
        # Representation: color_matrix[color][i][j] = 1 if edge (i,j) has this color
        # Or: colors[i][j] = color value. 
        # For fast C4 counting, we need adjacency matrices per color.
        
        # Let's use a flat array for colors: index = i*n - i*(i+1)//2 + j - i - 1
        # Or simply a 2D array colors[n][n] where colors[i][j] is the color.
        
        colors = np.random.randint(0, 5, (n, n), dtype=np.int8)
        # Symmetrize and zero out diagonal
        colors = (colors + colors.T) // 2
        np.fill_diagonal(colors, -1)
        
        # Initial adjacency matrices
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        for c in range(5):
            adj[c] = (colors == c).astype(np.int8)
        
        def get_edges(c):
            # Get list of edges for color c
            idx = np.where(adj[c] == 1)
            return list(zip(idx[0], idx[1]))
        
        # Initial objective
        total_violations = sum(count_c4s_fast(adj[c]) for c in range(5))
        best_violations = total_violations
        best_colors = colors.copy()
        
        # Simulated annealing
        temp = 1.0
        cooling = 0.995
        no_improve = 0
        
        while time.time() < start_time + time_limit:
            # Pick a random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = colors[i, j]
            if old_color == -1:
                continue
            
            # Pick a new color
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            # Update adjacency matrices
            adj[old_color][i, j] = 0
            adj[old_color][j, i] = 0
            adj[new_color][i, j] = 1
            adj[new_color][j, i] = 1
            
            # Compute new violations
            # We can recompute only for old_color and new_color
            v_old = count_c4s_fast(adj[old_color])
            v_new = count_c4s_fast(adj[new_color])
            # Other colors unchanged
            v_other = sum(count_c4s_fast(adj[c]) for c in range(5) if c != old_color and c != new_color)
            new_violations = v_old + v_new + v_other
            
            delta = new_violations - total_violations
            
            if delta < 0 or (random.random() < min(1.0, np.exp(-delta / temp))):
                colors[i, j] = new_color
                colors[j, i] = new_color
                total_violations = new_violations
                no_improve = 0
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors
            else:
                # Revert
                adj[old_color][i, j] = 1
                adj[old_color][j, i] = 1
                adj[new_color][i, j] = 0
                adj[new_color][j, i] = 0
                no_improve += 1
                if no_improve > 1000:
                    # Restart with randomization? Or just cool down
                    temp *= 0.9
                    no_improve = 0
            
            temp *= cooling
            
            if total_violations == 0:
                return colors
        
        return best_colors if best_violations == 0 else None

    def validate(colors, n):
        """Check if colors is a valid C4-free 5-coloring."""
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i, j]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
        for c in range(5):
            if count_c4s_fast(adj[c]) > 0:
                return False
        return True

    # Strategy: Try n=26, then 25, 24...
    for n in [26, 25, 24]:
        # Allocate time based on n
        # n=26 is hardest, give it most time
        if n == 26:
            t_limit = 80.0
        elif n == 25:
            t_limit = 30.0
        else:
            t_limit = 15.0
        
        if time.time() > start_time + t_limit:
            continue
        
        # Run local search
        result = local_search(n, t_limit)
        if result is not None:
            # Validate
            if validate(result, n):
                # Convert to required format
                color_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        color_list.append(int(result[i, j]))
                return {'n': n, 'colors': color_list}
        
        if time.time() > start_time + t_limit + 10:
            break
    
    # Fallback: Return a small valid graph
    # n=2 is always valid
    return {'n': 2, 'colors': [0]}

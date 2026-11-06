import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper to get edge index in lexicographic order
    # idx(i, j) for i < j
    # Number of edges before i: i * (n - 1) - i * (i - 1) // 2
    # Position of j among vertices > i: j - i - 1
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def generate_cycles(n):
        """Generate all C6 cycles in K_n as lists of edge indices."""
        cycles = []
        verts = list(range(n))
        # There are 30 distinct cycles on 6 labeled vertices (up to rotation/reflection)
        # But we just generate all permutations and deduplicate by edge set?
        # Or just generate all combinations of 6 vertices and all 30 cycle structures.
        
        # The 30 cycle structures on vertices 0..5:
        # A cycle is a permutation of 1..5 with 0 fixed at start, 
        # modulo reversal.
        # Actually, simpler: iterate all permutations of 1,2,3,4,5.
        # Total 120 permutations. Each cycle is counted twice (forward/backward).
        # So 60 unique cycles per set of 6 vertices.
        
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                # Canonical form to avoid duplicates: 
                # The cycle is defined by the set of edges.
                # To avoid duplicates, we can require perm[0] < perm[-1]
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    edges.append(get_edge_index(u, v, n))
                cycles.append(tuple(edges))
                
        return cycles

    def try_solve(n):
        num_edges = n * (n - 1) // 2
        
        # Precompute cycles
        cycles = generate_cycles(n)
        if not cycles:
            return None
            
        # Convert to numpy array for fast indexing
        # Each cycle is 6 edges
        cycles_arr = np.array(cycles, dtype=np.int32)
        
        # Initialize colors
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        def evaluate(c):
            # Get colors of edges for all cycles
            # Shape: (num_cycles, 6)
            c_edges = c[cycles_arr]
            # Check if all 6 are the same
            # Compare each element to the first element of the row
            is_mono = np.all(c_edges == c_edges[:, 0:1], axis=1)
            return np.sum(is_mono)
        
        current_score = evaluate(colors)
        if current_score == 0:
            return colors.tolist()
            
        best_score = current_score
        best_colors = colors.copy()
        
        # Hill Climbing with Random Restarts
        # We will run for a certain number of steps or time
        steps_per_restart = 2000
        max_restarts = 1000 # High number, limited by time
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            if r == 0:
                # Use initial random
                pass
            else:
                colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
                current_score = evaluate(colors)
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                if current_score == 0:
                    return colors.tolist()
                    
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    break
                    
                # Pick random edge
                idx = np.random.randint(num_edges)
                old_color = colors[idx]
                
                # Pick new color
                new_color = np.random.randint(0, 3)
                if new_color >= old_color:
                    new_color += 1
                    
                colors[idx] = new_color
                new_score = evaluate(colors)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                    if current_score == 0:
                        return colors.tolist()
                else:
                    # Revert
                    colors[idx] = old_color
        
        if best_score == 0:
            return best_colors.tolist()
        return None

    # Try n=17 first
    n_target = 17
    result = try_solve(n_target)
    if result is not None:
        return {'n': n_target, 'colors': result}
        
    # Fallback to seed solution n=9 if we didn't find a valid 17
    # The seed is:
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}

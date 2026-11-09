import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 115  # Leave 5s margin
    
    # We will try to find a coloring for n=17, then n=18
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    def precompute_cycles(n):
        """
        Precompute all C6s in K_n.
        Returns:
        cycle_edges: list of lists of edge indices
        edge_to_cycles: list of lists of C6 indices
        """
        # Map (u, v) to edge index
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx[(i, j)] = idx
                edge_idx[(j, i)] = idx
                idx += 1
        
        cycle_edges = []
        edge_to_cycles = [[] for _ in range(idx)]
        
        # Generate all combinations of 6 vertices
        for verts in itertools.combinations(range(n), 6):
            # Generate all 60 distinct cycles on these 6 vertices
            # Fix v0 as the smallest to reduce symmetry? 
            # No, we need all distinct cycles.
            # Number of Hamiltonian cycles in K_6 is 60.
            # We can generate them by permuting the other 5 vertices and fixing one?
            # A cycle is defined by the sequence of vertices.
            # To avoid duplicates, we can fix the smallest vertex at position 0,
            # and ensure the second vertex is smaller than the last?
            # Actually, just generate all 5! = 120 permutations of the other 5,
            # and divide by 2 for direction?
            # Let's just generate all 120 permutations of the 6 vertices,
            # and keep only those where v[0] is the smallest and v[1] < v[5].
            
            v0 = min(verts)
            others = [v for v in verts if v != v0]
            
            for perm in itertools.permutations(others):
                # Cycle: v0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - v0
                # Check symmetry: perm[0] < perm[4]
                if perm[0] < perm[4]:
                    # This is a unique cycle
                    verts_seq = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
                    
                    # Get edge indices for this cycle
                    edges = []
                    for k in range(6):
                        u = verts_seq[k]
                        v = verts_seq[(k+1) % 6]
                        edges.append(edge_idx[(u, v)])
                    
                    cycle_idx = len(cycle_edges)
                    cycle_edges.append(edges)
                    
                    # Update edge_to_cycles
                    for e in edges:
                        edge_to_cycles[e].append(cycle_idx)
                        
        return cycle_edges, edge_to_cycles

    def solve_for_n(n):
        """
        Find a valid 4-coloring of K_n using local search.
        Returns colors list or None.
        """
        num_edges = n * (n - 1) // 2
        
        # Precompute cycles
        cycle_edges, edge_to_cycles = precompute_cycles(n)
        num_cycles = len(cycle_edges)
        
        # If no cycles, any coloring is valid
        if num_cycles == 0:
            return [0] * num_edges
            
        # Initialize random coloring
        colors = np.array([random.randint(0, 3) for _ in range(num_edges)], dtype=np.int8)
        
        # Initialize c6_counts: shape (num_cycles, 4)
        c6_counts = np.zeros((num_cycles, 4), dtype=np.int8)
        
        # Populate c6_counts
        for k, edges in enumerate(cycle_edges):
            for e in edges:
                c6_counts[k, colors[e]] += 1
                
        # Calculate initial mono_count
        # A cycle is mono if any count is 6
        # Since sum is 6, mono means one count is 6.
        mono_count = np.sum(np.any(c6_counts == 6, axis=1))
        
        # Local search
        temp = 1.0
        cool_rate = 0.999
        best_mono = mono_count
        best_colors = colors.copy()
        
        while time.time() - start_time < deadline:
            if best_mono == 0:
                return best_colors.tolist()
                
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            c_old = colors[e]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            # Calculate delta
            delta = 0
            affected_cycles = edge_to_cycles[e]
            
            for k in affected_cycles:
                # Update counts
                c6_counts[k, c_old] -= 1
                c6_counts[k, c_new] += 1
                
                # Check if mono status changed
                # Was mono?
                was_mono = False
                for c in range(4):
                    if c6_counts[k, c] == 6:
                        was_mono = True
                        break
                        
                # Is mono now?
                is_mono = False
                for c in range(4):
                    if c6_counts[k, c] == 6:
                        is_mono = True
                        break
                        
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
                    
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / temp):
                colors[e] = c_new
                mono_count += delta
                if mono_count < best_mono:
                    best_mono = mono_count
                    best_colors = colors.copy()
            else:
                # Roll back
                for k in affected_cycles:
                    c6_counts[k, c_old] += 1
                    c6_counts[k, c_new] -= 1
                    
            temp *= cool_rate
            if temp < 0.01:
                temp = 0.01
                
        return None

    # Try n=17
    if time.time() - start_time < deadline - 10:
        res = solve_for_n(17)
        if res:
            best_n = 17
            best_colors = res
    
    # Try n=18 if time permits
    if best_n < 18 and time.time() - start_time < deadline - 15:
        res = solve_for_n(18)
        if res:
            best_n = 18
            best_colors = res
            
    return {'n': best_n, 'colors': best_colors}

import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Norm squared of each candidate = 4
    # Constraint for kissing config: |u-v|^2 >= max_norm_sq
    # Since all have norm^2 = 4, we need |u-v|^2 >= 4
    # |u-v|^2 = |u|^2 + |v|^2 - 2*u·v = 8 - 2*u·v >= 4
    # => u·v <= 2
    
    # Precompute dot product matrix for faster checking
    # M[i,j] = candidates[i] · candidates[j]
    # M is symmetric, diagonal is 4
    # We need M[i,j] <= 2 for all selected pairs
    
    # Build conflict graph: edge (i,j) if M[i,j] > 2 (conflict)
    # We want maximum independent set in this graph
    
    # Compute dot products
    # This is a 5280 x 5280 matrix, might be large but let's try
    # Actually, let's use a smarter approach with bitsets
    
    # For each vertex, compute its neighborhood (conflicts) as a bitmask
    # Using Python's arbitrary precision integers for bitsets
    
    # First, compute all dot products
    # candidates @ candidates.T gives the full dot product matrix
    # But 5280x5280 is ~28M entries, which is manageable
    
    dots = candidates @ candidates.T
    
    # Conflict matrix: conflict[i,j] = True if dots[i,j] > 2
    # Note: dots[i,i] = 4, so self-conflict, but we don't care about self
    # For i != j, conflict if dots[i,j] > 2
    
    # Build adjacency bitsets
    # adj[i] is a big int where bit j is set if i and j conflict
    adj = [0] * n_candidates
    for i in range(n_candidates):
        # Get conflicts for vertex i
        # conflicts are j where dots[i,j] > 2 and j != i
        mask = 0
        row = dots[i]
        for j in range(n_candidates):
            if j != i and row[j] > 2:
                mask |= (1 << j)
        adj[i] = mask
    
    # Now solve Maximum Independent Set using branch and bound
    # Tomita's algorithm with coloring for upper bound
    
    def greedy_coloring(vertices, adj):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns list of color classes and the number of colors used.
        vertices: list of vertex indices
        adj: list of adjacency bitsets
        """
        if not vertices:
            return [], 0
        
        remaining = set(vertices)
        color_classes = []
        
        while remaining:
            # Pick a vertex with smallest degree in remaining subgraph
            # For simplicity, just pick the first one
            v = next(iter(remaining))
            remaining.discard(v)
            
            color_class = [v]
            # Add other vertices that don't conflict with v and are in remaining
            # Actually, in greedy coloring for MIS, we want to find a maximal independent set
            # within remaining that is compatible with v
            
            # Let's do a proper greedy coloring:
            # We want to color the subgraph induced by vertices
            # Each color class is an independent set
            
            # Better approach: sequential coloring
            # For each vertex in order, assign smallest possible color
            # But for upper bound in MIS, we want a fast coloring
            
            # Let's use a simpler approach: 
            # color[v] for each v in vertices
            # Greedy: process vertices in some order, assign smallest available color
            
            break
        
        # Let me redo this properly
        # Standard greedy coloring for upper bound
        color = {}
        for v in vertices:
            # Find smallest color not used by neighbors of v in the subgraph
            used_colors = set()
            for u in vertices:
                if u != v and (adj[v] >> u) & 1:
                    if u in color:
                        used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
        
        max_color = max(color.values()) if color else 0
        return color, max_color + 1
    
    def bb_mis(vertices, adj, current_size, best_size, best_set, time_limit):
        """
        Branch and bound for Maximum Independent Set.
        vertices: list of vertex indices to consider
        adj: adjacency bitsets
        current_size: current size of independent set
        best_size: best size found so far
        best_set: best set found so far
        time_limit: deadline
        """
        if time.time() > time_limit:
            return current_size, best_set
        
        if not vertices:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set[:]
            return current_size, best_set
        
        # Upper bound using greedy coloring
        color, num_colors = greedy_coloring(vertices, adj)
        upper_bound = current_size + num_colors
        if upper_bound <= best_size[0]:
            return current_size, best_set
        
        # Branch on a vertex
        # Pick vertex with highest degree in the subgraph for better pruning
        # For simplicity, pick the first vertex
        v = vertices[0]
        remaining = vertices[1:]
        
        # Option 1: include v
        # Find all vertices in remaining that don't conflict with v
        compatible = [u for u in remaining if not ((adj[v] >> u) & 1)]
        current_set.append(v)
        bb_mis(compatible, adj, current_size + 1, best_size, best_set, time_limit)
        current_set.pop()
        
        # Option 2: exclude v
        bb_mis(remaining, adj, current_size, best_size, best_set, time_limit)
        
        return current_size, best_set
    
    # Initialize
    best_size = [0]
    best_set = [[]]
    current_set = []
    
    # Run branch and bound
    # Start with all vertices
    all_vertices = list(range(n_candidates))
    
    # To make it faster, let's first try a greedy approach to get a good initial solution
    # Then use B&B to improve
    
    # Greedy: repeatedly pick vertex with smallest degree in remaining subgraph
    remaining = set(range(n_candidates))
    greedy_set = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Find vertex with smallest degree in remaining
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = 0
            for u in remaining:
                if u != v and ((adj[v] >> u) & 1):
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        greedy_set.append(min_v)
        remaining.discard(min_v)
        # Remove all neighbors of min_v from remaining
        to_remove = []
        for u in remaining:
            if ((adj[min_v] >> u) & 1):
                to_remove.append(u)
        for u in to_remove:
            remaining.discard(u)
    
    if len(greedy_set) > best_size[0]:
        best_size[0] = len(greedy_set)
        best_set[0] = greedy_set[:]
    
    # Now try B&B with time limit
    # But B&B on 5280 vertices might be too slow
    # Let's try a simpler local search instead
    
    # Local search: start with greedy solution, try to add/remove/swap
    current_solution = set(best_set[0])
    
    # Try to add vertices
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Try to add a random vertex not in current solution
        not_in_solution = [v for v in range(n_candidates) if v not in current_solution]
        if not not_in_solution:
            break
        
        np.random.shuffle(not_in_solution)
        
        for v in not_in_solution:
            # Check if v can be added
            can_add = True
            for u in current_solution:
                if ((adj[v] >> u) & 1):
                    can_add = False
                    break
            
            if can_add:
                current_solution.add(v)
    
    if len(current_solution) > best_size[0]:
        best_size[0] = len(current_solution)
        best_set[0] = list(current_solution)
    
    # Try multiple random restarts with greedy
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random greedy
        remaining = set(range(n_candidates))
        greedy_set = []
        
        while remaining:
            if time.time() > deadline:
                break
            # Pick random vertex from remaining
            v = np.random.choice(list(remaining))
            greedy_set.append(v)
            remaining.discard(v)
            to_remove = []
            for u in remaining:
                if ((adj[v] >> u) & 1):
                    to_remove.append(u)
            for u in to_remove:
                remaining.discard(u)
        
        if len(greedy_set) > best_size[0]:
            best_size[0] = len(greedy_set)
            best_set[0] = greedy_set[:]
    
    # Convert to array
    if best_set[0]:
        result = candidates[best_set[0]]
    else:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        result = np.array([v], dtype=np.int64)
    
    return result

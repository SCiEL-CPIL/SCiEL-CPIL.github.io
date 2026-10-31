import numpy as np
import itertools
import time
import sys

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def max_clique_tomita(adj, deadline):
    """
    Tomita's algorithm for Maximum Clique.
    adj: (N, N) boolean adjacency matrix of the compatibility graph.
    deadline: time limit in seconds.
    """
    N = adj.shape[0]
    best_clique = []
    best_size = 0
    current_clique = []
    
    # Precompute degrees for initial ordering
    degrees = adj.sum(axis=1)
    
    # Order vertices by degree descending (or use a specific heuristic)
    # Tomita's original paper suggests ordering by degree in the induced subgraph, 
    # but a static ordering by global degree is a good start for the initial P set.
    order = np.argsort(-degrees)
    
    # We will maintain P (candidate set) as a list of indices
    # To make pruning effective, we need to compute coloring bounds on P.
    
    def greedy_coloring(P):
        """
        Greedy coloring of the subgraph induced by P.
        Returns an array of colors (upper bound on clique size in P)
        and the order of vertices in P sorted by color.
        """
        if len(P) == 0:
            return [], []
        
        # We want to color the graph such that adjacent vertices have different colors.
        # This gives an upper bound on the size of a clique in P (which is an independent set in the complement, but here P is a set of vertices in the clique graph, so a clique in the original graph corresponds to an independent set in the complement... wait.
        # In Max Clique, we look for a set of mutually adjacent vertices.
        # Coloring the subgraph induced by P: if we color it with k colors, any clique in P must have at most one vertex of each color. So max clique size <= k.
        # We assign colors greedily.
        
        P_set = set(P)
        colors = {}
        color_order = []
        
        # Sort P by degree in the induced subgraph descending to improve coloring quality (fewer colors)
        # Degree in induced subgraph:
        degs = {}
        for v in P:
            degs[v] = sum(1 for u in P if u != v and adj[v, u])
        
        sorted_P = sorted(P, key=lambda x: -degs[x])
        
        next_color = 0
        for v in sorted_P:
            # Find the smallest color not used by any neighbor in P
            used_colors = set()
            for u in P:
                if u != v and adj[v, u]:
                    if u in colors:
                        used_colors.add(colors[u])
            
            # Find smallest color not in used_colors
            while next_color in used_colors:
                next_color += 1
            
            colors[v] = next_color
            color_order.append((next_color, v))
        
        # Sort by color, then by vertex index (or any stable order)
        color_order.sort()
        return color_order

    def search(P, X, size_X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return False # Indicate timeout
            
        if len(P) == 0:
            if size_X > best_size:
                best_size = size_X
                best_clique = X.copy()
            return True
            
        # Get coloring of P
        color_order = greedy_coloring(P)
        
        # Process vertices in reverse order of color
        # This is the key to Tomita's algorithm: we branch on vertices with highest colors first?
        # Actually, the standard implementation processes from the last vertex in the color-sorted list backwards.
        # The color of a vertex gives an upper bound on the clique size achievable from that vertex and its predecessors in the sorted list.
        
        # Let's iterate from the end of color_order
        for i in range(len(color_order) - 1, -1, -1):
            color_val, v = color_order[i]
            
            # Upper bound: size_X + color_val
            if size_X + color_val <= best_size:
                # Since we are iterating in decreasing order of color (and thus decreasing upper bound),
                # if this branch is pruned, all subsequent branches (with lower or equal color) are also pruned.
                return True
            
            if time.time() > deadline:
                return False
                
            # Branch on v
            # New P is intersection of P and neighbors of v
            # We must only consider vertices in P that come after v in the color_order? 
            # No, the standard Tomita algorithm maintains P as the set of vertices that are compatible with X and have not been processed yet.
            # When we pick v, the new candidate set P' should be {u in P | u is adjacent to v and u is "before" v in the processing order?}
            # Actually, to avoid duplicates, we typically restrict P' to vertices that appear *after* v in the sorted list used for branching?
            # Let's stick to the standard recursive formulation:
            # P is the set of candidates. We pick v from P.
            # New P = {u in P | u adjacent to v and u != v}
            # To ensure we don't revisit, we can just pass the subset of P that is compatible with v.
            
            # Optimization: Only consider u in P that are also in the "future" of the color order?
            # The standard MCQ algorithm uses the coloring to define the branching order.
            # If we process vertices in order of increasing color (from low to high), we might miss things?
            # Tomita's algorithm:
            # 1. Color P.
            # 2. Let P be ordered by color.
            # 3. For i from len(P)-1 down to 0:
            #    v = P[i]
            #    if |X| + color(v) <= best_size: return
            #    X.add(v)
            #    P_new = P[i+1:] intersect neighbors(v)  <-- This is the key. We only look at vertices that were *after* v in the sorted list.
            #    search(P_new, X, |X|+1)
            #    X.remove(v)
            
            # So we need to slice P_new from the color_order list.
            # The vertices in P_new must be from color_order[i+1:] AND be neighbors of v.
            
            # Let's extract the list of vertices from color_order[i+1:]
            # And filter for neighbors of v.
            
            # Note: color_order contains (color, vertex) pairs.
            # We need to extract the vertices.
            
            # Efficiently getting neighbors:
            # adj[v] is a boolean vector.
            
            # Let's create a list of candidates from the "tail" of the color order
            tail_vertices = [vert for c, vert in color_order[i+1:]]
            
            # Filter tail_vertices to those adjacent to v
            # We can use numpy for this if we map vertex indices to their positions in the original graph.
            # But here vertices are just indices 0..N-1.
            # adj[v, tail_vertices] gives a boolean array.
            
            if len(tail_vertices) > 0:
                # Map tail_vertices to indices for numpy lookup
                # tail_vertices are already indices 0..N-1
                mask = adj[v, tail_vertices]
                new_P = [tail_vertices[j] for j in range(len(tail_vertices)) if mask[j]]
            else:
                new_P = []
                
            # Recursive call
            # We need to pass new_P as the candidate set.
            # The search function expects P as a list of indices.
            if not search(new_P, X + [v], size_X + 1):
                return False # Timeout propagated

        return True

    # Initial call
    # P is all vertices, ordered by degree?
    # The search function handles coloring internally.
    # We start with P = list(range(N))
    # But to help the first coloring, we can just pass the full list.
    
    # One optimization: The initial P can be ordered by degree to help the first greedy coloring?
    # The greedy_coloring function inside search sorts P by induced degree anyway.
    
    search(list(range(N)), [], 0)
    
    return best_clique

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Compatibility: dot <= 2
    # Conflict: dot > 2
    # We want Max Clique in the Compatibility Graph.
    # Compatibility matrix: True if dot <= 2 and i != j
    compat = (dots <= 2).astype(np.int8)
    np.fill_diagonal(compat, 0)
    
    # Convert to boolean for adjacency
    adj = compat.astype(bool)
    
    # Run Tomita Max Clique
    # Note: Tomita can be slow if the graph is very large and dense without good bounds.
    # 5280 nodes is manageable for optimized Tomita in C, but in Python it might be slow.
    # We need to ensure the deadline is respected.
    
    # To speed up Python Tomita, we can use bitsets or numpy operations for intersection.
    # However, the recursive nature and list manipulation are the bottlenecks.
    # Let's try running it. If it's too slow, we might need to fall back to greedy or a hybrid.
    # Given the 115s limit, and Python overhead, a pure Python Tomita on 5280 nodes might time out before finding the optimal solution if the branching factor is high.
    # However, the "coloring bound" is very strong.
    
    # Alternative: Use a greedy heuristic to find a good initial lower bound (best_size) to help pruning.
    # Let's run a quick greedy MIS (or Max Clique) to get a starting best_size.
    
    def greedy_max_clique(adj):
        # Greedy: pick vertex with highest degree, add to clique, remove it and non-neighbors.
        N = adj.shape[0]
        remaining = set(range(N))
        clique = []
        
        while remaining:
            # Find vertex with max degree in remaining
            # This is O(N^2) per step, total O(N^3). 5280^3 is too big.
            # We need a faster greedy.
            # Just pick a random vertex or highest degree in original graph?
            # Let's use a simple heuristic: pick vertex with highest degree in the *current* remaining set.
            # To speed up, we can just pick the first one? No, that's bad.
            
            # Let's use a simpler greedy:
            # Order by degree descending.
            # Iterate, if adjacent to all in clique, add.
            
            # Better: Use the greedy MIS from the original code as a lower bound.
            # The original code found ~1236 (score 2.47 * 500? No, score is k. Seed is 500? 
            # Task says "score is k". 
            # If score is 2.4727, and it's an integer count, maybe the score is normalized?
            # "search score is value/seed". 
            # If value is k, and score is 2.47, then k/seed = 2.47.
            # If seed is 1, k=2.47? No, k is integer.
            # Maybe the "seed program" score is 1.0, which corresponds to k=1?
            # Or maybe the score IS k, and the "2.4727" in the prompt is a relative score?
            # "score is k" ... "search score is value/seed".
            # If the current program scores 2.4727, and the seed scores 1.0, then the current program finds ~2.5 times more vectors than the seed.
            # If seed finds 1 vector, current finds 2 or 3? That's too low for dimension 11.
            # Kissing number in d=11 is known to be at least 165? No, that's small.
            # Actually, kissing number for d=11 is unknown exactly, but bounds are around 165-190?
            # Wait, weight 4 vectors in R^11.
            # Total weight 4 vectors: 5280.
            # If we can pack a significant fraction, we might get hundreds.
            # Let's assume the current greedy finds a good set.
            
            # For the lower bound in Tomita, let's just use the result of a fast greedy.
            # I'll implement a simple greedy clique finder.
            
            # Since we are inside construct, we can reuse the logic or just run a simplified version.
            
            # Let's just start Tomita with best_size = 0. The first path will set it.
            # Pruning will be weak initially, but as we find larger cliques, it improves.
            
            pass
            
    # To improve performance, we can initialize best_size with a greedy solution.
    # Let's implement a fast greedy clique.
    
    def fast_greedy_clique(adj):
        N = adj.shape[0]
        # Order by degree
        degrees = adj.sum(axis=1)
        order = np.argsort(-degrees)
        
        clique = []
        in_clique = np.zeros(N, dtype=bool)
        
        for v in order:
            # Check if v is adjacent to all in clique
            if not clique:
                clique.append(v)
                in_clique[v] = True
            else:
                # Check adjacency to all members of clique
                # adj[v, clique]
                if np.all(adj[v, clique]):
                    clique.append(v)
                    in_clique[v] = True
                    
        return clique

    # Get a lower bound
    init_clique = fast_greedy_clique(adj)
    init_size = len(init_clique)
    
    # Pass this to the search? 
    # The search function above doesn't take an initial best_size.
    # I need to modify the search to accept an initial best_size and best_clique.
    
    # Let's rewrite the Tomita part to be more efficient and accept initial bounds.
    
    def tomita_search(adj, deadline, init_clique):
        N = adj.shape[0]
        best_clique = init_clique.copy()
        best_size = len(init_clique)
        
        def greedy_coloring(P):
            if len(P) == 0:
                return []
            
            P_set = set(P)
            colors = {}
            
            # Calculate induced degrees
            degs = {}
            for v in P:
                # Count neighbors in P
                # Use numpy for speed if P is large?
                # P is a list of indices.
                # adj[v, P] -> boolean array
                # sum()
                degs[v] = np.sum(adj[v, P])
            
            sorted_P = sorted(P, key=lambda x: -degs[x])
            
            next_color = 0
            color_order = []
            for v in sorted_P:
                used_colors = set()
                # Get neighbors of v in P that have been colored
                # We can iterate through colored vertices?
                # Or just check all P?
                # For small P, iterating P is fine.
                for u in P:
                    if u != v and adj[v, u] and u in colors:
                        used_colors.add(colors[u])
                
                while next_color in used_colors:
                    next_color += 1
                    
                colors[v] = next_color
                color_order.append((next_color, v))
                
            color_order.sort()
            return color_order

        def search(P, X, size_X):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return False
                
            if len(P) == 0:
                if size_X > best_size:
                    best_size = size_X
                    best_clique = X.copy()
                return True
                
            color_order = greedy_coloring(P)
            
            for i in range(len(color_order) - 1, -1, -1):
                color_val, v = color_order[i]
                
                if size_X + color_val <= best_size:
                    return True
                    
                if time.time() > deadline:
                    return False
                    
                tail_vertices = [vert for c, vert in color_order[i+1:]]
                
                if len(tail_vertices) > 0:
                    mask = adj[v, tail_vertices]
                    new_P = [tail_vertices[j] for j in range(len(tail_vertices)) if mask[j]]
                else:
                    new_P = []
                    
                if not search(new_P, X + [v], size_X + 1):
                    return False
                    
            return True

        # Start search
        # We can start with all vertices
        # To improve initial pruning, we can order the initial P by degree?
        # The coloring step inside search does this.
        
        search(list(range(N)), [], 0)
        
        return best_clique

    final_clique = tomita_search(adj, deadline, init_clique)
    
    if not final_clique:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[final_clique]
    return result_vectors.astype(np.int64)

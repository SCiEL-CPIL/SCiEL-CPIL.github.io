"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget to be safe
    
    # The Fano plane is a known 3-chromatic 3-uniform hypergraph with 7 edges.
    # We use it as a fallback solution.
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    best_edges = fano_edges
    best_colors = fano_colors
    
    # We look for a 6-edge hypergraph that is 3-chromatic.
    # If found, it is optimal (minimum 3-chromatic 3-uniform hypergraph has 6 edges).
    target_k = 6
    
    # Precompute all possible 6-combinations of 9-vertex triples
    # This is C(84, 6) which is large, so we need a smarter search or pruning.
    # Instead of iterating all combinations, we can use a heuristic or random search 
    # with 2-coloring check.
    
    # A faster approach: Generate random 6-edge hypergraphs and check if they are 3-chromatic.
    # A hypergraph is 3-chromatic if it is not 2-colorable.
    # We can check 2-colorability using backtracking or SAT.
    
    def is_2_colorable(edges):
        """
        Check if the hypergraph defined by 'edges' is 2-colorable.
        Returns True if 2-colorable, False otherwise.
        Uses backtracking.
        """
        n = 9
        # Map vertex to list of edge indices it belongs to
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        # State: color assignment for vertices -1 (unassigned), 0, 1
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                # Check all edges
                for e in edges:
                    c0 = colors[e[0]]
                    c1 = colors[e[1]]
                    c2 = colors[e[2]]
                    if c0 == c1 == c2:
                        return False
                return True
            
            v = idx
            # Try color 0
            colors[v] = 0
            # Check if any edge containing v becomes monochromatic with this assignment
            valid = True
            for ei in vertex_edges[v]:
                e = edges[ei]
                others = [u for u in e if u != v]
                if all(colors[u] == 0 for u in others if colors[u] != -1):
                    # If all assigned others are 0, and v is 0, it's mono
                    # Actually, if any other is unassigned, we can't say it's mono yet.
                    # But if all assigned others are 0, it's not necessarily mono if some are unassigned.
                    # The constraint is: not all 3 are same.
                    # So if v=0, and we have assigned colors for others:
                    # If any other is 1, it's fine.
                    # If all others are 0, it's mono (invalid).
                    # If some are -1, we don't know yet.
                    pass
            
            # Let's use a simpler propagation:
            # An edge is "satisfied" if it has at least two different colors among assigned vertices.
            # An edge is "violated" if all assigned vertices have the same color and no unassigned vertices remain to break it? No.
            # An edge is "violated" if all its vertices are assigned and have the same color.
            
            # Let's just check constraints for fully assigned edges.
            # And prune if an edge has 2 vertices assigned to color 0, and the 3rd is unassigned?
            # No, the 3rd can be 1.
            # Prune if an edge has 2 vertices assigned to 0, and the 3rd is assigned to 0.
            
            # Actually, simple backtracking with constraint checking is fine for 6 edges.
            
            # Check if any edge is violated
            for e in edges:
                c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
                # If all are assigned and same, return False
                if c0 != -1 and c1 != -1 and c2 != -1:
                    if c0 == c1 == c2:
                        colors[v] = -1
                        return False
            
            # Pruning: If an edge has two vertices assigned to 0, and the third is unassigned,
            # the third MUST be 1. This is a unit clause.
            # We can implement unit propagation for efficiency.
            
            if backtrack(idx + 1):
                return True
            
            # Try color 1
            colors[v] = 1
            for e in edges:
                c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
                if c0 != -1 and c1 != -1 and c2 != -1:
                    if c0 == c1 == c2:
                        colors[v] = -1
                        return False
            
            if backtrack(idx + 1):
                return True
            
            colors[v] = -1
            return False
        
        return backtrack(0)

    # Random search for a 6-edge 3-chromatic hypergraph
    # We will try a few thousand random samples.
    import random
    random.seed(42)
    
    all_triples = list(itertools.combinations(range(9), 3))
    
    # To make it faster, we can structure the search.
    # But random sampling might work.
    # Let's try 1000 random samples.
    
    found = False
    for i in range(1000):
        if time.time() > deadline:
            break
            
        edges = random.sample(all_triples, 6)
        # Convert to list of lists
        edges_list = [list(e) for e in edges]
        
        if not is_2_colorable(edges_list):
            # Found a 3-chromatic hypergraph!
            # Now we need to provide a valid 3-coloring.
            # Since it's not 2-colorable, it requires 3 colors.
            # We can find a 3-coloring using backtracking.
            
            def find_3_coloring(edges):
                n = 9
                colors = [-1] * n
                
                def bt(idx):
                    if idx == n:
                        for e in edges:
                            c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
                            if c0 == c1 == c2:
                                return False
                        return True
                    
                    v = idx
                    for c in range(3):
                        colors[v] = c
                        valid = True
                        for e in edges:
                            if v in e:
                                others = [u for u in e if u != v]
                                # Check if this assignment causes a mono edge with assigned others
                                # If all others are assigned and equal to c, invalid.
                                if all(colors[u] == c for u in others if colors[u] != -1):
                                    # If any other is unassigned, it's not necessarily invalid yet.
                                    # But if all others are assigned and c, then it's mono.
                                    if all(colors[u] != -1 for u in others):
                                        valid = False
                                        break
                        if valid and bt(idx + 1):
                            return True
                    colors[v] = -1
                    return False
                
                if bt(0):
                    return colors
                return None
            
            colors = find_3_coloring(edges_list)
            if colors:
                # Verify
                valid = True
                for e in edges_list:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
                if valid:
                    best_edges = edges_list
                    best_colors = colors
                    found = True
                    break
    
    # Ensure types are correct
    best_edges = [list(map(int, e)) for e in best_edges]
    best_colors = list(map(int, best_colors))
    
    return {'edges': best_edges, 'colors': best_colors}

# EVOLVE-BLOCK-END

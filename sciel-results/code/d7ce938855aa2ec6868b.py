import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin
    
    best_edges = None
    best_n = None
    min_edges_found = float('inf')
    
    def is_non_2_colorable(n, edges):
        """Check if hypergraph is NOT 2-colorable using DPLL with unit propagation."""
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute: for each vertex, which edges contain it
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        # Use iterative DPLL with explicit stack
        # State: colors (list of -1/0/1), c0 (count of 0s per edge), c1 (count of 1s per edge)
        
        def propagate_and_check(colors, c0, c1, assignments):
            """
            Apply unit propagation. assignments is a list of (vertex, color) to apply.
            Returns False if conflict, True otherwise. Modifies colors, c0, c1 in place.
            Also updates assignments with any new forced assignments.
            """
            i = 0
            while i < len(assignments):
                v, color = assignments[i]
                i += 1
                
                if colors[v] != -1:
                    if colors[v] != color:
                        return False
                    continue
                
                colors[v] = color
                if color == 0:
                    for e_idx in v_to_e[v]:
                        c0[e_idx] += 1
                        if c0[e_idx] == 5:
                            return False
                        if c0[e_idx] == 4:
                            # Force the remaining unassigned vertex in this edge to be 1
                            edge = edges[e_idx]
                            for u in edge:
                                if colors[u] == -1:
                                    assignments.append((u, 1))
                                    break
                else:
                    for e_idx in v_to_e[v]:
                        c1[e_idx] += 1
                        if c1[e_idx] == 5:
                            return False
                        if c1[e_idx] == 4:
                            edge = edges[e_idx]
                            for u in edge:
                                if colors[u] == -1:
                                    assignments.append((u, 0))
                                    break
            return True
        
        # Recursive DPLL
        def dfs(colors, c0, c1):
            # Check if all assigned
            unassigned = [i for i in range(n) if colors[i] == -1]
            if not unassigned:
                # All assigned, check no edge is monochromatic
                # This should be guaranteed by propagation, but verify
                for e_idx in range(m):
                    if c0[e_idx] == 5 or c1[e_idx] == 5:
                        return False
                return True
            
            # Heuristic: pick vertex that appears in edges with highest "tightness"
            # Tightness = max(c0, c1) for edges containing v
            best_v = -1
            best_score = -1
            for v in unassigned:
                score = 0
                for e_idx in v_to_e[v]:
                    score = max(score, max(c0[e_idx], c1[e_idx]))
                if score > best_score:
                    best_score = score
                    best_v = v
                if best_score == 4:
                    break
            
            if best_v == -1:
                best_v = unassigned[0]
            
            # Try color 0
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            if propagate_and_check(new_colors, new_c0, new_c1, [(best_v, 0)]):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            # Try color 1
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            if propagate_and_check(new_colors, new_c0, new_c1, [(best_v, 1)]):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            return False
        
        init_colors = [-1] * n
        init_c0 = [0] * m
        init_c1 = [0] * m
        
        # If dfs returns True, there exists a valid 2-coloring, so it IS 2-colorable
        # If dfs returns False, no valid 2-coloring exists, so it is NOT 2-colorable
        return not dfs(init_colors, init_c0, init_c1)
    
    def extract_core(n, edges):
        """
        Extract a minimal unsatisfiable core from a set of edges.
        Returns a subset of edges that is still non-2-colorable, and is inclusion-minimal.
        """
        current = list(edges)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    return current
                test = [e for j, e in enumerate(current) if j != i]
                if is_non_2_colorable(n, test):
                    current = test
                    improved = True
                    break
        return current
    
    # Strategy 1: Try random dense hypergraphs on small n, extract core
    for n in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Generate random dense hypergraph
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a moderate number of edges
            m = random.randint(40, min(80, len(all_possible)))
            edges = [list(e) for e in all_possible[:m]]
            
            if is_non_2_colorable(n, edges):
                core = extract_core(n, edges)
                if len(core) < min_edges_found:
                    min_edges_found = len(core)
                    best_edges = core
                    best_n = n
    
    # Strategy 2: Try structured constructions (e.g., based on Fano plane or other designs)
    if time.time() < deadline and (best_edges is None or min_edges_found > 51):
        # Try a known small construction pattern
        # For n=12, try all 5-subsets with some structure
        for n in [10, 11, 12]:
            if time.time() > deadline:
                break
            
            # Generate edges with some bias towards overlapping structure
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Try smaller m values first
            for m in [35, 40, 45, 50, 55]:
                if time.time() > deadline:
                    break
                
                edges = [list(e) for e in all_possible[:m]]
                if is_non_2_colorable(n, edges):
                    core = extract_core(n, edges)
                    if len(core) < min_edges_found:
                        min_edges_found = len(core)
                        best_edges = core
                        best_n = n
    
    if best_edges is None:
        # Fallback: complete 5-uniform hypergraph on 9 vertices (126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    N = 20
    COLORS = 20
    all_pairs = list(combinations(range(N), 2))
    total_possible = len(all_pairs)  # 190
    
    # We'll maintain:
    # edges: set of (u, v) with u < v
    # color_of: dict (u, v) -> color
    # For each pair of edges, we can check if they form part of a 4-cycle
    # 
    # Key insight: A 4-cycle involves 4 vertices a,b,c,d with edges ab, bc, cd, da
    # For each pair of edges (e1, e2), if they share a vertex, they could be
    # two adjacent edges in a 4-cycle. The other two edges would need to connect
    # the other endpoints.
    #
    # More efficiently: for each unordered pair of edges, check if they can be
    # part of a 4-cycle. Actually, let's think differently.
    #
    # When we add an edge (u, v) with color c, we need to check:
    # For every pair of existing edges (a,b) and (c,d) that could form a 4-cycle
    # with (u,v), the 4 edges must have distinct colors.
    #
    # A 4-cycle containing edge (u,v) must use two other edges that connect
    # u and v through a path of length 3. So we need:
    #   u - x - y - v where (u,x), (x,y), (y,v) are edges
    # The 4-cycle is u-x-y-v-u with edges (u,x), (x,y), (y,v), (u,v)
    #
    # So when adding (u,v) with color c, for each path u-x-y-v of length 3
    # in the current graph, we need c != color(u,x), c != color(x,y), c != color(y,v)
    # AND the existing three edges must already have distinct colors among themselves
    # (which they should if the graph was valid before).
    #
    # Actually, the condition is: every 4-cycle must have 4 distinct colors.
    # So we need to check all 4-cycles in the graph.
    #
    # Strategy: Start with a large graph (e.g., complete graph K20), assign random colors,
    # then iteratively fix violations by changing colors or removing edges.
    
    def is_valid_graph(edge_color_map):
        """Check if the current edge-color assignment has no rainbow-violating 4-cycle."""
        # Build adjacency
        adj = {i: set() for i in range(N)}
        for (u, v), c in edge_color_map.items():
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all 4-cycles
        for a in range(N):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c <= b:
                        continue
                    for d in adj[c]:
                        if d <= c:
                            continue
                        if d in adj[a]:
                            # Check 4-cycle a-b-c-d-a
                            colors = [edge_color_map.get((min(a,b),max(a,b))),
                                     edge_color_map.get((min(b,c),max(b,c))),
                                     edge_color_map.get((min(c,d),max(c,d))),
                                     edge_color_map.get((min(a,d),max(a,d)))]
                            if len(set(colors)) < 4:
                                return False
        return True
    
    def count_edges(edge_color_map):
        return len(edge_color_map)
    
    # Strategy: Start with complete graph, random coloring, then iteratively
    # remove edges or recolor to fix violations, maximizing edge count.
    
    best_result = None
    best_edge_count = 0
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        if best_result is not None and best_edge_count >= total_possible:
            break
        
        # Initialize with complete graph and random colors
        edge_color_map = {}
        for i, j in all_pairs:
            edge_color_map[(i, j)] = random.randint(0, COLORS - 1)
        
        # Iteratively fix violations
        for iteration in range(500):
            if time.time() > deadline - 2:
                break
            
            # Find a violating 4-cycle
            adj = {i: set() for i in range(N)}
            for (u, v), c in edge_color_map.items():
                adj[u].add(v)
                adj[v].add(u)
            
            violating_cycle = None
            found = False
            for a in range(N):
                if found:
                    break
                for b in sorted(adj[a]):
                    if b <= a or found:
                        continue
                    for c in sorted(adj[b]):
                        if c <= b or found:
                            continue
                        for d in sorted(adj[c]):
                            if d <= c or found:
                                continue
                            if d in adj[a]:
                                colors = [edge_color_map.get((min(a,b),max(a,b))),
                                         edge_color_map.get((min(b,c),max(b,c))),
                                         edge_color_map.get((min(c,d),max(c,d))),
                                         edge_color_map.get((min(a,d),max(a,d)))]
                                if len(set(colors)) < 4:
                                    violating_cycle = (a, b, c, d, colors)
                                    found = True
                                    break
            
            if not found:
                break
            
            a, b, c, d, colors = violating_cycle
            # Try to fix by recoloring one of the edges
            fixed = False
            for edge in [(min(a,b),max(a,b)), (min(b,c),max(b,c)), (min(c,d),max(c,d)), (min(a,d),max(a,d))]:
                for new_color in range(COLORS):
                    if new_color == edge_color_map[edge]:
                        continue
                    # Check if recoloring fixes this cycle and doesn't create new issues
                    old_color = edge_color_map[edge]
                    edge_color_map[edge] = new_color
                    
                    # Quick check: does this cycle now have distinct colors?
                    new_colors = [edge_color_map.get((min(a,b),max(a,b))),
                                 edge_color_map.get((min(b,c),max(b,c))),
                                 edge_color_map.get((min(c,d),max(c,d))),
                                 edge_color_map.get((min(a,d),max(a,d)))]
                    if len(set(new_colors)) == 4:
                        fixed = True
                        break
                    else:
                        edge_color_map[edge] = old_color
                
                if fixed:
                    break
            
            if not fixed:
                # Remove an edge from the violating cycle
                edge_to_remove = random.choice([(min(a,b),max(a,b)), (min(b,c),max(b,c)), 
                                                 (min(c,d),max(c,d)), (min(a,d),max(a,d))])
                del edge_color_map[edge_to_remove]
        
        # Count edges
        ec = len(edge_color_map)
        if ec > best_edge_count:
            best_edge_count = ec
            best_result = [[u, v, c] for (u, v), c in edge_color_map.items()]
    
    if best_result is None:
        # Fallback: star graph
        best_result = [[0, v, 0] for v in range(1, N)]
    
    return best_result

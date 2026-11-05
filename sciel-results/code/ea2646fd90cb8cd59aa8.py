"""Strong edge colourings via Local Search on Conflict Graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate random graph with max degree d
        # The seed program uses a specific generation method, we must replicate it exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """
        Build a conflict graph where nodes are edge indices.
        Two nodes are connected if the corresponding edges in the original graph
        share a vertex or are joined by an edge (i.e., they cannot have the same color).
        """
        m = len(edges)
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Build adjacency list for conflict graph
        # conflict_adj[i] is a set of edge indices j that conflict with i
        conflict_adj = [set() for _ in range(m)]
        
        # 1. Edges sharing a vertex conflict
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        # 2. Edges joined by an edge conflict
        # If edge e1=(u,v) and edge e2=(w,x) have an edge (v,w) or (u,x) etc in the graph, they conflict.
        # Actually, "joined by an edge" means there is a path of length 1 between an endpoint of e1 and an endpoint of e2.
        # So for edge (u,v), any edge incident to a neighbor of u (other than v) or a neighbor of v (other than u) conflicts.
        # Let's iterate over edges and their neighbors.
        
        # First, build vertex adjacency for original graph
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            # Neighbors of u (excluding v)
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                # Any edge incident to nu conflicts with edge i
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            # Neighbors of v (excluding u)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        return conflict_adj

    def greedy_color(edges, conflict_adj, rng, order=None):
        """Greedy coloring with optional custom order."""
        m = len(edges)
        if m == 0:
            return []
        if order is None:
            # Order by conflict degree (highest first)
            order = sorted(range(m), key=lambda i: -len(conflict_adj[i]))
        
        colours = [0] * m
        for idx in order:
            blocked = set()
            for nb in conflict_adj[idx]:
                if colours[nb] != -1:
                    blocked.add(colours[nb])
            c = 0
            while c in blocked:
                c += 1
            colours[idx] = c
        return colours

    def try_remove_color(colours, conflict_adj, c_remove, rng, m):
        """Try to remove color c_remove by reassigning all edges with that color."""
        edges_with_c = [i for i in range(m) if colours[i] == c_remove]
        if not edges_with_c:
            return True  # Already removed
        
        old_colours = colours[:]
        rng.shuffle(edges_with_c)
        
        for idx in edges_with_c:
            available = []
            for c_new in range(c_remove):
                valid = True
                for nb in conflict_adj[idx]:
                    if colours[nb] == c_new:
                        valid = False
                        break
                if valid:
                    available.append(c_new)
            
            if not available:
                colours[:] = old_colours
                return False
            else:
                colours[idx] = rng.choice(available)
        
        return True

    def swap_based_local_search(colours, conflict_adj, rng, m, time_limit):
        """
        Local search using swap moves: try to swap colors between pairs of edges
        to enable removal of the highest color.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            max_c = max(colours) if m > 0 else -1
            
            # Try to remove colors from max_c down to 1
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                
                # First try direct reassignment
                if try_remove_color(colours, conflict_adj, c_remove, rng, m):
                    improved = True
                    break
                
                # If direct reassignment fails, try swap-based approach
                # For each edge with color c_remove, try to find a swap that makes it reassignable
                success = False
                for idx in edges_with_c:
                    if time.time() > time_limit:
                        break
                    
                    # Find which colors are blocked for idx
                    blocked = set()
                    for nb in conflict_adj[idx]:
                        blocked.add(colours[nb])
                    
                    # Try to unblock one of the colors < c_remove by swapping
                    for c_target in range(c_remove):
                        if c_target not in blocked:
                            continue
                        
                        # Find an edge j with color c_target that conflicts with idx
                        # and try to swap j's color with some other edge k
                        for nb in conflict_adj[idx]:
                            if colours[nb] != c_target:
                                continue
                            
                            # Try to move nb to another color
                            old_colours = colours[:]
                            for c_new in range(c_remove):
                                if c_new == c_target:
                                    continue
                                # Check if c_new is valid for nb
                                valid = True
                                for nb2 in conflict_adj[nb]:
                                    if colours[nb2] == c_new:
                                        valid = False
                                        break
                                if valid:
                                    colours[nb] = c_new
                                    # Now try to reassign idx
                                    available = []
                                    for c_new2 in range(c_remove):
                                        valid2 = True
                                        for nb3 in conflict_adj[idx]:
                                            if colours[nb3] == c_new2:
                                                valid2 = False
                                                break
                                        if valid2:
                                            available.append(c_new2)
                                    
                                    if available:
                                        colours[idx] = rng.choice(available)
                                        success = True
                                        break
                                    else:
                                        colours[:] = old_colours
                            if success:
                                break
                        if success:
                            break
                    if success:
                        break
                
                if success:
                    improved = True
                    break
        
        return colours

    def final_polish(colours, conflict_adj, rng, m, time_limit):
        """Final high-precision polish pass to squeeze out more colors."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            max_c = max(colours) if m > 0 else -1
            
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                
                # Try multiple random reassignment attempts
                for attempt in range(20):
                    old_colours = colours[:]
                    rng.shuffle(edges_with_c)
                    
                    success = True
                    for idx in edges_with_c:
                        available = []
                        for c_new in range(c_remove):
                            valid = True
                            for nb in conflict_adj[idx]:
                                if colours[nb] == c_new:
                                    valid = False
                                    break
                            if valid:
                                available.append(c_new)
                        
                        if not available:
                            success = False
                            break
                        else:
                            # Prefer lower colors
                            available.sort()
                            # With some probability, pick a random one; otherwise pick lowest
                            if rng.random() < 0.3:
                                colours[idx] = rng.choice(available)
                            else:
                                colours[idx] = available[0]
                    
                    if success:
                        improved = True
                        break
                    else:
                        colours[:] = old_colours
                
                if improved:
                    break
        
        return colours

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 80
        restart_count = 0
        
        while restart_count < max_restarts and time.time() < time_limit:
            # 1. Greedy Color with degree-based ordering
            order = sorted(range(m), key=lambda i: -len(conflict_adj[i]))
            # Add some randomness to the order
            if rng.random() < 0.5:
                order = list(range(m))
                rng.shuffle(order)
            
            colours = greedy_color(edges, conflict_adj, rng, order)
            
            # 2. Local Search with swap-based moves
            colours = swap_based_local_search(colours, conflict_adj, rng, m, time_limit)
            
            # 3. Final polish
            colours = final_polish(colours, conflict_adj, rng, m, time_limit)
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
            
            # If we've reached the theoretical lower bound, stop early
            # Lower bound: max degree of conflict graph + 1
            max_conflict_degree = max(len(adj) for adj in conflict_adj) if m > 0 else 0
            if best_count <= max_conflict_degree + 1:
                break
        
        if best_colours is None:
            # Fallback
            best_colours = greedy_color(edges, conflict_adj, rng)
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Allocate time per instance
        # Total 140s. 3 instances. ~45s each.
        inst_start = time.time()
        inst_limit = inst_start + 45.0
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        # Check global deadline
        if time.time() > deadline:
            # If we run out of time, we just return what we have.
            # The loop above handles individual limits.
            pass

    return results
# EVOLVE-BLOCK-END

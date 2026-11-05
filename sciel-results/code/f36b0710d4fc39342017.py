"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # For each neighbor x of u, all edges incident to x conflict with (u,v)
            # because (u,x) is an edge (joined by edge condition)
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            # For each neighbor y of v, all edges incident to y conflict with (u,v)
            for y in adj[v]:
                for x in adj[y]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        
        return conflicts

    def greedy_color(conflicts, order):
        """Greedy colouring following the given order of edge indices."""
        n = len(conflicts)
        colour = [-1] * n
        used_colours = set()
        
        for idx in order:
            blocked = set()
            for j in conflicts[idx]:
                if colour[j] != -1:
                    blocked.add(colour[j])
            c = 0
            while c in blocked:
                c += 1
            colour[idx] = c
            used_colours.add(c)
        
        return colour, len(used_colours)

    def local_search_reduce_colors(conflicts, colour, max_iter=300, deadline=None):
        """Iteratively try to reduce the number of colors by recoloring edges."""
        n = len(conflicts)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            if deadline and time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            # Find the highest color used
            max_c = -1
            for c in colour:
                if c > max_c:
                    max_c = c
            
            if max_c <= 0:
                break
            
            # Get all edges with the max color
            edges_with_max = [i for i in range(n) if colour[i] == max_c]
            
            # Try to recolor each edge with max_c to a lower color
            for idx in edges_with_max:
                blocked = set()
                for j in conflicts[idx]:
                    if colour[j] != -1 and colour[j] != max_c:
                        blocked.add(colour[j])
                
                for c in range(max_c):
                    if c not in blocked:
                        colour[idx] = c
                        improved = True
                        break
                
                if improved:
                    break
        
        return colour

    def two_opt_recolor(conflicts, colour, max_iter=200, deadline=None):
        """Try swapping colors of pairs of edges to reduce total colors."""
        n = len(conflicts)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            if deadline and time.time() > deadline:
                break
            improved = False
            iter_count += 1
            
            max_c = -1
            for c in colour:
                if c > max_c:
                    max_c = c
            
            if max_c <= 0:
                break
            
            edges_with_max = [i for i in range(n) if colour[i] == max_c]
            
            for idx in edges_with_max:
                # Try to find a color c < max_c such that we can recolor idx to c
                # by first recoloring some conflicting edge
                blocked = set()
                for j in conflicts[idx]:
                    if colour[j] != -1:
                        blocked.add(colour[j])
                
                # Try direct recoloring first
                for c in range(max_c):
                    if c not in blocked:
                        colour[idx] = c
                        improved = True
                        break
                
                if improved:
                    break
                
                # Try 2-opt: find a conflicting edge j with colour[j] == max_c
                # and try to move j to some other color, then recolor idx
                for j in conflicts[idx]:
                    if colour[j] == max_c:
                        # Find a color for j that's not blocked by j's conflicts
                        blocked_j = set()
                        for k in conflicts[j]:
                            if colour[k] != -1 and colour[k] != max_c:
                                blocked_j.add(colour[k])
                        
                        for c in range(max_c):
                            if c not in blocked_j:
                                # Move j to c
                                colour[j] = c
                                # Now try to recolor idx
                                blocked_idx = set()
                                for k in conflicts[idx]:
                                    if colour[k] != -1:
                                        blocked_idx.add(colour[k])
                                for c2 in range(max_c):
                                    if c2 not in blocked_idx:
                                        colour[idx] = c2
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                
                if improved:
                    break
        
        return colour

    def color_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict_graph(edges, adj)
        n_edges = len(edges)
        
        if n_edges == 0:
            return []
        
        best_colour = None
        best_count = n_edges
        
        # Vertex degree for ordering
        vertex_deg = defaultdict(int)
        for u, v in edges:
            vertex_deg[u] += 1
            vertex_deg[v] += 1
        
        # Conflict degree
        conflict_deg = [len(conflicts[i]) for i in range(n_edges)]
        
        # Pre-compute various orderings
        deg_desc = sorted(range(n_edges), key=lambda i: conflict_deg[i], reverse=True)
        deg_asc = sorted(range(n_edges), key=lambda i: conflict_deg[i])
        vdeg_desc = sorted(range(n_edges), key=lambda i: (vertex_deg[edges[i][0]] + vertex_deg[edges[i][1]]), reverse=True)
        vdeg_asc = sorted(range(n_edges), key=lambda i: (vertex_deg[edges[i][0]] + vertex_deg[edges[i][1]]))
        
        # Try fixed orderings first
        for order in [deg_desc, deg_asc, vdeg_desc, vdeg_asc]:
            if time.time() > start_time + time_budget:
                break
            colour, count = greedy_color(conflicts, order)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
        
        # Iterative restart loop with random shuffles
        rng = random.Random(12345)
        max_time = start_time + time_budget
        iteration = 0
        
        while time.time() < max_time - 2.0:
            iteration += 1
            
            # Choose strategy based on iteration
            if iteration % 5 == 0:
                # Use degree-sorted with random tie-breaking
                order = list(range(n_edges))
                # Group by conflict degree, shuffle within groups
                deg_groups = defaultdict(list)
                for i in range(n_edges):
                    deg_groups[conflict_deg[i]].append(i)
                order = []
                for deg in sorted(deg_groups.keys(), reverse=True):
                    group = deg_groups[deg][:]
                    rng.shuffle(group)
                    order.extend(group)
            elif iteration % 5 == 1:
                # Pure random
                order = list(range(n_edges))
                rng.shuffle(order)
            elif iteration % 5 == 2:
                # Vertex degree based with random tie-breaking
                order = list(range(n_edges))
                vdeg_groups = defaultdict(list)
                for i in range(n_edges):
                    vd = vertex_deg[edges[i][0]] + vertex_deg[edges[i][1]]
                    vdeg_groups[vd].append(i)
                order = []
                for vd in sorted(vdeg_groups.keys(), reverse=True):
                    group = vdeg_groups[vd][:]
                    rng.shuffle(group)
                    order.extend(group)
            elif iteration % 5 == 3:
                # Random weighted by conflict degree (high degree first with probability)
                order = list(range(n_edges))
                # Sort by random key weighted by conflict degree
                weights = [conflict_deg[i] + 1 for i in range(n_edges)]
                order.sort(key=lambda i: rng.random() / weights[i])
            else:
                # Random
                order = list(range(n_edges))
                rng.shuffle(order)
            
            colour, count = greedy_color(conflicts, order)
            
            if count < best_count:
                best_count = count
                best_colour = colour[:]
            
            # Apply local search if we have time
            if best_colour is not None and time.time() < max_time - 1.0:
                local_deadline = min(max_time - 1.0, time.time() + 0.5)
                best_colour = local_search_reduce_colors(conflicts, best_colour, max_iter=100, deadline=local_deadline)
                
                # Recompute count after local search
                used = set(best_colour)
                new_count = len(used)
                if new_count < best_count:
                    best_count = new_count
        
        # Final polish: try 2-opt recoloring
        if best_colour is not None and time.time() < max_time - 0.5:
            best_colour = two_opt_recolor(conflicts, best_colour, max_iter=100, deadline=max_time - 0.5)
        
        return [best_colour[i] for i in range(n_edges)]

    # Allocate time budget to each instance
    # Give more time to larger/harder instances
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    # Time allocation: proportional to n*d (complexity proxy)
    complexities = [n * d for n, d, _ in specs]
    total_complexity = sum(complexities)
    total_time = deadline - start_time - 5.0  # 5s buffer
    
    out = []
    for idx, spec in enumerate(specs):
        if idx == len(specs) - 1:
            # Last instance gets remaining time
            time_budget = deadline - start_time - 2.0
        else:
            time_budget = (complexities[idx] / total_complexity) * total_time
            time_budget = min(time_budget, deadline - start_time - 10.0)  # Leave time for remaining
        
        if time.time() > deadline - 5.0:
            # Emergency: use simple greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        else:
            out.append(color_instance(*spec, time_budget))
    
    return out
# EVOLVE-BLOCK-END

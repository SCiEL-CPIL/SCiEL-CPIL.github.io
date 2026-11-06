import random
import time
import itertools

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute all 4-cycles and their edges
    # Cycle is represented as a tuple of 4 edges: ((u1,v1), (u2,v2), (u3,v3), (u4,v4))
    # We also need to know which cycles contain which edge for fast lookup.
    
    all_cycles = []
    edge_to_cycles = {}
    
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 4):
        u, v, w, x = combo
        # Cycle 1: u-v-w-x-u
        c1 = ((u, v), (v, w), (w, x), (x, u))
        all_cycles.append(c1)
        for e in c1:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(len(all_cycles) - 1)
            
        # Cycle 2: u-v-x-w-u
        c2 = ((u, v), (v, x), (x, w), (w, u))
        all_cycles.append(c2)
        for e in c2:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(len(all_cycles) - 1)
            
        # Cycle 3: u-w-v-x-u
        c3 = ((u, w), (w, v), (v, x), (x, u))
        all_cycles.append(c3)
        for e in c3:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(len(all_cycles) - 1)
            
    total_cycles = len(all_cycles)
    
    def get_cycle_color_set(adj, color, cycle_idx):
        cycle = all_cycles[cycle_idx]
        if not all(adj[u][v] for u, v in cycle):
            return None # Cycle doesn't exist in current graph
        c1 = color[cycle[0][0]][cycle[0][1]]
        c2 = color[cycle[1][0]][cycle[1][1]]
        c3 = color[cycle[2][0]][cycle[2][1]]
        c4 = color[cycle[3][0]][cycle[3][1]]
        return {c1, c2, c3, c4}

    def is_cycle_rainbow(adj, color, cycle_idx):
        s = get_cycle_color_set(adj, color, cycle_idx)
        if s is None:
            return True
        return len(s) == 4

    def count_violations(adj, color):
        violations = 0
        for i in range(total_cycles):
            if not is_cycle_rainbow(adj, color, i):
                violations += 1
        return violations

    def get_violating_cycles_for_edge(adj, color, edge):
        """
        Returns list of cycle indices that are violating and contain this edge.
        """
        key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        if key not in edge_to_cycles:
            return []
        violating = []
        for idx in edge_to_cycles[key]:
            if not is_cycle_rainbow(adj, color, idx):
                violating.append(idx)
        return violating

    def run_local_search(seed):
        random.seed(seed)
        
        # Initialize with Complete Graph
        adj = [[1]*n for _ in range(n)]
        for i in range(n):
            adj[i][i] = 0
        color = [[-1]*n for _ in range(n)]
        
        # Random initial coloring
        for u in range(n):
            for v in range(u+1, n):
                c = random.randint(0, num_colors - 1)
                color[u][v] = c
                color[v][u] = c
                
        removed_edges = set()
        
        # Phase 1: Remove edges to satisfy constraint
        # We use a greedy approach: find a violating cycle, pick an edge, remove it.
        # To be smarter, we can pick the edge that is in the most violating cycles.
        
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
                
            # Find one violating cycle
            violating_cycle_idx = -1
            for i in range(total_cycles):
                if not is_cycle_rainbow(adj, color, i):
                    violating_cycle_idx = i
                    break
                    
            if violating_cycle_idx == -1:
                break # No violations, valid coloring found
                
            cycle = all_cycles[violating_cycle_idx]
            
            # Calculate "guilt" for each edge in this cycle
            # Guilt = number of other violating cycles this edge is part of
            # This is an approximation to find a "bad" edge
            best_edge = None
            max_guilt = -1
            for u, v in cycle:
                key = (min(u, v), max(u, v))
                # Count how many violating cycles this edge is in
                # We can just check all cycles containing this edge
                guilt = 0
                if key in edge_to_cycles:
                    for idx in edge_to_cycles[key]:
                        if not is_cycle_rainbow(adj, color, idx):
                            guilt += 1
                if guilt > max_guilt:
                    max_guilt = guilt
                    best_edge = (u, v)
                    
            if best_edge:
                u, v = best_edge
                adj[u][v] = 0
                adj[v][u] = 0
                removed_edges.add((u, v))
            
        # Phase 2: Try to recolor to fix remaining violations
        # This phase is tricky because we might have removed edges unnecessarily.
        # Let's try to recolor edges in the current graph to fix violations.
        
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Find one violating cycle
            violating_cycle_idx = -1
            for i in range(total_cycles):
                if not is_cycle_rainbow(adj, color, i):
                    violating_cycle_idx = i
                    break
                    
            if violating_cycle_idx == -1:
                break # Valid
                
            cycle = all_cycles[violating_cycle_idx]
            
            # Try to recolor one of the edges in this cycle
            improved = False
            for u, v in cycle:
                current_c = color[u][v]
                # Try all other colors
                for new_c in range(num_colors):
                    if new_c == current_c:
                        continue
                    # Temporarily change
                    color[u][v] = new_c
                    color[v][u] = new_c
                    
                    # Check if this specific cycle is fixed
                    if is_cycle_rainbow(adj, color, violating_cycle_idx):
                        # Check if we broke any other cycles
                        # Only check cycles containing (u, v)
                        broke_others = False
                        key = (min(u, v), max(u, v))
                        if key in edge_to_cycles:
                            for idx in edge_to_cycles[key]:
                                if idx == violating_cycle_idx:
                                    continue
                                if not is_cycle_rainbow(adj, color, idx):
                                    # Check if it was already broken before
                                    # We need to know if it was broken BEFORE the change
                                    # This requires storing state. 
                                    # Simpler: Just count total violations before and after?
                                    # That's expensive.
                                    # Let's assume if it's broken now, we need to check if it was broken before.
                                    # We can just revert and try next color if we don't have a fast check.
                                    # For now, let's just check if the *new* state has fewer violations?
                                    # No, let's just check if this specific change fixes the current cycle 
                                    # and doesn't *newly* break any cycle that was previously valid.
                                    
                                    # To do this properly, we need to know which cycles were valid before.
                                    # Let's just check if the new color is valid for all cycles containing (u,v).
                                    # If yes, it's a safe move.
                                    pass
                                    
                        # If we don't have a fast "was it broken before" check, 
                        # we can just check if the new coloring is valid for all cycles containing (u,v).
                        # If yes, it's a good move.
                        safe = True
                        if key in edge_to_cycles:
                            for idx in edge_to_cycles[key]:
                                if not is_cycle_rainbow(adj, color, idx):
                                    safe = False
                                    break
                                    
                        if safe:
                            improved = True
                            break
                        else:
                            # Revert
                            color[u][v] = current_c
                            color[v][u] = current_c
                            
                    else:
                        # Revert
                        color[u][v] = current_c
                        color[v][u] = current_c
                        
                if improved:
                    break
                    
            if not improved:
                # If no recolor works, remove an edge from this cycle
                u, v = random.choice(cycle)
                adj[u][v] = 0
                adj[v][u] = 0
                removed_edges.add((u, v))
                
        # Phase 3: Try to add back removed edges
        for u, v in list(removed_edges):
            if time.time() > deadline - 5:
                break
                
            for c in range(num_colors):
                # Temporarily add edge
                adj[u][v] = 1
                adj[v][u] = 1
                color[u][v] = c
                color[v][u] = c
                
                # Check if valid
                # Only need to check cycles containing (u, v)
                valid = True
                key = (min(u, v), max(u, v))
                if key in edge_to_cycles:
                    for idx in edge_to_cycles[key]:
                        if not is_cycle_rainbow(adj, color, idx):
                            valid = False
                            break
                            
                if valid:
                    removed_edges.discard((u, v))
                    break
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    
        # Count edges
        edge_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edge_count += 1
                    
        # Construct result
        result_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    result_edges.append([u, v, color[u][v]])
                    
        return result_edges

    best_edges = []
    best_count = 0
    total_possible_edges = n * (n - 1) // 2
    
    seed = 0
    while time.time() < deadline - 10:
        edges = run_local_search(seed)
        count = len(edges)
        
        if count > best_count:
            best_count = count
            best_edges = edges
            
        if best_count == total_possible_edges:
            break
            
        seed += 1

    return best_edges

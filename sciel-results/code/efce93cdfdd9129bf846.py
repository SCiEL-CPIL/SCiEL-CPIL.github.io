import time
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    n = 20
    # Map edge (u, v) to an index 0..189
    # u < v
    edge_map = {}
    edges_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edges_list.append((u, v))
            idx += 1
    
    # Generate all 4-cycles
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d}
    # There are 3 ways to form a cycle:
    # 1. a-b-c-d-a (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a (edges: ac, cb, bd, da)
    
    c4s = []
    c4_edge_indices = []
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_map[(a, b)]
        e2 = edge_map[(b, c)]
        e3 = edge_map[(c, d)]
        e4 = edge_map[(d, a)]
        c4s.append((a, b, c, d))
        c4_edge_indices.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_map[(a, b)]
        e2 = edge_map[(b, d)]
        e3 = edge_map[(d, c)]
        e4 = edge_map[(c, a)]
        c4s.append((a, b, d, c))
        c4_edge_indices.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_map[(a, c)]
        e2 = edge_map[(c, b)]
        e3 = edge_map[(b, d)]
        e4 = edge_map[(d, a)]
        c4s.append((a, c, b, d))
        c4_edge_indices.append((e1, e2, e3, e4))
        
    num_c4s = len(c4_edge_indices)
    
    # For each edge, list of c4 indices it belongs to
    edge_to_c4s = [[] for _ in range(len(edges_list))]
    for i, c4_edges in enumerate(c4_edge_indices):
        for e_idx in c4_edges:
            edge_to_c4s[e_idx].append(i)
            
    def is_rainbow(c4_idx, colors):
        e1, e2, e3, e4 = c4_edge_indices[c4_idx]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        return len({c1, c2, c3, c4}) == 4
        
    def count_violations(colors):
        count = 0
        for i in range(num_c4s):
            if not is_rainbow(i, colors):
                count += 1
        return count
        
    best_colors = None
    best_violations = float('inf')
    
    # Local Search
    # Initialize with random colors
    # Try a few restarts
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if _time_left() < 10:
            break
            
        colors = [random.randint(0, 19) for _ in range(len(edges_list))]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            best_colors = colors
            best_violations = 0
            break
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Local search steps
        # We can do a fixed number of steps or time-based
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and _time_left() > 5:
            steps += 1
            
            if current_violations == 0:
                best_colors = colors
                best_violations = 0
                break
                
            # Pick a random violated C4
            # To be efficient, we could maintain a list of violated C4s, 
            # but for now, just pick random and check.
            # Optimization: Maintain a set of violated C4s?
            # Given 14k C4s, checking all is 14k ops. 
            # Picking random is 1 op.
            
            while True:
                rand_c4 = random.randint(0, num_c4s - 1)
                if not is_rainbow(rand_c4, colors):
                    break
                    
            # Pick a random edge in this C4
            e1, e2, e3, e4 = c4_edge_indices[rand_c4]
            edge_indices = [e1, e2, e3, e4]
            target_edge = random.choice(edge_indices)
            
            old_color = colors[target_edge]
            
            # Try all other colors and pick the best one
            best_new_color = old_color
            best_delta = 0 # We want to minimize violations, so delta = new_viol - old_viol
            
            # Calculate current violations for C4s affected by target_edge
            # Actually, let's just calculate the change in violations for the whole graph?
            # That's expensive.
            # Let's calculate delta locally.
            # A change in color of target_edge affects all C4s containing target_edge.
            
            # Let's just try a few random new colors or all 19.
            # All 19 is better.
            
            min_delta = float('inf')
            best_c = old_color
            
            for new_c in range(20):
                if new_c == old_color:
                    continue
                # Temporarily change
                colors[target_edge] = new_c
                
                # Calculate delta
                # We only need to check C4s that contain target_edge
                delta = 0
                for c4_idx in edge_to_c4s[target_edge]:
                    was_rainbow = is_rainbow(c4_idx, colors)
                    # Wait, we changed the color, so we need to compare to BEFORE state
                    # This logic is tricky to do "in-place" without storing old state.
                    
                # Simpler: Just count violations for the affected C4s before and after.
                # But we already changed it.
                # Let's revert and do it properly.
                
            # Let's do a simpler hill climb:
            # 1. Store old color.
            # 2. Try a random new color.
            # 3. Calculate new total violations (expensive).
            # 4. If better, keep. Else revert.
            
            # To make it fast, we can approximate.
            # Or, we can just do a few random restarts of the whole local search.
            
            # Let's try a random new color and evaluate only affected C4s.
            # We need to know the "old" rainbow status of affected C4s.
            
            affected_c4s = edge_to_c4s[target_edge]
            
            # Store old status
            old_statuses = {}
            for c4_idx in affected_c4s:
                old_statuses[c4_idx] = is_rainbow(c4_idx, colors)
                
            # Try a random new color
            new_color = random.randint(0, 19)
            colors[target_edge] = new_color
            
            delta = 0
            for c4_idx in affected_c4s:
                new_status = is_rainbow(c4_idx, colors)
                old_status = old_statuses[c4_idx]
                if not old_status and new_status:
                    delta -= 1
                elif old_status and not new_status:
                    delta += 1
                    
            if delta <= 0:
                current_violations += delta
                # Accept
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Reject
                colors[target_edge] = old_color
                
    # If we found a valid coloring, return all edges
    if best_violations == 0:
        result = []
        for i, (u, v) in enumerate(edges_list):
            result.append([u, v, best_colors[i]])
        return result
    else:
        # Fallback: Return a C4-free graph (safe)
        # Use the greedy C4-free construction from the seed
        n = 20
        all_edges_template = [(i, j) for i in range(n) for j in range(i + 1, n)]
        best_edge_count = 0
        best_graph = []
        
        max_trials = 100
        for trial in range(max_trials):
            if _time_left() < 2:
                break
            random.seed(trial)
            all_edges = all_edges_template[:]
            random.shuffle(all_edges)
            
            neighbors = [set() for _ in range(n)]
            graph_edges = []
            
            for u, v in all_edges:
                creates = False
                for x in neighbors[u]:
                    if x == v:
                        continue
                    for y in neighbors[x]:
                        if y == u or y == v:
                            continue
                        if v in neighbors[y]:
                            creates = True
                            break
                    if creates:
                        break
                
                if not creates:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    graph_edges.append((u, v))
            
            if len(graph_edges) > best_edge_count:
                best_edge_count = len(graph_edges)
                best_graph = graph_edges[:]
                
        return [[u, v, 0] for u, v in best_graph]


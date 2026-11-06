import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Three distinct 4-cycles on 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    # State
    edge_colors = {}
    
    def get_conflict_edges():
        """Return set of edges that are part of at least one non-rainbow 4-cycle."""
        conflict_edges = set()
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_edges.add(e)
        return conflict_edges
    
    def is_color_valid_for_edge(e, c, current_edge_colors):
        """Check if assigning color c to edge e keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles.get(e, []):
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in current_edge_colors for oe in other_edges):
                continue
            # Check colors
            other_colors = [current_edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def try_add_edge(e, current_edge_colors):
        """Try to add edge e with a valid color. Returns True if successful."""
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_color_valid_for_edge(e, c, current_edge_colors):
                current_edge_colors[e] = c
                return True
        return False
    
    # Phase 1: Initialize with a structured dense graph
    # Use a near-complete graph but with strategic color assignment
    # Start with all edges, then repair
    for u in range(n):
        for v in range(u+1, n):
            edge_colors[(u,v)] = random.randint(0, num_colors - 1)
    
    # Phase 2: Repair - remove edges that cause conflicts
    # Use a smarter strategy: remove the edge in the most conflict cycles
    max_repair_steps = 500
    for step in range(max_repair_steps):
        if time.time() > deadline:
            break
        
        conflict_edges = get_conflict_edges()
        if not conflict_edges:
            break
        
        # Pick the edge that is in the most conflict cycles
        # Count how many conflict cycles each edge is in
        conflict_cycle_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        if e in conflict_edges:
                            conflict_cycle_count[e] += 1
        
        # Pick the edge with highest conflict count
        e_to_remove = max(conflict_edges, key=lambda e: conflict_cycle_count.get(e, 0))
        del edge_colors[e_to_remove]
    
    # Phase 3: Local search to add back edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edges = set(edge_colors.keys())
    missing_edges = [e for e in all_edges if e not in current_edges]
    
    best_edge_count = len(edge_colors)
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add a missing edge
        if missing_edges:
            e_add = random.choice(missing_edges)
            if try_add_edge(e_add, edge_colors):
                missing_edges.remove(e_add)
                improved = True
                if len(edge_colors) > best_edge_count:
                    best_edge_count = len(edge_colors)
        
        # 2. If couldn't add, try recoloring
        if not improved and edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_color = edge_colors[e_recolor]
            
            # Try all colors for this edge
            colors = list(range(num_colors))
            random.shuffle(colors)
            for new_color in colors:
                if new_color == old_color:
                    continue
                # Temporarily remove
                del edge_colors[e_recolor]
                if is_color_valid_for_edge(e_recolor, new_color, edge_colors):
                    edge_colors[e_recolor] = new_color
                    # Check if this allows adding a missing edge
                    if missing_edges:
                        e_add = random.choice(missing_edges)
                        if try_add_edge(e_add, edge_colors):
                            missing_edges.remove(e_add)
                            improved = True
                            if len(edge_colors) > best_edge_count:
                                best_edge_count = len(edge_colors)
                            break
                    else:
                        break
                else:
                    # Revert
                    edge_colors[e_recolor] = old_color
                    break
            else:
                # No valid color found, restore old color
                if e_recolor not in edge_colors:
                    edge_colors[e_recolor] = old_color
        
        # 3. Swap move: remove one edge, add another
        if not improved and edge_colors and missing_edges:
            e_remove = random.choice(list(edge_colors.keys()))
            old_color_remove = edge_colors[e_remove]
            del edge_colors[e_remove]
            
            e_add = random.choice(missing_edges)
            if try_add_edge(e_add, edge_colors):
                missing_edges.remove(e_add)
                # Try to re-add e_remove
                if not try_add_edge(e_remove, edge_colors):
                    # Lost an edge, revert
                    edge_colors[e_remove] = old_color_remove
            else:
                # e_add failed, re-add e_remove
                edge_colors[e_remove] = old_color_remove
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 200:
                # Try a more aggressive restart with different density
                no_improve_count = 0
                if time.time() > deadline - 20:
                    break
                # Restart with slightly different parameters
                for u in range(n):
                    for v in range(u+1, n):
                        if random.random() < 0.85:
                            edge_colors[(u,v)] = random.randint(0, num_colors - 1)
                        else:
                            edge_colors.pop((u,v), None)
                # Quick repair
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    conflict_edges = get_conflict_edges()
                    if not conflict_edges:
                        break
                    e_to_remove = random.choice(list(conflict_edges))
                    del edge_colors[e_to_remove]
        else:
            no_improve_count = 0
    
    # Final check and cleanup
    for _ in range(200):
        if time.time() > deadline:
            break
        conflict_edges = get_conflict_edges()
        if not conflict_edges:
            break
        # Remove edge with highest conflict count
        conflict_cycle_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        if e in conflict_edges:
                            conflict_cycle_count[e] += 1
        e_to_remove = max(conflict_edges, key=lambda e: conflict_cycle_count.get(e, 0))
        del edge_colors[e_to_remove]
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result

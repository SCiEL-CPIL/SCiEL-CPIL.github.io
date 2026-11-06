import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and edge-to-cycle mappings
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
            # Normalize edges
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            # Check for duplicates in the cycle itself
            if len(set(norm_cycle)) == 4:
                cycle_idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(cycle_idx)
    
    # State
    edge_colors = {}  # (u,v) -> color
    missing_edges = set()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Initialize with a dense random graph
    p = 0.95
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edge_colors[(u,v)] = random.randint(0, num_colors - 1)
            else:
                missing_edges.add((u,v))
    
    # Cache for is_color_valid_for_edge results: (edge, color) -> bool
    # Invalidated when edge_colors changes
    _color_valid_cache = {}
    _cache_version = 0
    
    def invalidate_cache():
        nonlocal _cache_version
        _cache_version += 1
        _color_valid_cache.clear()
    
    def get_conflict_edges():
        """Return set of edges that are part of at least one non-rainbow 4-cycle."""
        conflict_edges = set()
        conflict_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_edges.add(e)
                        conflict_count[e] += 1
        return conflict_edges, conflict_count
    
    def is_color_valid_for_edge(e, c):
        """Check if assigning color c to edge e results in all cycles containing e being rainbow."""
        if e not in edge_to_cycles:
            return True
        
        cache_key = (e, c, _cache_version)
        if cache_key in _color_valid_cache:
            return _color_valid_cache[cache_key]
        
        result = True
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            other_edges = [other_e for other_e in cycle if other_e != e]
            if not all(other_e in edge_colors for other_e in other_edges):
                continue
            # Check colors
            other_colors = [edge_colors[other_e] for other_e in other_edges]
            # The new color c must be distinct from all other_colors
            if c in other_colors:
                result = False
                break
            # If other colors have duplicates, the cycle is already broken
            if len(set(other_colors)) < 3:
                result = False
                break
        _color_valid_cache[cache_key] = result
        return result
    
    def try_add_edge(e):
        """Try to add edge e with a valid color. Returns True if successful."""
        if e in edge_colors:
            return True
        if e not in missing_edges:
            return False
        
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_color_valid_for_edge(e, c):
                edge_colors[e] = c
                missing_edges.remove(e)
                invalidate_cache()
                return True
        return False
    
    # Phase 1: Repair - remove conflicting edges, prioritizing those in most conflicts
    max_repair_steps = 1000
    for step in range(max_repair_steps):
        if time.time() > deadline:
            break
            
        conflict_edges, conflict_count = get_conflict_edges()
        if not conflict_edges:
            break
        
        # Pick the edge with highest conflict count
        max_count = max(conflict_count.values())
        candidates = [e for e in conflict_edges if conflict_count[e] == max_count]
        e_to_remove = random.choice(candidates)
        del edge_colors[e_to_remove]
        missing_edges.add(e_to_remove)
        invalidate_cache()
    
    # Phase 2: Local Search with more aggressive strategies
    best_edge_count = len(edge_colors)
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # 1. Try to add a missing edge - prioritize edges with fewer constraints
        if missing_edges:
            # Score missing edges by number of cycles they participate in (fewer = easier)
            scored_missing = []
            for e in missing_edges:
                num_cycles = len(edge_to_cycles.get(e, []))
                scored_missing.append((num_cycles, e))
            scored_missing.sort()
            
            # Try the easiest ones first
            for _, e_add in scored_missing[:min(30, len(scored_missing))]:
                if time.time() > deadline:
                    break
                if try_add_edge(e_add):
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    break
            if improved:
                continue
        
        # 2. Try random recoloring of existing edges to enable adding missing edges
        if edge_colors and missing_edges:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_color = edge_colors[e_recolor]
            
            # Try multiple new colors
            new_colors = list(range(num_colors))
            random.shuffle(new_colors)
            found_new = False
            for new_color in new_colors:
                if time.time() > deadline:
                    break
                if new_color == old_color:
                    continue
                del edge_colors[e_recolor]
                invalidate_cache()
                if is_color_valid_for_edge(e_recolor, new_color):
                    edge_colors[e_recolor] = new_color
                    # Check if this recoloring enables adding a missing edge
                    if missing_edges:
                        # Try a few missing edges
                        sample_size = min(15, len(missing_edges))
                        for e_try in random.sample(list(missing_edges), sample_size):
                            if try_add_edge(e_try):
                                improved = True
                                if len(edge_colors) > best_edge_count:
                                    best_edge_count = len(edge_colors)
                                no_improve_count = 0
                                found_new = True
                                break
                    if found_new:
                        break
                    else:
                        # Keep the new color if it's valid, even if no edge was added
                        pass
                else:
                    edge_colors[e_recolor] = old_color
                    invalidate_cache()
                    break
            
            if improved:
                continue
        
        # 3. Try a swap: remove one edge, add a missing one, try to re-add removed
        if random.random() < 0.3 and edge_colors and missing_edges:
            e_remove = random.choice(list(edge_colors.keys()))
            old_remove_color = edge_colors[e_remove]
            del edge_colors[e_remove]
            invalidate_cache()
            
            # Try to add a missing edge
            e_add = random.choice(list(missing_edges))
            added = try_add_edge(e_add)
            
            if added:
                # Try to re-add e_remove
                readded = False
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove, c):
                        edge_colors[e_remove] = c
                        readded = True
                        break
                
                if readded:
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    continue
                else:
                    # Net 0 change (swapped), but we have a different configuration
                    pass
            else:
                # e_add failed, re-add e_remove
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove, c):
                        edge_colors[e_remove] = c
                        break
        
        # 4. Try double swap: remove two edges, add two missing edges
        if random.random() < 0.1 and len(edge_colors) >= 2 and len(missing_edges) >= 2:
            e_remove1 = random.choice(list(edge_colors.keys()))
            e_remove2 = random.choice([e for e in edge_colors if e != e_remove1])
            c1 = edge_colors[e_remove1]
            c2 = edge_colors[e_remove2]
            del edge_colors[e_remove1]
            del edge_colors[e_remove2]
            invalidate_cache()
            
            e_add1 = random.choice(list(missing_edges))
            e_add2 = random.choice([e for e in missing_edges if e != e_add1])
            
            added1 = try_add_edge(e_add1)
            added2 = try_add_edge(e_add2)
            
            if added1 and added2:
                # Try to re-add both removed edges
                readded1 = False
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove1, c):
                        edge_colors[e_remove1] = c
                        readded1 = True
                        break
                
                readded2 = False
                if readded1:
                    for c in range(num_colors):
                        if is_color_valid_for_edge(e_remove2, c):
                            edge_colors[e_remove2] = c
                            readded2 = True
                            break
                
                if readded1 and readded2:
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    continue
                else:
                    # Partial re-add, net change is -1 or 0
                    pass
            else:
                # Failed to add both, re-add removed edges
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove1, c):
                        edge_colors[e_remove1] = c
                        break
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove2, c):
                        edge_colors[e_remove2] = c
                        break
        
        if not improved:
            no_improve_count += 1
    
    # Final cleanup: ensure no conflicts
    max_final_repair = 200
    for _ in range(max_final_repair):
        if time.time() > deadline:
            break
        conflict_edges, _ = get_conflict_edges()
        if not conflict_edges:
            break
        e_to_remove = random.choice(list(conflict_edges))
        del edge_colors[e_to_remove]
        missing_edges.add(e_to_remove)
        invalidate_cache()
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result

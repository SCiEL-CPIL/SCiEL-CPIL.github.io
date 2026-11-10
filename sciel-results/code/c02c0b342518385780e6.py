import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 150.0
    n = 16
    
    # Precompute all K4s and their edge sets
    all_k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in all_k4s:
        v1, v2, v3, v4 = quad
        edges = [(v1,v2), (v1,v3), (v1,v4), (v2,v3), (v2,v4), (v3,v4)]
        k4_edges.append(edges)
    
    # Edge index mapping
    edge_to_idx = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i,j)] = idx
            edges.append((i,j))
            idx += 1
    num_edges = len(edges)
    
    # Initialize coloring
    # Use a structured initial coloring
    colours = [(i + 2*j) % 13 for i, j in edges]
    
    def count_violations(colours):
        """Count K4s with < 5 distinct colours"""
        viol = 0
        for e in k4_edges:
            cs = set()
            for edge in e:
                idx = edge_to_idx[edge]
                cs.add(colours[idx])
            if len(cs) < 5:
                viol += 1
        return viol
    
    def try_recollapse(colours, target_colour):
        """Try to recolor all edges with target_colour to other colours"""
        if time.time() > deadline - 5:
            return None
        edges_with_target = [i for i, c in enumerate(colours) if c == target_colour]
        if not edges_with_target:
            return colours
        
        new_colours = colours[:]
        # Try to recolor each edge to a different existing colour
        other_colours = set(colours) - {target_colour}
        success = True
        for eidx in edges_with_target:
            found = False
            for new_c in other_colours:
                new_colours[eidx] = new_c
                if count_violations(new_colours) <= count_violations(colours):
                    found = True
                    break
                new_colours[eidx] = target_colour
            if not found:
                success = False
                new_colours = colours[:]
                break
        if success:
            return new_colours
        return None
    
    best_colours = colours[:]
    best_viol = count_violations(colours)
    
    # Local search: try to reduce violations and then minimize colours
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        improved = False
        
        # First, fix violations
        if best_viol > 0:
            # Find a violating K4 and try to fix it
            for e in k4_edges:
                cs = set()
                for edge in e:
                    idx = edge_to_idx[edge]
                    cs.add(best_colours[idx])
                if len(cs) < 5:
                    # Try to recolor one edge in this K4 to a new or different colour
                    for edge in e:
                        idx = edge_to_idx[edge]
                        old_c = best_colours[idx]
                        # Try changing to a colour not in current K4's colours
                        for new_c in range(0, 20):
                            if new_c == old_c:
                                continue
                            best_colours[idx] = new_c
                            viol = count_violations(best_colours)
                            if viol < best_viol:
                                best_viol = viol
                                improved = True
                                break
                            best_colours[idx] = old_c
                        if improved:
                            break
                if improved:
                    break
        else:
            # No violations, try to minimize number of colours
            used_colours = sorted(set(best_colours))
            if not used_colours:
                break
            
            # Try to eliminate the highest colour
            target = used_colours[-1]
            new_c = try_recollapse(best_colours, target)
            if new_c is not None:
                best_colours = new_c
                best_viol = count_violations(best_colours)
                if best_viol > 0:
                    # Revert
                    best_colours = [c for c in best_colours]
                    # Actually we need to revert properly
                    break
                improved = True
        
        if not improved and best_viol == 0:
            # Try random swaps
            if random.random() < 0.5:
                eidx = random.randint(0, num_edges - 1)
                old_c = best_colours[eidx]
                new_c = random.randint(0, max(best_colours) + 1)
                best_colours[eidx] = new_c
                viol = count_violations(best_colours)
                if viol <= best_viol:
                    best_viol = viol
                    if viol < best_viol:
                        improved = True
                else:
                    best_colours[eidx] = old_c
    
    # Final: count colours used
    used = sorted(set(best_colours))
    # Remap to 0..k-1
    remap = {c: i for i, c in enumerate(used)}
    result = [remap[c] for c in best_colours]
    
    return result

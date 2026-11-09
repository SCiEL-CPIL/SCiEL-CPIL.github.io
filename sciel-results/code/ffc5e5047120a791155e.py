import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    d = 7
    n = 1 << d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
    
    # Precompute for each edge, which faces it belongs to
    # A face is determined by (vertex, dim_i, dim_j) where i < j
    # Edge (x, y) with dim i belongs to faces with dim_j != i
    # Face (x, i, j): edges are (x, x^i), (x^j, x^i^j), (x, x^j), (x^i, x^i^j)
    # But we need to normalize: the face is the set of 4 edges
    
    # Build face structure: for each pair (i,j) and each "base" vertex, find the 4 edges
    # Actually, let's precompute for each edge, the list of faces (as tuples of 4 edge indices)
    
    # Alternative: for each edge, find all 4-cycles it could complete
    # Edge e = (x, y) with dim i. A 4-cycle containing e uses another dim j.
    # The other 3 edges are: (x, x^j), (x^j, y^j), (y, y^j) where y = x^i
    # Wait, let me be more careful.
    
    # Edge (x, x^i). For dim j != i, the 4-cycle is:
    # x --(i)--> x^i
    # |           |
    # (j)       (j)
    # v           v
    # x^j --(i)--> x^i^j
    # So the 4 edges are: (x,x^i), (x^i, x^i^j), (x, x^j), (x^j, x^i^j)
    
    # Let's precompute for each edge, the list of (j, other_3_edges) for all j != i
    edge_faces = [[] for _ in range(len(all_edges))]
    
    for idx, (x, y, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            # The 4 edges of this face:
            e1 = (x, y)       # dim i
            e2 = (y, yj)      # dim j
            e3 = (xj, yj)     # dim i
            e4 = (x, xj)      # dim j
            
            # Get indices
            def get_idx(a, b):
                if a > b:
                    a, b = b, a
                return edge_to_idx.get((a, b))
            
            e1_idx = idx
            e2_idx = get_idx(y, yj)
            e3_idx = get_idx(xj, yj)
            e4_idx = get_idx(x, xj)
            
            if e2_idx is not None and e3_idx is not None and e4_idx is not None:
                edge_faces[idx].append((j, e2_idx, e3_idx, e4_idx))
    
    # Greedy initial solution
    def greedy():
        chosen = set()
        out = []
        for idx in range(len(all_edges)):
            x, y, i = all_edges[idx]
            ok = True
            for (j, e2, e3, e4) in edge_faces[idx]:
                if e2 in chosen and e3 in chosen and e4 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(idx)
                out.append([x, y])
        return out, chosen
    
    # Local search: try to add edges, or swap
    def local_search(edges, chosen_set):
        # First, try to add any edge that doesn't create a C4
        improved = True
        while improved:
            improved = False
            for idx in range(len(all_edges)):
                if idx in chosen_set:
                    continue
                x, y, i = all_edges[idx]
                ok = True
                for (j, e2, e3, e4) in edge_faces[idx]:
                    if e2 in chosen_set and e3 in chosen_set and e4 in chosen_set:
                        ok = False
                        break
                if ok:
                    chosen_set.add(idx)
                    edges.append([x, y])
                    improved = True
        return edges, chosen_set
    
    # Try to swap: remove an edge and add two others? Or just do 1-for-1 swaps to enable additions.
    # Better: for each edge not in set, if it's blocked by some faces, try removing one edge from a blocking face
    # and see if we can then add our edge plus maybe another.
    
    def swap_improve(edges, chosen_set):
        improved = True
        while improved:
            improved = False
            # Try to add each non-chosen edge
            for idx in range(len(all_edges)):
                if idx in chosen_set:
                    continue
                x, y, i = all_edges[idx]
                # Find blocking faces
                blockers = []
                for (j, e2, e3, e4) in edge_faces[idx]:
                    if e2 in chosen_set and e3 in chosen_set and e4 in chosen_set:
                        blockers.append((e2, e3, e4))
                
                if not blockers:
                    chosen_set.add(idx)
                    edges.append([x, y])
                    improved = True
                    continue
                
                # For each blocker, try removing one of the 3 edges and see if we can add idx
                # and possibly gain net
                for (e2, e3, e4) in blockers:
                    for remove_idx in [e2, e3, e4]:
                        # Temporarily remove
                        chosen_set.discard(remove_idx)
                        # Try to add idx
                        ok = True
                        for (j2, a, b, c) in edge_faces[idx]:
                            if a in chosen_set and b in chosen_set and c in chosen_set:
                                ok = False
                                break
                        if ok:
                            # Check if we can also add the removed edge back? No, we want net gain
                            # We removed 1, add 1, net 0. But maybe we can add more.
                            chosen_set.add(idx)
                            edges.append([x, y])
                            # Now try to add the removed edge back or others
                            # Let's just do a quick pass to see if we can add more
                            added_more = False
                            for idx2 in range(len(all_edges)):
                                if idx2 in chosen_set:
                                    continue
                                x2, y2, i2 = all_edges[idx2]
                                ok2 = True
                                for (j3, a2, b2, c2) in edge_faces[idx2]:
                                    if a2 in chosen_set and b2 in chosen_set and c2 in chosen_set:
                                        ok2 = False
                                        break
                                if ok2:
                                    chosen_set.add(idx2)
                                    edges.append([x2, y2])
                                    added_more = True
                            # Net gain is at least 0, possibly more
                            improved = True
                            break
                        else:
                            # Restore
                            chosen_set.add(remove_idx)
                    
                    if improved:
                        break
                if improved:
                    break
        return edges, chosen_set
    
    best_edges = []
    best_chosen = set()
    best_count = 0
    
    # Multiple restarts with randomization
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        random.seed(restart * 12345 + 67890)
        chosen = set()
        edges = []
        
        # Shuffle edge order
        order = list(range(len(all_edges)))
        random.shuffle(order)
        
        for idx in order:
            x, y, i = all_edges[idx]
            ok = True
            for (j, e2, e3, e4) in edge_faces[idx]:
                if e2 in chosen and e3 in chosen and e4 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(idx)
                edges.append([x, y])
        
        # Local search
        edges, chosen = local_search(edges, chosen)
        edges, chosen = swap_improve(edges, chosen)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_chosen = chosen.copy()
        
        if best_count >= 308:
            break
    
    # Final attempt: if we haven't hit 308, try more aggressive search
    if best_count < 308 and time.time() < deadline:
        # Try starting from the best and doing more swaps
        edges = best_edges[:]
        chosen = best_chosen.copy()
        for _ in range(100):
            if time.time() > deadline:
                break
            edges, chosen = swap_improve(edges, chosen)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
                best_chosen = chosen.copy()
            if best_count >= 308:
                break
    
    return best_edges

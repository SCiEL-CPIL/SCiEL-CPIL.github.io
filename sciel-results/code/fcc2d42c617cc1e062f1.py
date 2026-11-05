import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d  # 128 vertices

    # Precompute all edges
    # Edge encoded as (u, v) where u < v
    # For each edge, also precompute the "conflict" edges:
    # An edge (u,v) along dimension i conflicts with any edge that would complete a C4 with it.
    # Actually, the constraint is: no 2D face has all 4 edges.
    # A 2D face is determined by two dimensions i, j and a vertex x where bits i and j are 0.
    # The four edges of that face are:
    #   (x, x^1<<i), (x, x^1<<j), (x^1<<i, x^(1<<i)^(1<<j)), (x^1<<j, x^(1<<i)^(1<<j))
    
    # Let's index edges by (u, v) with u < v.
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                all_edges.append(e)
                edge_index[e] = len(all_edges)

    m_total = len(all_edges)  # 448

    # For each edge, find which 2D faces it belongs to.
    # An edge (u, v) where v = u ^ (1<<i) belongs to faces (i, j) for all j != i.
    # For a given face (i, j), the edge (u, v) with v = u^(1<<i) is one of the 4 edges.
    # The other 3 edges in that face can be determined.
    
    # Let's precompute for each edge, the list of "conflict sets":
    # For each edge e, and for each face f that e is part of, the other 3 edges in f.
    # If all 4 edges are selected, we have a conflict.
    # So edge e creates a conflict if the other 3 edges in any face containing e are all selected.
    
    # Precompute: for each edge, list of (other_edge1, other_edge2, other_edge3) tuples
    # where these 3 edges + e form a 2D face.
    
    edge_conflicts = [[] for _ in range(m_total)]
    
    for x in range(n):
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                # Face with dimensions i and j, base vertex x (with bits i,j = 0)
                # Actually, we need to be careful. Let's just iterate over all pairs of dimensions
                # and all base vertices where bits i and j are 0.
                pass

    # Simpler approach: iterate over all 2D faces directly.
    # A 2D face is defined by two dimensions (i, j) and a vertex x where bit i and bit j of x are 0.
    # The 4 vertices of the face are: x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j)
    # The 4 edges are: (x, x^(1<<i)), (x, x^(1<<j)), (x^(1<<i), x^(1<<i)^(1<<j)), (x^(1<<j), x^(1<<i)^(1<<j))
    
    for i in range(d):
        for j in range(i+1, d):
            for x in range(n):
                if (x >> i) & 1 or (x >> j) & 1:
                    continue
                v0 = x
                v1 = x ^ (1 << i)
                v2 = x ^ (1 << j)
                v3 = x ^ (1 << i) ^ (1 << j)
                
                e0 = (min(v0, v1), max(v0, v1))
                e1 = (min(v0, v2), max(v0, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v3), max(v2, v3))
                
                idx0 = edge_index[e0]
                idx1 = edge_index[e1]
                idx2 = edge_index[e2]
                idx3 = edge_index[e3]
                
                # For each edge in this face, the other 3 edges form a conflict set
                edge_conflicts[idx0].append((idx1, idx2, idx3))
                edge_conflicts[idx1].append((idx0, idx2, idx3))
                edge_conflicts[idx2].append((idx0, idx1, idx3))
                edge_conflicts[idx3].append((idx0, idx1, idx2))

    # Now implement a randomized greedy + local search approach.
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        edge_order = list(range(m_total))
        rng.shuffle(edge_order)
        
        selected = [False] * m_total
        count = 0
        
        for idx in edge_order:
            # Check if adding this edge creates a conflict
            ok = True
            for (a, b, c) in edge_conflicts[idx]:
                if selected[a] and selected[b] and selected[c]:
                    ok = False
                    break
            if ok:
                selected[idx] = True
                count += 1
        
        return selected, count

    def local_search(selected, count, seed, max_iters=5000):
        rng = random.Random(seed)
        best_count = count
        best_selected = selected[:]
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add a random unselected edge
            unselected = [i for i in range(m_total) if not best_selected[i]]
            if not unselected:
                break
            
            idx = rng.choice(unselected)
            
            # Check if adding it creates a conflict
            conflicts = 0
            for (a, b, c) in edge_conflicts[idx]:
                if best_selected[a] and best_selected[b] and best_selected[c]:
                    conflicts += 1
            
            if conflicts == 0:
                best_selected[idx] = True
                best_count += 1
                continue
            
            # If it creates conflicts, try to remove one edge from each conflict set
            # to make room. This is a "repair" step.
            # For each conflict, we need to remove at least one of (a, b, c).
            # We pick the edge that appears in the most conflicts among the conflict edges.
            
            # Collect all edges that need to be removed
            to_remove = set()
            for (a, b, c) in edge_conflicts[idx]:
                if best_selected[a] and best_selected[b] and best_selected[c]:
                    # Pick the one with highest degree in the selected subgraph
                    # or just pick randomly
                    candidates = [a, b, c]
                    # Remove the one that is in the fewest other conflict sets
                    min_conf = float('inf')
                    min_idx = -1
                    for cand in candidates:
                        conf_count = len([x for x in edge_conflicts[cand] if best_selected[x[0]] and best_selected[x[1]] and best_selected[x[2]]])
                        if conf_count < min_conf:
                            min_conf = conf_count
                            min_idx = cand
                    if min_idx >= 0:
                        to_remove.add(min_idx)
            
            if len(to_remove) < conflicts:
                # Can't fix all conflicts with minimal removals, skip
                continue
            
            # Remove the edges
            for rm in to_remove:
                best_selected[rm] = False
                best_count -= 1
            
            # Add the new edge
            best_selected[idx] = True
            best_count += 1
            
            if best_count > count:
                count = best_count
                selected = best_selected[:]
        
        return selected, count

    # Start with a few random greedy runs
    best_selected = None
    best_count = 0
    
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
        selected, count = greedy_randomized(seed)
        if count > best_count:
            best_count = count
            best_selected = selected
    
    # Local search from the best
    if best_selected is not None:
        best_selected, best_count = local_search(best_selected, best_count, 42)

    # Try a few more random starts with local search
    for seed in range(20, 100):
        if time.time() - start_time > time_limit:
            break
        selected, count = greedy_randomized(seed)
        if count >= best_count - 10:  # Only do local search if close
            selected, count = local_search(selected, count, seed, max_iters=2000)
            if count > best_count:
                best_count = count
                best_selected = selected

    # Build the output list
    out = []
    for i in range(m_total):
        if best_selected[i]:
            out.append(list(all_edges[i]))
    
    return out

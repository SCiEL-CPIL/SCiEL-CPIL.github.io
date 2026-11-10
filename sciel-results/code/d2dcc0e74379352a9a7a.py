import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    d = 7
    n = 1 << d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) where dim is the bit that differs
    # u < v always
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    # Precompute for each edge, the other two edges needed to complete a C4
    # Edge (u, v, i): for each j != i, the C4 is u-v, u^(1<<j)-v^(1<<j), u-v^(1<<j), u^(1<<j)-v^(1<<j)
    # The two other edges are:
    #   (min(u, u^(1<<j)), max(u, u^(1<<j))) and (min(v, v^(1<<j)), max(v, v^(1<<j)))
    #   Wait, let me think more carefully.
    # A C4 with edges (u,v) [dim i] and (u^j, v^j) [dim i] has corners:
    #   u, v, u^(1<<j), v^(1<<j)
    # The four edges are:
    #   (u, v) - dim i
    #   (u^(1<<j), v^(1<<j)) - dim i
    #   (u, u^(1<<j)) - dim j
    #   (v, v^(1<<j)) - dim j
    # So to complete a C4, we need edges (u, u^(1<<j)) and (v, v^(1<<j)) to also be present.
    
    # For each edge index, store list of (other_edge1_idx, other_edge2_idx) pairs
    # where both other edges being present would complete a C4
    
    # Build edge lookup: (u, v) -> index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, find the C4-completing pairs
    c4_pairs = [[] for _ in range(len(all_edges))]
    for idx, (u, v, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            # The other two edges in the C4
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Edge (u, u2) with dim j
            e1 = (min(u, u2), max(u, u2))
            # Edge (v, v2) with dim j
            e2 = (min(v, v2), max(v, v2))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            if idx1 is not None and idx2 is not None:
                c4_pairs[idx].append((idx1, idx2))
    
    num_edges = len(all_edges)
    
    # Greedy algorithm with randomization
    best_count = 0
    best_edges = []
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Create a random permutation of edges
        edge_indices = list(range(num_edges))
        rng.shuffle(edge_indices)
        
        chosen = [False] * num_edges
        count = 0
        out = []
        
        for idx in edge_indices:
            # Check if adding this edge completes any C4
            ok = True
            for (i1, i2) in c4_pairs[idx]:
                if chosen[i1] and chosen[i2]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
                u, v, i = all_edges[idx]
                out.append([u, v])
        
        if count > best_count:
            best_count = count
            best_edges = out[:]
        
        # If we've hit a good number, we can stop early
        if best_count >= 304:
            break
    
    return best_edges

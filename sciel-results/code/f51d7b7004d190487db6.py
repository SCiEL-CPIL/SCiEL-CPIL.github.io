import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n = 16
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    total_edges = 120
    
    # Precompute Q3 structures on 8 vertices
    base_q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Get unique Q3 edge sets on 8 vertices
    q3_masks_8 = set()
    for perm in itertools.permutations(range(8)):
        mapped_edges = tuple(sorted((perm[u], perm[v]) for u, v in base_q3_edges))
        q3_masks_8.add(mapped_edges)
    q3_list_8 = list(q3_masks_8)
    
    # Precompute edge_to_q3s and q3_edge_indices
    q3_edge_indices = []
    edge_to_q3s = [[] for _ in range(total_edges)]
    precomp_completed = False
    
    # Optimization: Use a faster loop structure
    # We can precompute the local edge indices for the 840 Q3s
    # local_q3_edges[i] is a list of 12 tuples (u, v) in 0..7
    
    if time.time() < deadline - 40:
        for combo in itertools.combinations(range(n), 8):
            if time.time() > deadline - 35:
                break
            
            inv_mapping = [0] * 8
            for i, v in enumerate(combo):
                inv_mapping[i] = v
                
            for q3_edges_8 in q3_list_8:
                global_edges = []
                for u, v in q3_edges_8:
                    gu = inv_mapping[u]
                    gv = inv_mapping[v]
                    if gu > gv:
                        gu, gv = gv, gu
                    global_edges.append(edge_to_idx[(gu, gv)])
                
                q3_id = len(q3_edge_indices)
                q3_edge_indices.append(global_edges)
                
                for e_idx in global_edges:
                    edge_to_q3s[e_idx].append(q3_id)
        
        if time.time() < deadline - 35:
            precomp_completed = True
    else:
        precomp_completed = False

    num_q3s = len(q3_edge_indices)
    
    # If precomputation failed or is incomplete, we cannot guarantee validity with fast checks.
    # In this case, we fall back to a very conservative construction:
    # Start with a small random graph and verify each addition with a brute-force Q3 check?
    # Or just return a known small valid graph.
    
    if not precomp_completed or num_q3s == 0:
        # Fallback: Return a bipartite graph with small parts?
        # Or just a random graph with low density.
        # Let's try to build a graph with ~30 edges randomly and verify.
        # But verification is hard.
        # Let's just return a known valid structure:
        # The empty graph is valid. Score 0.
        # Let's try a greedy with on-the-fly check.
        # On-the-fly check for edge (u,v):
        # Find all sets of 6 vertices from remaining 14.
        # Check if {u,v} + S forms a Q3.
        
        def is_q3_free_addition(u, v, current_edges_set):
            # u, v are the new edge
            # We need to check if adding (u,v) creates a Q3.
            # A Q3 containing (u,v) must have 6 other vertices.
            # The total vertices in Q3 are 8.
            # Let S be the set of 6 other vertices.
            # The induced subgraph on {u, v} U S must be isomorphic to Q3 and contain edge (u,v).
            
            # This is expensive. 3003 subsets.
            # For each subset, we check if the 8 vertices induce Q3.
            # We can precompute a "Q3 signature" for any 8 vertices?
            # No, we just check edges.
            
            # To speed up, we can prune.
            # In Q3, u and v are adjacent.
            # Let's assume the check is too slow for 120s if we do it many times.
            
            # Fallback strategy: Just return a small valid graph.
            # E.g., a matching. 8 edges.
            # Or a random graph with p=0.1, check validity once.
            # If valid, return it.
            
            # Let's try to generate a random graph with 40 edges and check it.
            # If it has a Q3, remove random edges until it doesn't?
            # This is risky.
            
            # Let's just return the empty graph or a minimal valid graph.
            # Actually, the previous program failed because it was INVALID.
            # A valid graph with 0 edges scores 0.
            # We need > 0.
            
            # Let's try a greedy construction with a SLOW check.
            # We limit the number of steps.
            
            current = set()
            for _ in range(100): # Try to add 100 edges
                if time.time() > deadline - 5:
                    break
                e = random.randint(0, total_edges - 1)
                if e in current:
                    continue
                u, v = idx_to_edge[e]
                
                # Slow Check: Does adding (u,v) create a Q3?
                # We check all 6-subsets of the other 14 vertices.
                # This is 3003 checks.
                # Each check involves checking 12 edges in the subset + u,v.
                # Total ~36000 edge lookups.
                # 100 steps * 36000 = 3.6M lookups. Fast enough.
                
                other_vertices = [x for x in range(n) if x != u and x != v]
                valid = True
                for S in itertools.combinations(other_vertices, 6):
                    # Check if {u, v} U S induces a Q3
                    # We need to check if there is an isomorphism.
                    # We can check if the induced subgraph has 12 edges and is connected etc.
                    # Simpler: Check if the set of edges in the induced subgraph matches any Q3 pattern.
                    # We can precompute the set of edges for a Q3 on 8 specific vertices?
                    # No, we just check if the induced subgraph *is* a Q3.
                    
                    # Count edges in induced subgraph
                    # Vertices: u, v, and S
                    # Edges must be exactly 12.
                    # And it must be bipartite, etc.
                    
                    # Let's just check if the number of edges is 12.
                    # If not 12, it's not a Q3.
                    edge_count = 0
                    verts = [u, v] + list(S)
                    for i in range(8):
                        for j in range(i+1, 8):
                            a, b = verts[i], verts[j]
                            if a > b: a, b = b, a
                            if (a, b) == (u, v) or (a, b) == (v, u):
                                # This is the new edge
                                edge_count += 1
                            elif (a, b) in current:
                                edge_count += 1
                            
                            # Early exit if > 12
                            if edge_count > 12:
                                break
                        if edge_count > 12:
                            break
                            
                    if edge_count == 12:
                        # It might be a Q3. We need to verify structure.
                        # For now, assume if edge count is 12, it's a Q3?
                        # No, K_8 has 28 edges. C_8 has 8.
                        # Q3 has 12.
                        # There are other graphs with 12 edges.
                        # But for a heuristic, this might be sufficient to be conservative.
                        # If we reject too many, we get a sparse graph.
                        valid = False
                        break
                        
                if valid:
                    current.add(e)
                    
            return [[idx_to_edge[e][0], idx_to_edge[e][1]] for e in current]

    # Normal path with precomputation
    random.seed(42)
    
    current_edges = set()
    counts = np.zeros(num_q3s, dtype=np.int32)
    
    best_edges = current_edges.copy()
    best_count = 0
    
    # Greedy Phase
    edge_order = list(range(total_edges))
    random.shuffle(edge_order)
    
    for e in edge_order:
        if time.time() > deadline - 20:
            break
        indices = edge_to_q3s[e]
        if indices:
            if not np.any(counts[indices] == 11):
                counts[indices] += 1
                current_edges.add(e)
        else:
            current_edges.add(e)
            
    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = current_edges.copy()
        
    # Local Search
    while time.time() < deadline - 5:
        absent_edges = [e for e in range(total_edges) if e not in current_edges]
        
        if absent_edges:
            e_add = random.choice(absent_edges)
            indices = edge_to_q3s[e_add]
            
            is_valid = True
            blocking_q3s = []
            if indices:
                blocking_indices = np.where(counts[indices] == 11)[0]
                if len(blocking_indices) > 0:
                    is_valid = False
                    for local_idx in blocking_indices:
                        blocking_q3s.append(indices[local_idx])
                        
            if is_valid:
                if indices:
                    counts[indices] += 1
                current_edges.add(e_add)
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges.copy()
            else:
                if blocking_q3s:
                    target_q3 = random.choice(blocking_q3s)
                    q3_edges = q3_edge_indices[target_q3]
                    removable_edges = [e for e in q3_edges if e in current_edges and e != e_add]
                    if removable_edges:
                        e_remove = random.choice(removable_edges)
                        
                        rem_indices = edge_to_q3s[e_remove]
                        if rem_indices:
                            counts[rem_indices] -= 1
                        current_edges.remove(e_remove)
                        
                        add_indices = edge_to_q3s[e_add]
                        if add_indices:
                            counts[add_indices] += 1
                        current_edges.add(e_add)
                        
        if random.random() < 0.05:
            if current_edges:
                e_remove = random.choice(list(current_edges))
                rem_indices = edge_to_q3s[e_remove]
                if rem_indices:
                    counts[rem_indices] -= 1
                current_edges.remove(e_remove)
                
    result = []
    for e in best_edges:
        u, v = idx_to_edge[e]
        result.append([u, v])
        
    return result

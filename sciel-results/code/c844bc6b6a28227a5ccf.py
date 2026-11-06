import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2 # 448

    # 1. Precompute edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # 2. Precompute faces (squares)
    # A face is defined by a starting vertex u and two distinct bits i, j
    # The 4 edges of the face are:
    # u - u^(1<<i)
    # u^(1<<i) - u^(1<<i)^(1<<j)
    # u^(1<<i)^(1<<j) - u^(1<<j)
    # u^(1<<j) - u
    faces = []
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << i) ^ (1 << j)
                v4 = u ^ (1 << j)
                
                # Get edge indices for the 4 edges
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v3), max(v2, v3))
                e3 = (min(v3, v4), max(v3, v4))
                e4 = (min(v4, v1), max(v4, v1))
                
                idx1 = edge_to_index[e1]
                idx2 = edge_to_index[e2]
                idx3 = edge_to_index[e3]
                idx4 = edge_to_index[e4]
                
                faces.append((idx1, idx2, idx3, idx4))

    # 3. Build edge -> faces mapping
    # For each edge, list of face indices it belongs to
    edge_to_faces = [[] for _ in range(total_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)

    # 4. Greedy Hitting Set Algorithm
    # We want to remove a minimum number of edges such that every face has at least one removed edge.
    # This is equivalent to Set Cover where sets are edges and elements are faces.
    
    def greedy_hitting_set(seed):
        rng = random.Random(seed)
        
        # active_faces_count[e] = number of active faces covered by edge e
        # Initially, all faces are active.
        active_faces_count = [len(edge_to_faces[e]) for e in range(total_edges)]
        
        # active_faces[f] = number of edges in face f that are still "available" (not in hitting set)
        # Actually, we need to know if a face is covered.
        # Let's maintain:
        # 1. A boolean array `face_hit` of size len(faces), initially False.
        # 2. For each edge, the list of faces it covers.
        # 3. For each edge, the count of *active* faces it covers.
        
        # To efficiently update counts when an edge is selected:
        # When we select edge e:
        # - For each face f in edge_to_faces[e]:
        #   - If face f is not hit yet:
        #     - Mark face f as hit.
        #     - For each other edge e' in face f:
        #       - Decrement active_faces_count[e'] by 1.
        
        face_hit = [False] * len(faces)
        hitting_set = []
        remaining_faces = len(faces)
        
        # We need to be able to pick the edge with max active_faces_count quickly.
        # Since total_edges is small (448), we can just scan or use a simple heap.
        # Scanning 448 items is very fast.
        
        available_edges = list(range(total_edges))
        
        while remaining_faces > 0:
            # Find edge with max active_faces_count among available edges
            best_edge = -1
            max_count = -1
            # Random tie-breaking to diversify
            for e in available_edges:
                if active_faces_count[e] > max_count:
                    max_count = active_faces_count[e]
                    best_edge = e
                elif active_faces_count[e] == max_count and max_count > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_edge = e
            
            if best_edge == -1 or max_count == 0:
                # Should not happen if remaining_faces > 0 and graph is connected
                break
                
            # Select best_edge
            hitting_set.append(best_edge)
            available_edges.remove(best_edge)
            
            # Update counts
            for f_idx in edge_to_faces[best_edge]:
                if not face_hit[f_idx]:
                    face_hit[f_idx] = True
                    remaining_faces -= 1
                    
                    # Decrement counts for other edges in this face
                    face_edges = faces[f_idx]
                    for e_prime in face_edges:
                        if e_prime != best_edge:
                            active_faces_count[e_prime] -= 1
                            
        return hitting_set

    best_hitting_set = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    seed = 0
    # Run multiple greedy passes with different seeds
    # Each pass is very fast, so we can run many.
    while time.time() < deadline:
        hs = greedy_hitting_set(seed)
        if len(hs) < best_size:
            best_size = len(hs)
            best_hitting_set = hs
            
        # If we reach a very good solution, we might stop early or continue to try for better
        # Theoretical lower bound for hitting set in Q7?
        # Max edges = 308 => Min hitting set = 448 - 308 = 140.
        # Best known max edges = 304 => Min hitting set = 144.
        if best_size <= 140:
            break
            
        seed += 1

    if best_hitting_set is None:
        # Fallback if no time was spent (should not happen)
        best_hitting_set = greedy_hitting_set(0)
        
    # Construct result: all edges NOT in hitting_set
    hitting_set_set = set(best_hitting_set)
    result = []
    for i in range(total_edges):
        if i not in hitting_set_set:
            result.append(all_edges[i])
            
    return result

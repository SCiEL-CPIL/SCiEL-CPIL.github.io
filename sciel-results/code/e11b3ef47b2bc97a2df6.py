import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    
    # 1. Generate all edges
    all_edges_list = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_list.append((x, y))
    
    # 2. Generate all faces and map edges to faces
    # Face is defined by (i, j, fixed_bits)
    # We use a list of lists for faces: faces[k] = [e1, e2, e3, e4]
    # edge_faces: dict edge -> list of face indices
    faces = []
    edge_faces = {}
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            other_bits = [k for k in range(d) if k != i and k != j]
            
            for val in range(1 << len(other_bits)):
                base = 0
                for idx, bit in enumerate(other_bits):
                    if (val >> idx) & 1:
                        base |= (1 << bit)
                
                v0 = base
                v1 = base | mask_i
                v2 = base | mask_j
                v3 = base | mask_i | mask_j
                
                e0 = (min(v0, v1), max(v0, v1))
                e1 = (min(v1, v3), max(v1, v3))
                e2 = (min(v3, v2), max(v3, v2))
                e3 = (min(v2, v0), max(v2, v0))
                
                face_edges = [e0, e1, e2, e3]
                face_idx = len(faces)
                faces.append(face_edges)
                
                for e in face_edges:
                    if e not in edge_faces:
                        edge_faces[e] = []
                    edge_faces[e].append(face_idx)
                    
    num_faces = len(faces)
    
    # 3. Greedy Deletion Algorithm
    # We run multiple iterations with random tie-breaking to explore the solution space.
    best_set = set()
    best_size = 0
    start_time = time.time()
    time_limit = 100.0
    
    for run in range(50):
        if time.time() - start_time > time_limit:
            break
            
        current_edges = set(all_edges_list)
        # counts[f] = number of active edges in face f
        counts = [4] * num_faces
        
        # We need to efficiently find bad faces.
        # Since we re-scan, we just check counts.
        
        while True:
            # Find all bad faces
            bad_face_ids = [f for f in range(num_faces) if counts[f] == 4]
            
            if not bad_face_ids:
                break
                
            # For each edge in these bad faces, count how many bad faces it belongs to.
            # Only consider edges that are currently active.
            edge_bad_count = {}
            for fid in bad_face_ids:
                for e in faces[fid]:
                    if e in current_edges:
                        edge_bad_count[e] = edge_bad_count.get(e, 0) + 1
            
            if not edge_bad_count:
                # Should not happen if bad_face_ids is not empty
                break
                
            # Find the edge with the maximum bad face count
            max_count = max(edge_bad_count.values())
            candidates = [e for e, c in edge_bad_count.items() if c == max_count]
            
            # Random tie-breaking
            random.shuffle(candidates)
            e_to_remove = candidates[0]
            
            # Remove edge
            current_edges.remove(e_to_remove)
            
            # Update counts for faces containing this edge
            for fid in edge_faces[e_to_remove]:
                counts[fid] -= 1
                
        if len(current_edges) > best_size:
            best_size = len(current_edges)
            best_set = current_edges
            
    return [[u, v] for u, v in best_set]

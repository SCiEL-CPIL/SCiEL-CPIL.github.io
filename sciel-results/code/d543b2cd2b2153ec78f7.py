import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    num_edges = d * (n // 2) # 7 * 64 = 448
    
    # Precompute edges
    edges = []
    edge_to_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_index[(u, v)] = idx
                
    # Precompute faces (4-cycles)
    # A face is defined by 2 dimensions i, j and 5 fixed bits.
    # Vertices: x, x^i, x^j, x^i^j (where x has 0 in bits i, j)
    faces = []
    edge_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            # Iterate over all combinations of fixed bits
            # There are 2^(d-2) such faces
            mask_i = 1 << i
            mask_j = 1 << j
            # Generate all vertices with 0 in i and j
            # We can just iterate u from 0 to n-1 and check if u has 0 in i and j
            # Then the face is u, u^i, u^j, u^i^j
            for u in range(n):
                if (u & mask_i) == 0 and (u & mask_j) == 0:
                    v1 = u ^ mask_i
                    v2 = u ^ mask_j
                    v3 = u ^ mask_i ^ mask_j
                    
                    # Edges of this face
                    e1 = edge_to_index.get((min(u, v1), max(u, v1)))
                    e2 = edge_to_index.get((min(v1, v3), max(v1, v3)))
                    e3 = edge_to_index.get((min(v3, v2), max(v3, v2)))
                    e4 = edge_to_index.get((min(v2, u), max(v2, u)))
                    
                    # They should all exist
                    face_edges = [e1, e2, e3, e4]
                    face_idx = len(faces)
                    faces.append(face_edges)
                    
                    for e in face_edges:
                        edge_faces[e].append(face_idx)

    num_faces = len(faces)
    
    best_size = 0
    best_edges = []
    
    while time.time() < deadline:
        # State
        edge_sel = np.zeros(num_edges, dtype=bool)
        face_count = np.zeros(num_faces, dtype=int)
        
        # Random order
        order = list(range(num_edges))
        random.shuffle(order)
        
        current_size = 0
        
        # Greedy Add
        for e in order:
            # Check if adding e is valid
            # e is valid if for all faces f containing e, face_count[f] < 3
            valid = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    valid = False
                    break
            
            if valid:
                edge_sel[e] = True
                current_size += 1
                for f in edge_faces[e]:
                    face_count[f] += 1
        
        if current_size > best_size:
            best_size = current_size
            best_edges = [edges[i] for i in range(num_edges) if edge_sel[i]]
            
        # Kick and Repair
        # Try to improve current solution by kicking
        # We do a few kicks if time permits
        for _ in range(5): # 5 kick attempts
            if time.time() > deadline - 5.0:
                break
                
            # Remove K random edges
            K = random.randint(2, 10)
            selected_indices = [i for i in range(num_edges) if edge_sel[i]]
            if len(selected_indices) < K:
                break
            to_remove = random.sample(selected_indices, K)
            
            for e in to_remove:
                edge_sel[e] = False
                current_size -= 1
                for f in edge_faces[e]:
                    face_count[f] -= 1
            
            # Repair: Try to add edges in random order
            # We can just iterate through all edges, or just unselected ones
            # Iterating through all is safer and simple
            repair_order = list(range(num_edges))
            random.shuffle(repair_order)
            
            for e in repair_order:
                if edge_sel[e]:
                    continue
                valid = True
                for f in edge_faces[e]:
                    if face_count[f] == 3:
                        valid = False
                        break
                if valid:
                    edge_sel[e] = True
                    current_size += 1
                    for f in edge_faces[e]:
                        face_count[f] += 1
            
            if current_size > best_size:
                best_size = current_size
                best_edges = [edges[i] for i in range(num_edges) if edge_sel[i]]
                
            # If we didn't improve, maybe break early? No, keep trying.
            
    # Return list of [u, v]
    return [[u, v] for u, v in best_edges]

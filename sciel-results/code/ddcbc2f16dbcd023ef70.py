import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    deadline = time.time() + 110
    
    # Precomputation
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
    
    num_edges = len(edges)
    
    faces = []
    edge_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            for u in range(n_vertices):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v2, v4), max(v2, v4))]
                    e3 = edge_index[(min(v4, v3), max(v4, v3))]
                    e4 = edge_index[(min(v3, v1), max(v3, v1))]
                    
                    face_id = len(faces)
                    faces.append([e1, e2, e3, e4])
                    edge_faces[e1].append(face_id)
                    edge_faces[e2].append(face_id)
                    edge_faces[e3].append(face_id)
                    edge_faces[e4].append(face_id)
    
    num_faces = len(faces)
    
    best_count = 0
    best_set = set()
    rng = random.Random()
    
    def greedy_init():
        chosen = set()
        face_counts = [0] * num_faces
        edge_indices = list(range(num_edges))
        rng.shuffle(edge_indices)
        
        for e_idx in edge_indices:
            valid = True
            for f_id in edge_faces[e_idx]:
                if face_counts[f_id] == 3:
                    valid = False
                    break
            if valid:
                chosen.add(e_idx)
                for f_id in edge_faces[e_idx]:
                    face_counts[f_id] += 1
        return chosen, face_counts
    
    while time.time() < deadline:
        chosen, face_counts = greedy_init()
        current_count = len(chosen)
        
        if current_count > best_count:
            best_count = current_count
            best_set = set(chosen)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            unselected = [e for e in range(num_edges) if e not in chosen]
            rng.shuffle(unselected)
            
            for e in unselected:
                valid = True
                for f_id in edge_faces[e]:
                    if face_counts[f_id] == 3:
                        valid = False
                        break
                if valid:
                    chosen.add(e)
                    for f_id in edge_faces[e]:
                        face_counts[f_id] += 1
                    current_count += 1
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_set = set(chosen)
            
            if not improved and chosen:
                # Kick
                to_remove = rng.sample(list(chosen), min(3, len(chosen)))
                for e_to_remove in to_remove:
                    chosen.remove(e_to_remove)
                    for f_id in edge_faces[e_to_remove]:
                        face_counts[f_id] -= 1
                    current_count -= 1
                improved = True
        
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])
    return result

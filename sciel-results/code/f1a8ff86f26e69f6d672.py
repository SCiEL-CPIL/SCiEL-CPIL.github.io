import random
import time
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = n_vertices * d // 2  # 448
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
                
    # 2. Precompute faces
    faces = []
    seen_faces = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_index[(min(v1, v2), max(v1, v2))]
                e2 = edge_index[(min(v3, v4), max(v3, v4))]
                e3 = edge_index[(min(v1, v3), max(v1, v3))]
                e4 = edge_index[(min(v2, v4), max(v2, v4))]
                
                face_key = tuple(sorted([e1, e2, e3, e4]))
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    faces.append(face_key)
    
    n_faces = len(faces)
    
    # Precompute edge_faces mapping
    edge_faces = [[] for _ in range(n_edges_total)]
    for f_idx, f_edges in enumerate(faces):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
            
    def solve_hitting_set(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = set(range(n_faces))
        edge_count = [0] * n_edges_total
        for f_idx in uncovered:
            for e_idx in faces[f_idx]:
                edge_count[e_idx] += 1
        
        removed_edges = set()
        
        while uncovered:
            if time.time() - start_time > time_limit - 5:
                break
            
            max_count = -1
            candidates = []
            for e in range(n_edges_total):
                if edge_count[e] > max_count:
                    max_count = edge_count[e]
                    candidates = [e]
                elif edge_count[e] == max_count:
                    candidates.append(e)
            
            if not candidates:
                break
                
            best_edge = random.choice(candidates)
            removed_edges.add(best_edge)
            
            for f_idx in edge_faces[best_edge]:
                if f_idx in uncovered:
                    uncovered.remove(f_idx)
                    for e_idx in faces[f_idx]:
                        if e_idx != best_edge:
                            edge_count[e_idx] -= 1
            edge_count[best_edge] = 0
        
        return removed_edges
    
    def prune_removed(removed_edges):
        face_cover_count = [0] * n_faces
        for e in removed_edges:
            for f_idx in edge_faces[e]:
                face_cover_count[f_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for e in list(removed_edges):
                is_redundant = True
                for f_idx in edge_faces[e]:
                    if face_cover_count[f_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    removed_edges.remove(e)
                    for f_idx in edge_faces[e]:
                        face_cover_count[f_idx] -= 1
                    changed = True
        
        return removed_edges
    
    best_removed_count = n_edges_total
    best_removed_set = None
    
    for attempt in range(200):
        if time.time() - start_time > time_limit - 10:
            break
            
        removed = solve_hitting_set(seed=attempt)
        removed = prune_removed(removed)
        
        count = len(removed)
        if count < best_removed_count:
            best_removed_count = count
            best_removed_set = removed
            
        if best_removed_count <= 140:
            break

    final_edges = []
    for idx, (u, v, i) in enumerate(edges):
        if idx not in best_removed_set:
            final_edges.append([u, v])
            
    return final_edges

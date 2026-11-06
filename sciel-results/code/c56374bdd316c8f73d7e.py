import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))
    
    # 1. Precompute edges and faces
    edges = []
    edge_map = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
                
    faces = []
    face_set = set()
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_map[(min(u, v1), max(u, v1))]
                e2 = edge_map[(min(v2, v3), max(v2, v3))]
                e3 = edge_map[(min(u, v2), max(u, v2))]
                e4 = edge_map[(min(v1, v3), max(v1, v3))]
                
                face_tuple = tuple(sorted([e1, e2, e3, e4]))
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append(face_tuple)
                    
    edge_faces = [[] for _ in range(len(edges))]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)

    # 2. Greedy Initialization
    in_set = [False] * len(edges)
    face_count = [0] * len(faces)
    
    order = list(range(len(edges)))
    random.shuffle(order)
    
    for e_idx in order:
        can_add = True
        for f_idx in edge_faces[e_idx]:
            if face_count[f_idx] == 3:
                can_add = False
                break
        if can_add:
            in_set[e_idx] = True
            for f_idx in edge_faces[e_idx]:
                face_count[f_idx] += 1
                
    current_count = sum(in_set)
    
    # 3. Local Search
    deadline = time.time() + 110.0
    
    improved = True
    # Limit number of passes to avoid infinite loops if randomization fails?
    # No, just use time.
    
    while improved and time.time() < deadline:
        improved = False
        search_order = list(range(len(edges)))
        random.shuffle(search_order)
        
        for e_remove in search_order:
            if time.time() > deadline:
                break
            
            if not in_set[e_remove]:
                continue
                
            # 1. Remove e_remove
            in_set[e_remove] = False
            affected_faces = edge_faces[e_remove]
            for f_idx in affected_faces:
                face_count[f_idx] -= 1
                
            # 2. Find candidates
            candidates = set()
            for f_idx in affected_faces:
                for e_idx in faces[f_idx]:
                    if e_idx != e_remove and not in_set[e_idx]:
                        candidates.add(e_idx)
                        
            # 3. Determine addable
            addable = []
            for r in candidates:
                ok = True
                for f_idx in edge_faces[r]:
                    if face_count[f_idx] >= 3:
                        ok = False
                        break
                if ok:
                    addable.append(r)
                    
            if len(addable) < 2:
                # Revert
                in_set[e_remove] = True
                for f_idx in affected_faces:
                    face_count[f_idx] += 1
                continue
            
            # 4. Simulate adds
            temp_face_count = face_count.copy()
            successful_adds = []
            
            # Sort addable to prioritize? 
            # Maybe edges with fewer faces? No, all edges have 21 faces.
            # Random order is fine.
            random.shuffle(addable)
            
            for r in addable:
                ok = True
                for f_idx in edge_faces[r]:
                    if temp_face_count[f_idx] >= 3:
                        ok = False
                        break
                if ok:
                    successful_adds.append(r)
                    for f_idx in edge_faces[r]:
                        temp_face_count[f_idx] += 1
                        
            if len(successful_adds) > 1:
                # Commit
                for r in successful_adds:
                    in_set[r] = True
                    for f_idx in edge_faces[r]:
                        face_count[f_idx] += 1
                        
                current_count += len(successful_adds) - 1
                improved = True
            else:
                # Revert
                in_set[e_remove] = True
                for f_idx in affected_faces:
                    face_count[f_idx] += 1

    # 5. Output
    out = []
    for i, (u, v) in enumerate(edges):
        if in_set[i]:
            out.append([u, v])
            
    return out

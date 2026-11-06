import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    # Precompute face masks for each edge (big int)
    face_mask = [0] * n_edges
    for e in range(n_edges):
        mask = 0
        for f in edge_faces[e]:
            mask |= (1 << f)
        face_mask[e] = mask
        
    # Precompute set of edges for each face for O(1) lookup
    face_edge_set = [set(face_edges[f]) for f in range(n_faces)]

    start_time = time.time()
    deadline = start_time + 140

    def greedy_construct(randomized=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        selected = [False] * n_edges
        face_count = [0] * n_faces
        uncovered = set(range(n_faces))
        
        while uncovered:
            best_e = -1
            best_cov = 0
            candidates = range(n_edges)
            if randomized:
                candidates = list(range(n_edges))
                random.shuffle(candidates)
            
            for e in candidates:
                if selected[e]:
                    continue
                cov = 0
                for f in edge_faces[e]:
                    if f in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_e = e
            
            if best_e == -1:
                break
            
            selected[best_e] = True
            for f in edge_faces[best_e]:
                face_count[f] += 1
                if face_count[f] > 0:
                    uncovered.discard(f)
        
        return selected, face_count

    def try_remove_edges(selected, face_count):
        removed_any = True
        while removed_any:
            removed_any = False
            for e in range(n_edges):
                if not selected[e]:
                    continue
                can_remove = True
                for f in edge_faces[e]:
                    if face_count[f] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    selected[e] = False
                    for f in edge_faces[e]:
                        face_count[f] -= 1
                    removed_any = True
        return removed_any

    def local_search_swap(selected, face_count, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            r_edges = [e for e in range(n_edges) if selected[e]]
            not_r_edges = [e for e in range(n_edges) if not selected[e]]
            
            if not not_r_edges:
                break
                
            random.shuffle(r_edges)
            random.shuffle(not_r_edges)
            
            n_r = len(r_edges)
            for i in range(n_r):
                if time.time() > time_limit:
                    break
                e1 = r_edges[i]
                for j in range(i + 1, n_r):
                    if time.time() > time_limit:
                        break
                    e2 = r_edges[j]
                    
                    mask_crit = 0
                    faces_to_check = set(edge_faces[e1]) | set(edge_faces[e2])
                    for f in faces_to_check:
                        delta = 0
                        if e1 in face_edge_set[f]:
                            delta += 1
                        if e2 in face_edge_set[f]:
                            delta += 1
                        if face_count[f] - delta < 1:
                            mask_crit |= (1 << f)
                    
                    if mask_crit == 0:
                        selected[e1] = False
                        selected[e2] = False
                        for f in edge_faces[e1]:
                            face_count[f] -= 1
                        for f in edge_faces[e2]:
                            face_count[f] -= 1
                        improved = True
                        break
                    else:
                        found = False
                        for e3 in not_r_edges:
                            if (mask_crit & ~face_mask[e3]) == 0:
                                selected[e1] = False
                                selected[e2] = False
                                for f in edge_faces[e1]:
                                    face_count[f] -= 1
                                for f in edge_faces[e2]:
                                    face_count[f] -= 1
                                
                                selected[e3] = True
                                for f in edge_faces[e3]:
                                    face_count[f] += 1
                                
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if improved:
                    break
        return improved

    selected, face_count = greedy_construct(randomized=False)
    try_remove_edges(selected, face_count)
    best_count = sum(selected)
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    local_search_swap(selected, face_count, deadline)
    try_remove_edges(selected, face_count)
    current_count = sum(selected)
    if current_count < best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
        
    for restart in range(100):
        if time.time() > deadline - 2:
            break
        
        seed = random.randint(0, 2**31 - 1)
        selected, face_count = greedy_construct(randomized=True, seed=seed)
        try_remove_edges(selected, face_count)
        local_search_swap(selected, face_count, deadline)
        try_remove_edges(selected, face_count)
        
        current_count = sum(selected)
        if current_count < best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
    
    result = []
    for i, (u, v) in enumerate(edges):
        if not best_selected[i]:
            result.append([u, v])
            
    return result

import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d  # 128
    num_edges_total = n_vertices * d // 2  # 448
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
                
    num_edges = len(all_edges) # 448

    # Precompute all faces (C4s) with deduplication
    unique_faces_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                e1 = (min(u, u ^ (1<<i)), max(u, u ^ (1<<i)))
                e2 = (min(u, u ^ (1<<j)), max(u, u ^ (1<<j)))
                e3 = (min(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<i), u ^ (1<<i) ^ (1<<j)))
                e4 = (min(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)), max(u ^ (1<<j), u ^ (1<<i) ^ (1<<j)))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_key = tuple(sorted([idx1, idx2, idx3, idx4]))
                unique_faces_set.add(face_key)
                
    faces = list(unique_faces_set)
    num_faces = len(faces) # Should be 672
    
    # Precompute for each edge, which faces it covers
    edge_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
            
    # Precompute for each face, the 4 edges
    face_edges = [list(f) for f in faces]
    
    # Convert edge_faces to numpy arrays for faster iteration
    edge_faces_arr = [np.array(ef, dtype=np.int32) for ef in edge_faces]
    
    def greedy_hit_set(uncovered_faces, rng):
        """
        Greedily add edges to cover all uncovered faces.
        uncovered_faces: set of face indices to cover.
        Returns a set of edge indices.
        """
        R = set()
        uncovered = set(uncovered_faces)
        
        while uncovered:
            best_edge = -1
            best_count = 0
            
            for e in range(num_edges):
                if e in R:
                    continue
                ef = edge_faces_arr[e]
                count = sum(1 for f in ef if f in uncovered)
                if count > best_count:
                    best_count = count
                    best_edge = e
                    if best_count >= len(uncovered):
                        break
            
            if best_edge == -1:
                break
                
            R.add(best_edge)
            for f in edge_faces_arr[best_edge]:
                if f in uncovered:
                    uncovered.discard(f)
                    
        return R

    def local_search_hit_set(R, deadline, rng):
        """
        Local search to minimize |R| using fast coverage counting.
        """
        R = set(R)
        cover_count = np.zeros(num_faces, dtype=np.int32)
        for e in R:
            for f in edge_faces_arr[e]:
                cover_count[f] += 1
        
        while time.time() < deadline:
            improved = False
            R_list = list(R)
            rng.shuffle(R_list)
            
            for e_remove in R_list:
                if time.time() > deadline:
                    break
                
                # Try deleting e_remove
                valid = True
                for f in edge_faces_arr[e_remove]:
                    if cover_count[f] == 1:
                        valid = False
                        break
                
                if valid:
                    R.discard(e_remove)
                    for f in edge_faces_arr[e_remove]:
                        cover_count[f] -= 1
                    improved = True
                    break
                else:
                    # Kick: remove e_remove, greedily refill
                    if rng.random() < 0.15:
                        R_temp = set(R)
                        R_temp.discard(e_remove)
                        
                        uncovered = set()
                        for f in edge_faces_arr[e_remove]:
                            if cover_count[f] == 1:
                                uncovered.add(f)
                        
                        if uncovered:
                            R_add = greedy_hit_set(uncovered, rng)
                            R_new = R_temp | R_add
                            
                            if len(R_new) < len(R):
                                R = R_new
                                cover_count = np.zeros(num_faces, dtype=np.int32)
                                for e in R:
                                    for f in edge_faces_arr[e]:
                                        cover_count[f] += 1
                                improved = True
                                break
                            elif len(R_new) == len(R) and rng.random() < 0.2:
                                R = R_new
                                cover_count = np.zeros(num_faces, dtype=np.int32)
                                for e in R:
                                    for f in edge_faces_arr[e]:
                                        cover_count[f] += 1
                                improved = True
                                break
            
            if not improved:
                # Try larger kicks
                kick_size = rng.choice([2, 3, 4, 5])
                if len(R) >= kick_size:
                    to_remove = rng.sample(list(R), kick_size)
                    R_temp = set(R)
                    for e in to_remove:
                        R_temp.discard(e)
                    
                    cover_count_temp = np.zeros(num_faces, dtype=np.int32)
                    for e in R_temp:
                        for f in edge_faces_arr[e]:
                            cover_count_temp[f] += 1
                    
                    uncovered = set(f for f in range(num_faces) if cover_count_temp[f] == 0)
                    
                    if uncovered:
                        R_add = greedy_hit_set(uncovered, rng)
                        R_new = R_temp | R_add
                        
                        if len(R_new) <= len(R):
                            R = R_new
                            cover_count = np.zeros(num_faces, dtype=np.int32)
                            for e in R:
                                for f in edge_faces_arr[e]:
                                    cover_count[f] += 1
                            improved = True
            
            if not improved and time.time() < deadline:
                # Periodic full restart within local search
                if rng.random() < 0.08:
                    R = greedy_hit_set(set(range(num_faces)), rng)
                    cover_count = np.zeros(num_faces, dtype=np.int32)
                    for e in R:
                        for f in edge_faces_arr[e]:
                            cover_count[f] += 1
        
        return R

    def polish_pass(R, deadline, rng):
        """
        Dedicated polish: repeatedly try to remove single edges until no more can be removed.
        """
        R = set(R)
        cover_count = np.zeros(num_faces, dtype=np.int32)
        for e in R:
            for f in edge_faces_arr[e]:
                cover_count[f] += 1
        
        while time.time() < deadline:
            improved = False
            R_list = list(R)
            rng.shuffle(R_list)
            
            for e_remove in R_list:
                if time.time() > deadline:
                    break
                
                valid = True
                for f in edge_faces_arr[e_remove]:
                    if cover_count[f] == 1:
                        valid = False
                        break
                
                if valid:
                    R.discard(e_remove)
                    for f in edge_faces_arr[e_remove]:
                        cover_count[f] -= 1
                    improved = True
                    # Restart from beginning since R changed
                    break
            
            if not improved:
                break
        
        return R

    best_R = set()
    best_size = num_edges
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    rng = random.Random(42)
    
    # Initial Greedy
    R = greedy_hit_set(set(range(num_faces)), rng)
    R = polish_pass(R, deadline, rng)
    best_R = set(R)
    best_size = len(R)
    
    # Local Search on initial solution
    R = local_search_hit_set(R, deadline, rng)
    R = polish_pass(R, deadline, rng)
    if len(R) < best_size:
        best_R = set(R)
        best_size = len(R)
    
    # Multiple restarts
    while time.time() < deadline:
        R_new = greedy_hit_set(set(range(num_faces)), rng)
        R_new = local_search_hit_set(R_new, deadline, rng)
        R_new = polish_pass(R_new, deadline, rng)
        
        if len(R_new) < best_size:
            best_R = set(R_new)
            best_size = len(R_new)
        
        # Kick from best
        if time.time() < deadline and len(best_R) >= 2:
            R_kick = set(best_R)
            kick_size = rng.choice([2, 3, 4])
            to_remove = rng.sample(list(R_kick), min(kick_size, len(R_kick)))
            for e in to_remove:
                R_kick.discard(e)
            
            cover_count_temp = np.zeros(num_faces, dtype=np.int32)
            for e in R_kick:
                for f in edge_faces_arr[e]:
                    cover_count_temp[f] += 1
            uncovered = set(f for f in range(num_faces) if cover_count_temp[f] == 0)
            
            if uncovered:
                R_add = greedy_hit_set(uncovered, rng)
                R_kick = R_kick | R_add
            
            R_kick = local_search_hit_set(R_kick, deadline, rng)
            R_kick = polish_pass(R_kick, deadline, rng)
            
            if len(R_kick) < best_size:
                best_R = set(R_kick)
                best_size = len(R_kick)

    # Final polish on best
    best_R = polish_pass(best_R, deadline, rng)

    # Convert best_R (set of edges to remove) to list of kept edges
    result = []
    for idx in range(num_edges):
        if idx not in best_R:
            u, v = idx_to_edge[idx]
            result.append([u, v])
            
    return result

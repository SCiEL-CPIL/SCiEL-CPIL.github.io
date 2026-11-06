import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                idx_to_edge[idx] = (x, y)
    
    # Precompute face structures
    # face_to_edges: list of lists of edge indices
    # edge_to_faces: list of lists of face indices
    face_to_edges = []
    edge_to_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_ij = (1 << i) | (1 << j)
            for u in range(n_vertices):
                if (u & mask_ij) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ mask_ij
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idx1 = edge_to_idx[e1]
                    idx2 = edge_to_idx[e2]
                    idx3 = edge_to_idx[e3]
                    idx4 = edge_to_idx[e4]
                    
                    face_idx = len(face_to_edges)
                    face_to_edges.append([idx1, idx2, idx3, idx4])
                    edge_to_faces[idx1].append(face_idx)
                    edge_to_faces[idx2].append(face_idx)
                    edge_to_faces[idx3].append(face_idx)
                    edge_to_faces[idx4].append(face_idx)
    
    n_faces = len(face_to_edges)
    
    # Helper: Greedy Hitting Set Initialization
    def greedy_init():
        removed = np.zeros(n_edges, dtype=bool)
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        unhit_faces = set(range(n_faces))
        
        # To speed up, we can just iterate.
        # Since n_edges is small (448), simple greedy is fine.
        while unhit_faces:
            if time.time() > deadline - 5:
                break
            best_edge = -1
            best_hits = 0
            # Optimize: Only check edges that are not removed
            # We can maintain a list of candidates?
            # For now, scan all.
            for e in range(n_edges):
                if removed[e]:
                    continue
                hits = 0
                for f in edge_to_faces[e]:
                    if f in unhit_faces:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_edge = e
                    if best_hits == 6:
                        break
            if best_edge == -1:
                break
            removed[best_edge] = True
            for f in edge_to_faces[best_edge]:
                face_hit_count[f] += 1
                if face_hit_count[f] == 1:
                    unhit_faces.discard(f)
        return removed, face_hit_count

    # Helper: Clean redundant edges
    def clean(removed, face_hit_count):
        improved = True
        while improved:
            improved = False
            for e in range(n_edges):
                if removed[e]:
                    can_remove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
        return removed, face_hit_count

    # Helper: Simulated Annealing Local Search
    def local_search(removed, face_hit_count, max_time):
        removed = removed.copy()
        face_hit_count = face_hit_count.copy()
        
        current_size = np.sum(removed)
        best_size = current_size
        best_removed = removed.copy()
        best_face_count = face_hit_count.copy()
        
        T = 10.0
        cooling = 0.999
        
        while time.time() < max_time:
            # 1. Try to remove redundant edges
            improved = False
            for e in range(n_edges):
                if removed[e]:
                    can_remove = True
                    for f in edge_to_faces[e]:
                        if face_hit_count[f] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                        improved = True
                        current_size -= 1
            if improved:
                if current_size < best_size:
                    best_size = current_size
                    best_removed = removed.copy()
                    best_face_count = face_hit_count.copy()
                continue # Restart scan as state changed significantly
            
            # 2. Kick: Remove 1, Add k
            # Pick random removed edge
            sel_indices = np.where(removed)[0]
            if len(sel_indices) == 0:
                break
            e1 = sel_indices[random.randint(0, len(sel_indices)-1)]
            
            # Remove e1
            removed[e1] = False
            current_size -= 1
            unhit = []
            for f in edge_to_faces[e1]:
                face_hit_count[f] -= 1
                if face_hit_count[f] == 0:
                    unhit.append(f)
            
            if not unhit:
                # Improvement
                if current_size < best_size:
                    best_size = current_size
                    best_removed = removed.copy()
                    best_face_count = face_hit_count.copy()
                continue
            
            # Need to cover 'unhit' faces
            # Greedily add edges
            added_edges = []
            covered = set(unhit)
            while covered:
                f = next(iter(covered))
                # Find edge in face f that covers most uncovered faces
                best_e = -1
                best_cov = 0
                for e in face_to_edges[f]:
                    if removed[e]:
                        continue
                    cov = 0
                    for f2 in edge_to_faces[e]:
                        if f2 in covered:
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_e = e
                if best_e == -1:
                    # Should not happen
                    break
                removed[best_e] = True
                added_edges.append(best_e)
                current_size += 1
                for f2 in edge_to_faces[best_e]:
                    face_hit_count[f2] += 1
                    if f2 in covered:
                        covered.discard(f2)
            
            delta = len(added_edges) - 1
            
            if delta < 0:
                # Impossible, but if it happens, accept
                if current_size < best_size:
                    best_size = current_size
                    best_removed = removed.copy()
                    best_face_count = face_hit_count.copy()
            elif delta == 0:
                # Accept swap
                if current_size < best_size:
                    best_size = current_size
                    best_removed = removed.copy()
                    best_face_count = face_hit_count.copy()
            else:
                # delta > 0, accept with prob exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    if current_size < best_size:
                        best_size = current_size
                        best_removed = removed.copy()
                        best_face_count = face_hit_count.copy()
                else:
                    # Revert
                    for e in added_edges:
                        removed[e] = False
                        for f in edge_to_faces[e]:
                            face_hit_count[f] -= 1
                    removed[e1] = True
                    for f in edge_to_faces[e1]:
                        face_hit_count[f] += 1
                    current_size += 1 # Revert size change for e1
                    current_size -= len(added_edges) # Revert size change for added
            
            T *= cooling
        
        return best_removed, best_face_count, best_size

    # Run multiple attempts
    best_removed = None
    best_size = n_edges
    
    # Try a few different greedy initializations with random tie-breaking
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        
        # Randomized Greedy
        removed = np.zeros(n_edges, dtype=bool)
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        unhit_faces = set(range(n_faces))
        
        while unhit_faces:
            if time.time() > deadline - 5:
                break
            best_edge = -1
            best_hits = 0
            # Add noise
            for e in range(n_edges):
                if removed[e]:
                    continue
                hits = 0
                for f in edge_to_faces[e]:
                    if f in unhit_faces:
                        hits += 1
                # Add small random noise to break ties
                score = hits + random.random() * 0.1
                if score > best_hits:
                    best_hits = score
                    best_edge = e
            
            if best_edge == -1:
                break
            removed[best_edge] = True
            for f in edge_to_faces[best_edge]:
                face_hit_count[f] += 1
                if face_hit_count[f] == 1:
                    unhit_faces.discard(f)
        
        # Clean
        removed, face_hit_count = clean(removed, face_hit_count)
        size = np.sum(removed)
        
        # Local Search
        removed, face_hit_count, size = local_search(removed, face_hit_count, deadline - 2 - attempt)
        
        if size < best_size:
            best_size = size
            best_removed = removed.copy()
    
    # Final Polish
    if best_removed is not None:
        face_hit_count = np.zeros(n_faces, dtype=np.int8)
        for e in range(n_edges):
            if best_removed[e]:
                for f in edge_to_faces[e]:
                    face_hit_count[f] += 1
        best_removed, face_hit_count, best_size = local_search(best_removed, face_hit_count, deadline - 1)
    
    # Convert to selected edges
    selected = ~best_removed
    result = []
    for e in range(n_edges):
        if selected[e]:
            u, v = idx_to_edge[e]
            result.append([u, v])
    
    return result

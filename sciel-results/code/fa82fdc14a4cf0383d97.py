import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # 1. Build Edge List and Face Data
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Identify faces
    # edge_id -> list of face_ids
    # face_id -> list of edge_ids
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
    # Enumerate all faces
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
                
                face_id = len(faces)
                faces.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
                    
    n_faces = len(faces)
    
    # Precompute for each edge, the list of faces it belongs to (for fast access)
    # Also precompute for each face, the 4 edges
    
    def greedy_init(seed):
        """Greedy hitting set initialization with given random seed."""
        random.seed(seed)
        removed = set()
        face_counts = [0] * n_faces
        uncovered_faces = set(range(n_faces))
        
        while uncovered_faces:
            # Pick a random uncovered face
            f = random.choice(list(uncovered_faces))
            
            # Find edge in this face that covers most uncovered faces
            best_e = -1
            max_cover = -1
            for e in faces[f]:
                if e in removed:
                    continue
                cover = 0
                for ff in edge_faces[e]:
                    if ff in uncovered_faces:
                        cover += 1
                if cover > max_cover:
                    max_cover = cover
                    best_e = e
            
            if best_e == -1:
                best_e = faces[f][0]
            
            removed.add(best_e)
            for ff in edge_faces[best_e]:
                face_counts[ff] += 1
                if face_counts[ff] == 1:
                    uncovered_faces.discard(ff)
        
        return removed, face_counts
    
    def local_search(removed, face_counts, deadline):
        """Local search to minimize removed set. Returns (best_removed, best_size)."""
        current_size = len(removed)
        best_removed = set(removed)
        best_size = current_size
        best_face_counts = face_counts[:]
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Delete non-critical edges
            for e in list(removed):
                critical = False
                for f in edge_faces[e]:
                    if face_counts[f] == 1:
                        critical = True
                        break
                if not critical:
                    removed.discard(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_removed = set(removed)
                        best_face_counts = face_counts[:]
            
            # Phase 2: Try swaps
            if not improved:
                # Try multiple swap attempts
                swap_attempts = 0
                max_swaps = 500
                while swap_attempts < max_swaps and time.time() < deadline:
                    swap_attempts += 1
                    if not removed:
                        break
                    
                    e_out = random.choice(list(removed))
                    
                    # Find critical faces for e_out
                    critical_faces = [f for f in edge_faces[e_out] if face_counts[f] == 1]
                    
                    if not critical_faces:
                        # Deletion candidate - should have been caught in Phase 1
                        continue
                    
                    # Find candidate edges in intersection of critical faces
                    candidate_edges = set(faces[critical_faces[0]])
                    for f in critical_faces[1:]:
                        candidate_edges &= set(faces[f])
                    
                    candidate_edges.discard(e_out)
                    valid_candidates = [e for e in candidate_edges if e not in removed]
                    
                    if valid_candidates:
                        e_in = random.choice(valid_candidates)
                        removed.discard(e_out)
                        removed.add(e_in)
                        for f in edge_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_faces[e_in]:
                            face_counts[f] += 1
                        improved = True
                        # Size unchanged, but may enable deletions
            
            if not improved:
                # Stuck - try perturbation or break
                # Try a few random valid swaps to escape local minimum
                perturbed = False
                for _ in range(50):
                    if time.time() >= deadline:
                        break
                    if not removed:
                        break
                    e_out = random.choice(list(removed))
                    critical_faces = [f for f in edge_faces[e_out] if face_counts[f] == 1]
                    if not critical_faces:
                        continue
                    candidate_edges = set(faces[critical_faces[0]])
                    for f in critical_faces[1:]:
                        candidate_edges &= set(faces[f])
                    candidate_edges.discard(e_out)
                    valid_candidates = [e for e in candidate_edges if e not in removed]
                    if valid_candidates:
                        e_in = random.choice(valid_candidates)
                        removed.discard(e_out)
                        removed.add(e_in)
                        for f in edge_faces[e_out]:
                            face_counts[f] -= 1
                        for f in edge_faces[e_in]:
                            face_counts[f] += 1
                        perturbed = True
                        break
                if not perturbed:
                    break
        
        return best_removed, best_size
    
    def polish(removed):
        """Final polish pass to remove any non-critical edges."""
        face_counts = [0] * n_faces
        for e in removed:
            for f in edge_faces[e]:
                face_counts[f] += 1
        
        changed = True
        while changed:
            changed = False
            for e in list(removed):
                critical = False
                for f in edge_faces[e]:
                    if face_counts[f] == 1:
                        critical = True
                        break
                if not critical:
                    removed.discard(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    changed = True
        
        return removed
    
    # 2. Multi-restart strategy
    start_time = time.time()
    total_deadline = start_time + 115
    
    # Try multiple restarts with different seeds
    best_removed = None
    best_size = float('inf')
    
    # First, do one thorough run
    removed, face_counts = greedy_init(42)
    best_removed, best_size = local_search(removed, face_counts, total_deadline)
    
    # If time remains, try more restarts
    if time.time() < total_deadline - 10:
        for seed in [123, 456, 789, 101, 202]:
            if time.time() >= total_deadline - 10:
                break
            removed, face_counts = greedy_init(seed)
            removed, size = local_search(removed, face_counts, total_deadline)
            if size < best_size:
                best_size = size
                best_removed = removed
    
    # 3. Final polish
    best_removed = polish(best_removed)
    
    # Reconstruct result
    result = []
    for i in range(n_edges):
        if i not in best_removed:
            result.append(list(edges[i]))
            
    return result

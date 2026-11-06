"""Combined program: Hitting Set formulation with 2-for-1 swaps and spectral initialization."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # Precompute for each edge, the list of faces it belongs to
    # A face is defined by (u, i, j) where u has bits i,j = 0, and the face contains
    # vertices u, u^i, u^j, u^i^j
    edge_to_faces = [[] for _ in range(n_edges)]
    face_count = 0
    faces = []  # list of (face_id, [e1, e2, e3, e4])
    
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
                    
                    idx1 = edge_index.get(e1)
                    idx2 = edge_index.get(e2)
                    idx3 = edge_index.get(e3)
                    idx4 = edge_index.get(e4)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                        face_id = face_count
                        face_count += 1
                        faces.append((face_id, [idx1, idx2, idx3, idx4]))
                        edge_to_faces[idx1].append(face_id)
                        edge_to_faces[idx2].append(face_id)
                        edge_to_faces[idx3].append(face_id)
                        edge_to_faces[idx4].append(face_id)
    
    # Hitting Set formulation:
    # We want to find a minimum set R of edges to remove such that every face has at least one edge in R.
    # Then selected edges = all_edges \ R.
    # State: boolean array `removed` of length n_edges. True if edge is removed.
    # Objective: Minimize sum(removed) such that for every face, at least one edge is removed.
    # Equivalently: Maximize sum(1 - removed) = n_edges - sum(removed) subject to constraints.
    
    # Let's use local search on the hitting set (removed edges).
    # Start with a greedy hitting set.
    
    # Greedy Set Cover:
    # Start with all faces unhit.
    # Repeatedly pick the edge that hits the most unhit faces, add to R.
    
    unhit_faces = set(range(face_count))
    face_to_edges = {fid: edges_list for fid, edges_list in faces}
    
    # For each edge, count how many unhit faces it covers
    edge_coverage = [0] * n_edges
    for fid in unhit_faces:
        for e in face_to_edges[fid]:
            edge_coverage[e] += 1
    
    removed = [False] * n_edges
    removed_set = set()
    
    while unhit_faces and time.time() < deadline - 30:
        # Find edge with max coverage
        best_edge = -1
        best_cov = -1
        for e in range(n_edges):
            if not removed[e]:
                if edge_coverage[e] > best_cov:
                    best_cov = edge_coverage[e]
                    best_edge = e
        if best_edge == -1 or best_cov <= 0:
            break
        removed[best_edge] = True
        removed_set.add(best_edge)
        # Update coverage for all faces containing best_edge
        for fid in edge_to_faces[best_edge]:
            if fid in unhit_faces:
                unhit_faces.remove(fid)
                for e in face_to_edges[fid]:
                    if not removed[e]:
                        edge_coverage[e] -= 1
    
    # Now we have a valid hitting set. Let's try to improve it via local search.
    # Move types:
    # 1. Remove an edge from R (if it doesn't create a violation)
    # 2. Add an edge to R (if it helps, i.e., allows removing others)
    # 3. Swap: remove one edge from R, add another
    
    # Let's track the number of edges in R (removed_set size)
    current_size = len(removed_set)
    best_size = current_size
    best_removed = removed[:]
    
    # Helper: check if removing edge e from R creates a violation
    # i.e., is there a face where all 4 edges are NOT in R?
    def check_validity_after_removal(e):
        # e is being removed from R
        for fid in edge_to_faces[e]:
            # Check if all other edges in this face are in R
            all_others_in_R = True
            for e2 in face_to_edges[fid]:
                if e2 != e and not removed[e2]:
                    all_others_in_R = False
                    break
            if not all_others_in_R:
                # This face would become uncovered
                return False
        return True
    
    # Helper: check if adding edge e to R is useful
    # It's useful if it covers at least one currently uncovered face? No, all faces are covered.
    # Adding an edge to R is only useful if it allows removing other edges.
    # So we don't add edges in isolation; we use swaps.
    
    # Let's do a local search loop with time limit
    T = 1.0
    alpha = 0.999
    T_min = 0.01
    
    # Precompute for each edge, the list of faces it is the ONLY edge in R for
    # This helps in checking validity quickly.
    # For each face, count how many edges are in R.
    face_R_count = [0] * face_count
    for fid, edges_list in faces:
        for e in edges_list:
            if removed[e]:
                face_R_count[fid] += 1
    
    # Local search
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        # Pick a random move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove an edge from R
            # Pick a random edge in R
            if not removed_set:
                continue
            e = random.choice(list(removed_set))
            # Check if we can remove it
            if check_validity_after_removal(e):
                # Remove it
                removed[e] = False
                removed_set.remove(e)
                current_size -= 1
                # Update face_R_count
                for fid in edge_to_faces[e]:
                    face_R_count[fid] -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_removed = removed[:]
            else:
                # Accept with probability exp(-delta/T) where delta = 1 (we tried to improve but failed)
                # Actually, this move is invalid, so we just reject it.
                # In simulated annealing, we might accept invalid moves temporarily,
                # but for hitting set, invalid means some face has 0 edges in R.
                # Let's just reject.
                pass
        elif move_type < 0.8:
            # Try to add an edge to R (only useful if it enables removals)
            # Pick a random edge NOT in R
            if len(removed_set) == n_edges:
                continue
            # Find an edge not in R
            candidates = [e for e in range(n_edges) if not removed[e]]
            if not candidates:
                continue
            e = random.choice(candidates)
            # Add it to R
            removed[e] = True
            removed_set.add(e)
            current_size += 1
            for fid in edge_to_faces[e]:
                face_R_count[fid] += 1
            # Now try to remove some edges that are now redundant
            # For each face that e covers, if face_R_count > 1, we might be able to remove another edge
            # This is a 2-for-1 swap: add e, remove f
            # Let's try to find an edge f that can be removed
            # For simplicity, just accept the addition for now and let the next iteration try to remove
        else:
            # Try a 2-for-1 swap: add e1, remove e2
            # Pick e1 not in R, e2 in R
            if not removed_set:
                continue
            candidates_add = [e for e in range(n_edges) if not removed[e]]
            if not candidates_add:
                continue
            e1 = random.choice(candidates_add)
            e2 = random.choice(list(removed_set))
            
            # Temporarily add e1 and remove e2
            removed[e1] = True
            removed_set.add(e1)
            removed[e2] = False
            removed_set.remove(e2)
            current_size += 1  # Net change: +1 - 1 = 0? No, we added one, removed one. Size stays same.
            # Wait, 2-for-1 means we add one and remove one? That's a swap.
            # Let's call it a swap.
            
            # Check validity
            valid = True
            for fid in edge_to_faces[e2]:
                if face_R_count[fid] == 0:  # After removing e2, if this face had only e2 in R
                    # But we added e1, so if e1 is in this face, it's still covered
                    if e1 in face_to_edges[fid]:
                        pass
                    else:
                        valid = False
                        break
            if valid:
                for fid in edge_to_faces[e1]:
                    # Adding e1 can't make a face uncovered
                    pass
            
            if not valid:
                # Revert
                removed[e1] = False
                removed_set.remove(e1)
                removed[e2] = True
                removed_set.add(e2)
                current_size -= 1
            else:
                # Update face_R_count
                for fid in edge_to_faces[e2]:
                    face_R_count[fid] -= 1
                for fid in edge_to_faces[e1]:
                    face_R_count[fid] += 1
                # Size didn't change, so no improvement in size.
                # But this might allow future removals.
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final polish: try to remove edges from R one by one
    for e in list(removed_set):
        if time.time() > deadline - 2:
            break
        if check_validity_after_removal(e):
            removed[e] = False
            removed_set.remove(e)
            current_size -= 1
            for fid in edge_to_faces[e]:
                face_R_count[fid] -= 1
    
    # Use best_removed if it's better
    if best_size < current_size:
        removed = best_removed[:]
        removed_set = set(e for e in range(n_edges) if removed[e])
    
    # Convert to list of selected edges
    result = []
    for i in range(n_edges):
        if not removed[i]:
            result.append(list(edges[i]))
    
    return result

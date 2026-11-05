import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    d = 7
    n = 2 ** d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute for each edge, which 2-faces it belongs to
    # A 2-face is identified by (vertex, bit_i, bit_j) where i < j
    # The 4 edges of face (x, i, j) are: (x,x^i), (x,x^j), (x^i,x^i^j), (x^j,x^i^j)
    edge_to_faces = {}
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                # 4 edges of this face
                v1, v2, v3, v4 = x, x ^ (1 << i), x ^ (1 << j), x ^ (1 << i) ^ (1 << j)
                edges_in_face = [
                    (min(v1, v2), max(v1, v2)),
                    (min(v2, v3), max(v2, v3)),
                    (min(v3, v4), max(v3, v4)),
                    (min(v4, v1), max(v4, v1))
                ]
                face_id = (x, i, j)
                for e in edges_in_face:
                    if e not in edge_to_faces:
                        edge_to_faces[e] = []
                    edge_to_faces[e].append(face_id)
    
    # Count how many edges are in each face
    # face_to_edges: face_id -> list of 4 edges
    face_to_edges = {}
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                v1, v2, v3, v4 = x, x ^ (1 << i), x ^ (1 << j), x ^ (1 << i) ^ (1 << j)
                edges_in_face = [
                    (min(v1, v2), max(v1, v2)),
                    (min(v2, v3), max(v2, v3)),
                    (min(v3, v4), max(v3, v4)),
                    (min(v4, v1), max(v4, v1))
                ]
                face_id = (x, i, j)
                face_to_edges[face_id] = edges_in_face
    
    def is_c4_free(selected):
        """Check if selected edges form a C4-free set"""
        for face_id, edges in face_to_edges.items():
            if all(e in selected for e in edges):
                return False
        return True
    
    # Greedy initial solution
    def greedy_initial(order):
        selected = set()
        result = []
        for edge in order:
            # Check if adding this edge creates a C4
            ok = True
            for face_id in edge_to_faces[edge]:
                edges_in_face = face_to_edges[face_id]
                if all(e in selected for e in edges_in_face if e != edge):
                    ok = False
                    break
            if ok:
                selected.add(edge)
                result.append(list(edge))
        return selected, result
    
    # Try multiple random orderings for greedy
    best_selected = None
    best_result = None
    best_count = 0
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = all_edges.copy()
        random.shuffle(order)
        selected, result = greedy_initial(order)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            best_result = result
    
    # If no greedy worked, use deterministic
    if best_selected is None:
        selected, result = greedy_initial(all_edges)
        best_selected = selected
        best_result = result
        best_count = len(selected)
    
    # Local search with swaps
    # For each edge not in selected, try adding it and removing minimal edges to avoid C4
    # Or: try removing one edge, then greedily add back as many as possible
    
    def try_improve(selected, result):
        """Try to improve by removing an edge and adding back others"""
        if time.time() > deadline:
            return selected, result, len(selected)
        
        # Build adjacency: for each edge, which other edges are in the same face
        # and which faces would be completed
        
        # Try removing each edge in selected
        for edge in list(selected):
            if time.time() > deadline:
                return selected, result, len(selected)
            
            # Remove this edge
            new_selected = selected - {edge}
            
            # Try to add back edges that were previously blocked
            # Find all edges that share a face with 'edge'
            candidate_edges = set()
            for face_id in edge_to_faces[edge]:
                for e in face_to_edges[face_id]:
                    if e != edge and e not in new_selected:
                        candidate_edges.add(e)
            
            # Try to add candidates greedily
            for cand in candidate_edges:
                if time.time() > deadline:
                    return selected, result, len(selected)
                # Check if adding cand creates a C4
                ok = True
                for face_id in edge_to_faces[cand]:
                    edges_in_face = face_to_edges[face_id]
                    if all(e in new_selected for e in edges_in_face if e != cand):
                        ok = False
                        break
                if ok:
                    new_selected.add(cand)
            
            if len(new_selected) > len(selected):
                # Build result
                result = [list(e) for e in new_selected]
                return new_selected, result, len(new_selected)
        
        return selected, result, len(selected)
    
    # Run local search for a while
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        best_selected, best_result, best_count = try_improve(best_selected, best_result)
        if len(best_selected) > best_count:
            improved = True
            best_count = len(best_selected)
    
    # Additional local search: try random swaps
    if time.time() < deadline:
        for _ in range(500):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            edge = random.choice(list(best_selected))
            new_selected = best_selected - {edge}
            
            # Try to add back random unselected edges
            unselected = [e for e in all_edges if e not in new_selected]
            random.shuffle(unselected)
            
            added = 0
            for cand in unselected[:50]:  # limit attempts
                ok = True
                for face_id in edge_to_faces[cand]:
                    edges_in_face = face_to_edges[face_id]
                    if all(e in new_selected for e in edges_in_face if e != cand):
                        ok = False
                        break
                if ok:
                    new_selected.add(cand)
                    added += 1
                    if added >= 2:
                        break
            
            if len(new_selected) > len(best_selected):
                best_selected = new_selected
                best_result = [list(e) for e in best_selected]
    
    return best_result

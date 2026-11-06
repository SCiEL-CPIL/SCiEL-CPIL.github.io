import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)  # 448
    
    # 1. Generate edges and map them to indices
    edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    # 2. Generate faces (4-cycles)
    # A face is defined by two distinct dimensions i, j and a vertex x 
    # where x has 0 in bits i and j (or any specific pattern, but we must avoid duplicates).
    # Standard way: for each pair of dimensions (i, j), and for each assignment of the other d-2 bits,
    # the vertices are x, x^(1<<i), x^(1<<j), x^(1<<(i|j)) is not quite right.
    # Vertices: v0 = x, v1 = x^(1<<i), v2 = x^(1<<j), v3 = x^(1<<i)^(1<<j).
    # To generate unique faces, we can iterate over all x and pairs (i,j) with i<j,
    # but this generates each face 4 times (once for each vertex in the face as the 'base' x if we aren't careful).
    # Actually, if we fix i < j, and iterate x from 0 to 2^d-1, we generate each face 4 times?
    # Let's check. Face vertices: {x, x^i, x^j, x^i^j}.
    # If we pick x' = x^i, the set is {x^i, x, x^i^j, x^j}. Same set.
    # So yes, 4 duplicates.
    # We can use a set of frozensets of edge indices to deduplicate.
    
    faces = []
    face_set = set()
    
    # Precompute edge indices for quick lookup
    # We need to map a pair of vertices to edge index
    # We already have edge_to_idx for (min(u,v), max(u,v))
    
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all vertices x
            # To reduce duplicates, we can require that x has 0 in bit i and 0 in bit j?
            # No, that only generates 2^(d-2) faces for this pair of dimensions.
            # Total faces should be C(d, 2) * 2^(d-2).
            # For d=7: 21 * 32 = 672.
            # So yes, we can just iterate x where bit i is 0 and bit j is 0.
            # This ensures unique generation.
            mask_i = 1 << i
            mask_j = 1 << j
            
            for x in range(n_vertices):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v0 = x
                    v1 = x ^ mask_i
                    v2 = x ^ mask_j
                    v3 = x ^ mask_i ^ mask_j
                    
                    # Get edge indices
                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    e32 = edge_to_idx[(min(v3, v2), max(v3, v2))]
                    e20 = edge_to_idx[(min(v2, v0), max(v2, v0))]
                    
                    face_edges = (e01, e13, e32, e20)
                    # Sort to ensure consistent tuple for set
                    face_key = tuple(sorted(face_edges))
                    
                    if face_key not in face_set:
                        face_set.add(face_key)
                        faces.append(face_key)
    
    n_faces = len(faces)
    
    # 3. Precompute edge-to-face incidence
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    def greedy_solve(order, current_time=None):
        state = [False] * n_edges
        face_counts = [0] * n_faces
        
        for e_idx in order:
            # Check if adding e_idx is valid
            valid = True
            for f_idx in edge_faces[e_idx]:
                if face_counts[f_idx] >= 3:
                    valid = False
                    break
            if valid:
                state[e_idx] = True
                for f_idx in edge_faces[e_idx]:
                    face_counts[f_idx] += 1
        
        return state, face_counts

    def local_search(state, face_counts, deadline):
        current_count = sum(state)
        best_state = state[:]
        best_count = current_count
        best_face_counts = face_counts[:]
        
        # Precompute list of edges for random selection
        all_edges = list(range(n_edges))
        
        while time.time() < deadline:
            # Try to add a random missing edge
            # To speed up, we can just pick a random missing edge
            # But we need to know which are missing.
            
            # Optimization: Maintain a list of missing edges?
            # Or just pick random edge indices and check if state is False.
            
            improved = False
            
            # Strategy 1: Try to add a single edge
            # Pick 50 random missing edges
            for _ in range(50):
                e_idx = random.choice(all_edges)
                if state[e_idx]:
                    continue
                
                # Check validity
                valid = True
                for f_idx in edge_faces[e_idx]:
                    if face_counts[f_idx] >= 3:
                        valid = False
                        break
                
                if valid:
                    # Add it
                    state[e_idx] = True
                    for f_idx in edge_faces[e_idx]:
                        face_counts[f_idx] += 1
                    current_count += 1
                    improved = True
                    break # Restart loop if we made progress
            
            if improved:
                if current_count > best_count:
                    best_count = current_count
                    best_state = state[:]
                    best_face_counts = face_counts[:]
                continue
            
            # Strategy 2: Remove one edge, then try to add as many as possible
            # Pick a random edge in the set to remove
            # To make it efficient, we can pick an edge that is "critical" or just random.
            # Let's pick a random edge from the current set.
            # Getting a random edge from the set efficiently:
            # We can maintain a list of active edges, but it changes.
            # For now, just pick random indices until we find one in the set.
            
            e_out = None
            for _ in range(100):
                candidate = random.choice(all_edges)
                if state[candidate]:
                    e_out = candidate
                    break
            
            if e_out is None:
                break # No edges in set? Unlikely.
            
            # Remove e_out
            state[e_out] = False
            for f_idx in edge_faces[e_out]:
                face_counts[f_idx] -= 1
            current_count -= 1
            
            # Try to add back edges, prioritizing those that were blocked by e_out
            # Candidates: All missing edges that share a face with e_out
            candidates = set()
            for f_idx in edge_faces[e_out]:
                for e_in_face in faces[f_idx]:
                    if not state[e_in_face]:
                        candidates.add(e_in_face)
            
            # Also, some edges might have been blocked by other faces that are now "lighter"?
            # No, removing an edge only decreases counts. It doesn't make other faces lighter in a way that unblocks edges not connected to e_out's faces.
            # Wait, if e_out was in face F1 and F2.
            # Edges in F1 might be unblocked. Edges in F2 might be unblocked.
            # Edges in other faces are not affected.
            # So candidates are correct.
            
            # Shuffle candidates to break symmetry
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            added_any = False
            for e_in in cand_list:
                valid = True
                for f_idx in edge_faces[e_in]:
                    if face_counts[f_idx] >= 3:
                        valid = False
                        break
                
                if valid:
                    state[e_in] = True
                    for f_idx in edge_faces[e_in]:
                        face_counts[f_idx] += 1
                    current_count += 1
                    added_any = True
            
            # If we added more edges than we removed (net gain > 0), we improved.
            # If net gain == 0, it's a neutral move. We might accept it to escape local minima.
            # If net gain < 0, we made it worse.
            
            if current_count >= best_count:
                # Accept if it's better or equal (to allow exploration)
                # But if it's equal, we might loop. 
                # Let's only accept if strictly better OR if we are stuck (no progress for a while).
                # For simplicity, accept if >= best_count.
                if current_count > best_count:
                    best_count = current_count
                    best_state = state[:]
                    best_face_counts = face_counts[:]
            else:
                # Revert
                state[e_out] = True
                for f_idx in edge_faces[e_out]:
                    face_counts[f_idx] += 1
                current_count += 1
                # Revert additions
                for e_in in cand_list:
                    if state[e_in]:
                        state[e_in] = False
                        for f_idx in edge_faces[e_in]:
                            face_counts[f_idx] -= 1
                        current_count -= 1
                
                # If we reverted, we are in the same state.
                # To avoid infinite loops, we might need to track visited states or just accept the risk.
                # Given the randomness, it's unlikely to loop forever on the exact same sequence.
                
        return best_state, best_count

    # Main Loop: Multiple Restarts
    best_overall_count = 0
    best_overall_state = None
    
    # Run several restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random order for greedy
        order = list(range(n_edges))
        random.shuffle(order)
        
        state, face_counts = greedy_solve(order)
        count = sum(state)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_state = state[:]
        
        # Local search
        if time.time() < deadline:
            improved_state, improved_count = local_search(state, face_counts, deadline)
            if improved_count > best_overall_count:
                best_overall_count = improved_count
                best_overall_state = improved_state[:]
        
        # Check time
        if time.time() > deadline:
            break

    # Convert best state to list of edges
    result = []
    if best_overall_state:
        for i, is_selected in enumerate(best_overall_state):
            if is_selected:
                u, v = edges[i]
                result.append([u, v])
    else:
        # Fallback: Empty or simple greedy
        order = list(range(n_edges))
        state, _ = greedy_solve(order)
        for i, is_selected in enumerate(state):
            if is_selected:
                u, v = edges[i]
                result.append([u, v])

    return result

# Run a quick sanity check if executed directly
if __name__ == "__main__":
    # Set seed for reproducibility in testing, but remove for final submission if randomness is desired
    # random.seed(42)
    res = construct()
    print(f"Edges: {len(res)}")
    # Verify no C4
    # ... (verification code omitted for brevity in thought process)

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
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def greedy_construct(randomized=False, seed=None):
        """Greedy vertex cover on the hypergraph of C4 faces.
        We want to remove minimum edges so no face has all 4 edges.
        Equivalent: maximize edges kept such that no face has all 4.
        """
        if seed is not None:
            random.seed(seed)
        
        selected = [True] * n_edges
        face_count = [4] * n_faces
        
        # Track violated faces (count == 4)
        violated = set(range(n_faces))
        
        while violated:
            # Count how many violated faces each edge participates in
            edge_viol_count = [0] * n_edges
            for f in violated:
                for e in face_edges[f]:
                    edge_viol_count[e] += 1
            
            if max(edge_viol_count) == 0:
                break
            
            if randomized:
                max_c = max(edge_viol_count)
                candidates = [e for e in range(n_edges) if edge_viol_count[e] == max_c]
                best_e = random.choice(candidates)
            else:
                best_e = max(range(n_edges), key=lambda e: edge_viol_count[e])
            
            selected[best_e] = False
            for f in edge_faces[best_e]:
                face_count[f] -= 1
                if face_count[f] == 4:
                    violated.add(f)
                else:
                    violated.discard(f)
        
        return selected, face_count
    
    def try_add_edges(selected, face_count):
        """Try to add back removed edges greedily."""
        unselected = [e for e in range(n_edges) if not selected[e]]
        random.shuffle(unselected)
        added = 0
        for e in unselected:
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
                added += 1
        return added
    
    def local_search_swap(selected, face_count, time_limit):
        """Local search: try to remove one edge and add back two others (net +1).
        Also try single add moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Single add: try to add a removed edge
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
            for e in unselected:
                if time.time() > time_limit:
                    break
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
            
            # Swap: remove edge e1, try to add edges e2, e3 that become possible
            # Strategy: for each removed edge, check if removing it from consideration
            # (i.e., it's already removed) allows adding two other edges
            # Better: pick a selected edge to remove, then try to add back 2 others
            if time.time() < time_limit:
                selected_edges = [e for e in range(n_edges) if selected[e]]
                random.shuffle(selected_edges)
                
                for e_remove in selected_edges[:50]:  # try up to 50 removal candidates
                    if time.time() > time_limit:
                        break
                    
                    # Temporarily remove e_remove
                    selected[e_remove] = False
                    for f in edge_faces[e_remove]:
                        face_count[f] -= 1
                    
                    # Now try to add back up to 2 edges
                    added_back = 0
                    unselected_now = [e for e in range(n_edges) if not selected[e] and e != e_remove]
                    random.shuffle(unselected_now)
                    
                    for e_add in unselected_now:
                        if added_back >= 2:
                            break
                        if time.time() > time_limit:
                            break
                        can_add = True
                        for f in edge_faces[e_add]:
                            if face_count[f] >= 3:
                                can_add = False
                                break
                        if can_add:
                            selected[e_add] = True
                            for f in edge_faces[e_add]:
                                face_count[f] += 1
                            added_back += 1
                    
                    # Net change: -1 (removed e_remove) + added_back
                    if added_back <= 1:
                        # No improvement or worse, revert
                        selected[e_remove] = True
                        for f in edge_faces[e_remove]:
                            face_count[f] += 1
                        # Undo the adds
                        for e_add in unselected_now:
                            if selected[e_add] and e_add != e_remove:
                                # This was added during this swap attempt
                                # Actually we need to track which were added
                                pass
                        # Simpler: just check if we improved
                        # If added_back <= 1, net is 0 or -1, revert everything
                        # Let's redo: track what was added
                        pass
                    
                    # This approach is getting complex. Let me simplify.
                    break  # Skip complex swap for now, use simpler approach
        
        return improved
    
    def simple_local_search(selected, face_count, time_limit):
        """Simple local search: repeatedly try single adds until no improvement."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            unselected = [e for e in range(n_edges) if not selected[e]]
            random.shuffle(unselected)
            for e in unselected:
                if time.time() > time_limit:
                    break
                can_add = True
                for f in edge_faces[e]:
                    if face_count[f] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    improved = True
        return improved
    
    def swap_local_search(selected, face_count, time_limit):
        """Swap-based local search: remove one edge, try to add two others.
        Net gain of +1 if both adds succeed."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            selected_edges = [e for e in range(n_edges) if selected[e]]
            random.shuffle(selected_edges)
            
            for e_remove in selected_edges:
                if time.time() > time_limit:
                    break
                
                # Temporarily remove
                selected[e_remove] = False
                for f in edge_faces[e_remove]:
                    face_count[f] -= 1
                
                # Try to add back 2 edges
                added_back = []
                unselected_now = [e for e in range(n_edges) if not selected[e]]
                random.shuffle(unselected_now)
                
                for e_add in unselected_now:
                    if len(added_back) >= 2:
                        break
                    can_add = True
                    for f in edge_faces[e_add]:
                        if face_count[f] >= 3:
                            can_add = False
                            break
                    if can_add:
                        selected[e_add] = True
                        for f in edge_faces[e_add]:
                            face_count[f] += 1
                        added_back.append(e_add)
                
                if len(added_back) < 2:
                    # Revert: put back e_remove, remove added_back
                    selected[e_remove] = True
                    for f in edge_faces[e_remove]:
                        face_count[f] += 1
                    for e_add in added_back:
                        selected[e_add] = False
                        for f in edge_faces[e_add]:
                            face_count[f] -= 1
                else:
                    improved = True
                    break  # Found an improvement, restart loop
        
        return improved
    
    # Phase 1: Initial greedy (deterministic)
    selected, face_count = greedy_construct(randomized=False)
    try_add_edges(selected, face_count)
    best_count = sum(selected)
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    # Phase 2: Local search on initial solution
    simple_local_search(selected, face_count, deadline)
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
    
    swap_local_search(selected, face_count, deadline)
    current_count = sum(selected)
    if current_count > best_count:
        best_count = current_count
        best_selected = selected.copy()
        best_face_count = face_count.copy()
    
    # Phase 3: Multiple randomized restarts
    for restart in range(300):
        if time.time() > deadline - 2:
            break
        
        seed = random.randint(0, 2**31 - 1)
        selected, face_count = greedy_construct(randomized=True, seed=seed)
        try_add_edges(selected, face_count)
        simple_local_search(selected, face_count, deadline - 1)
        swap_local_search(selected, face_count, deadline - 1)
        
        current_count = sum(selected)
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            best_face_count = face_count.copy()
    
    # Build result from best solution
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result

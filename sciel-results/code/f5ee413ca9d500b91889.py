import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Edge representation: index 0..447
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # Precompute face neighbors
    face_neighbors = [[] for _ in range(n_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u >= v: continue
            e_idx = edge_to_idx[(u, v)]
            
            for j in range(d):
                if i == j: continue
                u_j = u ^ (1 << j)
                v_j = v ^ (1 << j)
                
                def get_idx(a, b):
                    if a > b: a, b = b, a
                    return edge_to_idx.get((a, b), -1)
                
                e_opp_idx = get_idx(u_j, v_j)
                e_side1_idx = get_idx(u, u_j)
                e_side2_idx = get_idx(v, v_j)
                
                face_neighbors[e_idx].append((e_opp_idx, e_side1_idx, e_side2_idx))
                face_neighbors[e_idx].append((e_side1_idx, e_opp_idx, e_side2_idx))
                face_neighbors[e_idx].append((e_side2_idx, e_opp_idx, e_side1_idx))
                
    selected = [False] * n_edges
    conflict_count = [0] * n_edges
    
    def add_edge(e):
        if selected[e]: return False
        if conflict_count[e] > 0: return False
        selected[e] = True
        for n, a, b in face_neighbors[e]:
            if selected[a] and selected[b]:
                conflict_count[n] += 1
        return True

    def remove_edge(e):
        if not selected[e]: return False
        selected[e] = False
        for n, a, b in face_neighbors[e]:
            if selected[a] and selected[b]:
                conflict_count[n] -= 1
        return True
        
    def greedy_fill():
        edge_indices = list(range(n_edges))
        random.shuffle(edge_indices)
        added = 0
        for e in edge_indices:
            if conflict_count[e] == 0:
                add_edge(e)
                added += 1
        return added

    def get_count():
        return sum(selected)

    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    best_count = 0
    best_selected = [False] * n_edges
    
    # Initial Greedy
    greedy_fill()
    best_count = get_count()
    best_selected = selected[:]
    
    # Local Search
    # We alternate between small moves (Add/Swap) and large moves (Kick)
    
    no_improve_count = 0
    max_no_improve = 50 # After 50 swaps without improvement, do a kick
    
    while time.time() < deadline:
        curr_count = get_count()
        
        # 1. Try to add any valid edge
        # Find all addable edges
        addable = [e for e in range(n_edges) if not selected[e] and conflict_count[e] == 0]
        
        if addable:
            e = random.choice(addable)
            add_edge(e)
            no_improve_count = 0
            continue
            
        # 2. No addable edges. Try Swap.
        if no_improve_count < max_no_improve:
            unselected = [e for e in range(n_edges) if not selected[e]]
            if unselected:
                e = random.choice(unselected)
                # Find a blocking face
                # conflict_count[e] > 0
                blocking_faces = []
                for n, a, b in face_neighbors[e]:
                    if selected[n] and selected[a] and selected[b]:
                        blocking_faces.append((n, a, b))
                        
                if blocking_faces:
                    n, a, b = random.choice(blocking_faces)
                    candidates = [n, a, b]
                    r = random.choice(candidates)
                    
                    # Perform swap
                    remove_edge(r)
                    # Try to add e
                    if conflict_count[e] == 0:
                        add_edge(e)
                    
                    # Recover with greedy fill
                    greedy_fill()
                    
                    new_count = get_count()
                    if new_count > best_count:
                        best_count = new_count
                        best_selected = selected[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        # If we didn't improve, should we revert?
                        # Reverting is safer for local search, but ILS accepts.
                        # Let's accept to explore.
                else:
                    no_improve_count += 1
            else:
                break # All selected? Impossible for max independent set of this type, but safe.
        else:
            # 3. Do a Kick
            k = random.randint(3, 8)
            selected_indices = [i for i in range(n_edges) if selected[i]]
            if selected_indices:
                to_remove = random.sample(selected_indices, min(k, len(selected_indices)))
                for e in to_remove:
                    remove_edge(e)
                
                greedy_fill()
                
                new_count = get_count()
                if new_count > best_count:
                    best_count = new_count
                    best_selected = selected[:]
                
                no_improve_count = 0 # Reset counter after kick

    # Return the best found
    out = []
    for i in range(n_edges):
        if best_selected[i]:
            u, v = idx_to_edge[i]
            out.append([u, v])
    return out

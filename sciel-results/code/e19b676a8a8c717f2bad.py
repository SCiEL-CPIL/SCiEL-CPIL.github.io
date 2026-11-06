import time
import random
import sys

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
    # A face is a set of 4 edges.
    # We need to map:
    # edge_id -> list of face_ids
    # face_id -> list of edge_ids
    
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
    # Enumerate all faces
    # Face defined by dimensions i, j and mask of other d-2 bits
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
    
    # 2. Initial State: Greedy "Remove until valid"
    # Start with all edges selected
    selected = set(range(n_edges))
    face_counts = [0] * n_faces
    
    # Calculate initial counts
    for e in selected:
        for f in edge_faces[e]:
            face_counts[f] += 1
            
    # Repair: Remove edges from full faces
    # To make it deterministic or fast, we can just loop until no full faces
    # But random removal helps avoid bias.
    changed = True
    while changed:
        changed = False
        # Find a full face
        full_face_idx = -1
        for f in range(n_faces):
            if face_counts[f] == 4:
                full_face_idx = f
                break
                
        if full_face_idx != -1:
            # Pick a random edge from this face to remove
            e_to_remove = random.choice(faces[full_face_idx])
            selected.remove(e_to_remove)
            for f in edge_faces[e_to_remove]:
                face_counts[f] -= 1
            changed = True
            
    current_size = len(selected)
    best_size = current_size
    best_selected = set(selected)
    
    # 3. Simulated Annealing
    start_time = time.time()
    timeout = 115
    
    # Temperature parameters
    T = 1.0
    cooling_rate = 0.9999
    
    # Pre-compute list of all edges for random selection
    all_edges_list = list(range(n_edges))
    
    def try_add(e):
        """Check if edge e can be added. Returns True if valid."""
        for f in edge_faces[e]:
            if face_counts[f] == 3:
                return False
        return True

    def add_edge(e):
        selected.add(e)
        for f in edge_faces[e]:
            face_counts[f] += 1

    def remove_edge(e):
        selected.remove(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1

    def try_swap(e_out, e_in):
        """Check if we can swap e_out with e_in. Returns True if valid."""
        # Temporarily remove e_out
        for f in edge_faces[e_out]:
            face_counts[f] -= 1
            
        # Check if e_in can be added
        valid = True
        for f in edge_faces[e_in]:
            if face_counts[f] == 3:
                valid = False
                break
                
        if valid:
            # Perform swap
            selected.remove(e_out)
            selected.add(e_in)
            # Re-add counts for e_out (since we subtracted them, and we are replacing it)
            # Actually, we subtracted e_out. Now we add e_in.
            for f in edge_faces[e_in]:
                face_counts[f] += 1
            # We need to restore e_out's contribution? No, e_out is gone.
            # Wait, the logic is:
            # State S.
            # S' = S - {e_out} + {e_in}
            # We subtracted e_out from counts.
            # We checked e_in against (counts - e_out).
            # If valid, we add e_in to counts.
            # So counts now reflect S'.
            return True
        else:
            # Revert
            for f in edge_faces[e_out]:
                face_counts[f] += 1
            return False

    # Main Loop
    steps = 0
    last_improvement_time = start_time
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to Add
            # Pick a random unselected edge
            # Efficiently: pick random edge, if selected, try another?
            # Or maintain a list of unselected? That's expensive to update.
            # Just pick random edge.
            e = random.choice(all_edges_list)
            if e not in selected:
                if try_add(e):
                    add_edge(e)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = set(selected)
                        last_improvement_time = time.time()
                        no_improve_count = 0
                        # Restart cooling slightly to continue exploring
                        T = max(T, 0.5)
                else:
                    # Failed to add. Maybe try a swap to unblock?
                    # For now, just skip or handle in other branches.
                    pass
        elif move_type < 0.9:
            # Try to Swap
            # Pick random selected e_out
            # Pick random unselected e_in
            # To pick random selected:
            # list(selected) is O(N). N~300. Acceptable.
            # But we do this millions of times.
            # Better: pick random e, if selected, use as e_out.
            e_out = random.choice(all_edges_list)
            if e_out in selected:
                e_in = random.choice(all_edges_list)
                if e_in not in selected:
                    if try_swap(e_out, e_in):
                        # Size unchanged
                        # Accept with probability 1 (exploration)
                        pass
        else:
            # Try to Remove
            e = random.choice(all_edges_list)
            if e in selected:
                # Accept with probability exp(-1/T)
                if random.random() < 0.05: # Fixed low prob for removal to avoid shrinking too much
                    remove_edge(e)
                    current_size -= 1
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        steps += 1
        
        # Restart mechanism: if no improvement for a while, restart with higher T
        if time.time() - last_improvement_time > 2.0:
            no_improve_count += 1
            if no_improve_count > 50:
                # Reset to best state and restart
                selected = set(best_selected)
                face_counts = [0] * n_faces
                for e in selected:
                    for f in edge_faces[e]:
                        face_counts[f] += 1
                current_size = best_size
                T = 1.0
                no_improve_count = 0
                last_improvement_time = time.time()
                break
        
    # 4. Final Polish: Greedy Add
    # Restore best state for final polish
    selected = set(best_selected)
    face_counts = [0] * n_faces
    for e in selected:
        for f in edge_faces[e]:
            face_counts[f] += 1
    current_size = best_size
    
    # Try to add any remaining edges greedily
    improved = True
    while improved:
        improved = False
        for e in all_edges_list:
            if e not in selected:
                if try_add(e):
                    add_edge(e)
                    current_size += 1
                    improved = True
        if current_size > best_size:
            best_size = current_size
            best_selected = set(selected)

    # Reconstruct best_selected to ensure validity?
    # The SA maintained validity, so best_selected should be valid.
    
    result = []
    for i in best_selected:
        result.append(list(edges[i]))
        
    return result

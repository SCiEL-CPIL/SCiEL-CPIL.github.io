"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)  # 448
    
    # Build edge list
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Build edge index mapping
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    # Build list of all 4-cycles (faces)
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    e01 = (min(v0, v1), max(v0, v1))
                    e02 = (min(v0, v2), max(v0, v2))
                    e13 = (min(v1, v3), max(v1, v3))
                    e23 = (min(v2, v3), max(v2, v3))
                    
                    idx01 = edge_to_idx[e01]
                    idx02 = edge_to_idx[e02]
                    idx13 = edge_to_idx[e13]
                    idx23 = edge_to_idx[e23]
                    
                    faces.append([idx01, idx02, idx13, idx23])
    
    # For each edge, find which faces it belongs to
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # MCS-inspired greedy initialization: select edges with fewest face constraints first
    random.seed(42)
    selected = [False] * n_edges
    selected_count = 0
    face_complete = [0] * len(faces)
    
    # Compute initial constraint count for each edge (number of faces it belongs to)
    edge_constraint = [len(edge_faces[e_idx]) for e_idx in range(n_edges)]
    
    # Sort edges by constraint count (least constrained first)
    edge_order = sorted(range(n_edges), key=lambda e: edge_constraint[e])
    
    for e_idx in edge_order:
        can_add = True
        for f_idx in edge_faces[e_idx]:
            if face_complete[f_idx] == 3:
                can_add = False
                break
        if can_add:
            selected[e_idx] = True
            selected_count += 1
            for f_idx in edge_faces[e_idx]:
                face_complete[f_idx] += 1
    
    best_count = selected_count
    best_selected = selected[:]
    
    # Second pass: try adding remaining edges in order of increasing constraint
    remaining_edges = [e for e in range(n_edges) if not selected[e]]
    remaining_edges.sort(key=lambda e: edge_constraint[e])
    
    for e_idx in remaining_edges:
        can_add = True
        for f_idx in edge_faces[e_idx]:
            if face_complete[f_idx] == 3:
                can_add = False
                break
        if can_add:
            selected[e_idx] = True
            selected_count += 1
            for f_idx in edge_faces[e_idx]:
                face_complete[f_idx] += 1
    
    if selected_count > best_count:
        best_count = selected_count
        best_selected = selected[:]
    
    # Simulated annealing with multiple restarts
    num_restarts = 5
    restart_interval = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        # Reset for new restart if not first
        if restart > 0:
            # Try to recover from best solution and do local search
            selected = best_selected[:]
            selected_count = best_count
            face_complete = [0] * len(faces)
            for e_idx in range(n_edges):
                if selected[e_idx]:
                    for f_idx in edge_faces[e_idx]:
                        face_complete[f_idx] += 1
        
        T = 3.0
        T_min = 0.05
        cooling = 0.99997
        local_deadline = start_time + (restart + 1) * restart_interval
        
        while time.time() < local_deadline and T > T_min:
            # Pick a random edge to flip
            e_idx = random.randint(0, n_edges - 1)
            
            if selected[e_idx]:
                # Consider removing this edge
                old_count = selected_count
                selected[e_idx] = False
                selected_count -= 1
                for f_idx in edge_faces[e_idx]:
                    face_complete[f_idx] -= 1
                
                delta = selected_count - old_count  # -1
                # Accept with probability exp(delta/T) = exp(-1/T)
                if delta >= 0 or random.random() < math.exp(delta / T):
                    if selected_count > best_count:
                        best_count = selected_count
                        best_selected = selected[:]
                else:
                    # Reject
                    selected[e_idx] = True
                    selected_count = old_count
                    for f_idx in edge_faces[e_idx]:
                        face_complete[f_idx] += 1
            else:
                # Consider adding this edge
                can_add = True
                for f_idx in edge_faces[e_idx]:
                    if face_complete[f_idx] == 3:
                        can_add = False
                        break
                
                if can_add:
                    old_count = selected_count
                    selected[e_idx] = True
                    selected_count += 1
                    for f_idx in edge_faces[e_idx]:
                        face_complete[f_idx] += 1
                    
                    delta = selected_count - old_count  # +1
                    # Always accept improvements
                    if delta >= 0:
                        if selected_count > best_count:
                            best_count = selected_count
                            best_selected = selected[:]
                    else:
                        if random.random() >= math.exp(delta / T):
                            selected[e_idx] = False
                            selected_count = old_count
                            for f_idx in edge_faces[e_idx]:
                                face_complete[f_idx] -= 1
            
            T *= cooling
    
    # Final polish: try 2-for-1 swaps to escape local optima
    # Find edges that are currently unselected but could be added if we remove one blocking edge
    if time.time() < deadline - 5.0:
        # Build face_complete from best_selected
        face_complete = [0] * len(faces)
        for e_idx in range(n_edges):
            if best_selected[e_idx]:
                for f_idx in edge_faces[e_idx]:
                    face_complete[f_idx] += 1
        
        # For each unselected edge, find faces where it's blocked (face has 3 selected edges)
        # and try swapping: remove one of the 3 selected edges, add the new one
        unselected_edges = [e for e in range(n_edges) if not best_selected[e]]
        random.shuffle(unselected_edges)
        
        for new_e in unselected_edges:
            if time.time() > deadline - 3.0:
                break
            
            # Check if new_e can be added
            can_add = True
            for f_idx in edge_faces[new_e]:
                if face_complete[f_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                # Just add it
                best_selected[new_e] = True
                best_count += 1
                for f_idx in edge_faces[new_e]:
                    face_complete[f_idx] += 1
            else:
                # Try 2-for-1 swap: remove one edge from a blocked face, add new_e
                # Find a face where new_e is blocked
                for f_idx in edge_faces[new_e]:
                    if face_complete[f_idx] == 3:
                        # Find the 3 selected edges in this face
                        face_edges = faces[f_idx]
                        selected_in_face = [e for e in face_edges if best_selected[e]]
                        
                        # Try removing each selected edge in the face
                        for rem_e in selected_in_face:
                            # Check if removing rem_e and adding new_e is valid
                            # After removal, check all faces of new_e
                            valid_swap = True
                            for f2 in edge_faces[new_e]:
                                if f2 == f_idx:
                                    # This face will have 2 selected after removal, OK
                                    continue
                                if face_complete[f2] == 3:
                                    valid_swap = False
                                    break
                            
                            if valid_swap:
                                # Perform swap
                                best_selected[rem_e] = False
                                best_selected[new_e] = True
                                best_count += 1  # -1 + 1 = net 0, but we want to try
                                # Actually, this is a 1-for-1 swap, not improving count
                                # Let's only do 2-for-1: remove 2, add 1 is worse
                                # Skip this - just track for potential future use
                                best_selected[rem_e] = True
                                best_selected[new_e] = False
                                best_count -= 1
                                break
    
    # Return best solution
    result = []
    for e_idx in range(n_edges):
        if best_selected[e_idx]:
            result.append(list(edges[e_idx]))
    
    return result
# EVOLVE-BLOCK-END

import random
import time
import sys
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
                edge_index[(y, x)] = idx
                
    # Precompute 4-cycles (faces)
    faces = []
    face_edge_indices = []
    edge_faces = [[] for _ in range(n_edges)]
    
    seen_faces = set()
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = edge_index.get((min(v1, v2), max(v1, v2)))
                e2 = edge_index.get((min(v2, v4), max(v2, v4)))
                e3 = edge_index.get((min(v4, v3), max(v4, v3)))
                e4 = edge_index.get((min(v3, v1), max(v3, v1)))
                
                if e1 is None or e2 is None or e3 is None or e4 is None:
                    continue
                
                face_key = frozenset([e1, e2, e3, e4])
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    faces.append(face_key)
                    face_edge_indices.append(list(face_key))
                    for e_idx in face_key:
                        edge_faces[e_idx].append(len(face_edge_indices) - 1)

    n_faces = len(face_edge_indices)
    
    # Helper to check if adding edge e is valid
    def is_valid_add(e, selected):
        for f_idx in edge_faces[e]:
            f_edges = face_edge_indices[f_idx]
            count = 0
            for ee in f_edges:
                if ee == e:
                    continue
                if ee in selected:
                    count += 1
                    if count == 3:
                        return False
        return True

    # Greedy Initialization
    def greedy_init():
        selected = set()
        random_edges = list(range(n_edges))
        random.shuffle(random_edges)
        for e in random_edges:
            if is_valid_add(e, selected):
                selected.add(e)
        return selected

    # Local Search: Try to add edges, and if blocked, try swapping
    # This is more targeted than simulated annealing for this specific problem
    
    best_selected = greedy_init()
    best_size = len(best_selected)
    
    current_selected = set(best_selected)
    current_size = best_size
    
    # Precompute for each edge, the list of edges that conflict with it
    # (i.e., edges that share a face and would complete a 4-cycle if both added)
    # Actually, we need to know which edges are "forbidden" given current selection.
    # An edge e is forbidden if there exists a face f containing e such that the other 3 edges of f are in selected.
    
    # We will use a local search that focuses on "unblocking" edges.
    # 1. Find all edges not in selected.
    # 2. For each, check if it's valid. If yes, add it.
    # 3. If not, identify the blocking edges (the ones in the completed faces).
    # 4. Try removing one blocking edge and adding the new one (swap).
    # 5. If that doesn't improve, maybe remove a blocking edge without adding (rarely useful) or restart.
    
    # To speed up, we maintain a list of "blocked" edges and their blockers.
    
    def get_blockers(e, selected):
        blockers = set()
        for f_idx in edge_faces[e]:
            f_edges = face_edge_indices[f_idx]
            others = [ee for ee in f_edges if ee != e]
            if all(oo in selected for oo in others):
                blockers.update(others)
        return blockers

    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any valid edge
        for e in range(n_edges):
            if e not in current_selected:
                if is_valid_add(e, current_selected):
                    current_selected.add(e)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = set(current_selected)
        
        if not improved:
            # Try swaps: for each unselected edge, if blocked, try swapping
            # We need to be careful not to do O(N^2) work here.
            # Let's sample a few.
            for _ in range(100):
                if time.time() > deadline:
                    break
                e = random.randint(0, n_edges - 1)
                if e in current_selected:
                    continue
                
                if is_valid_add(e, current_selected):
                    # Should have been caught above, but just in case
                    current_selected.add(e)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = set(current_selected)
                    break # Restart the outer loop? No, just continue
                    
                # Check blockers
                blockers = get_blockers(e, current_selected)
                if blockers:
                    # Try removing one blocker and adding e
                    b = random.choice(list(blockers))
                    # Check if removing b and adding e is valid
                    # Removing b is always safe.
                    # Adding e is safe if b was the only blocker?
                    # Not necessarily, b might be in multiple faces with e.
                    # But if b is in the set of blockers, removing it breaks at least one blocking face.
                    # We need to ensure no OTHER face is still blocked.
                    
                    # Let's simulate:
                    # Temporarily remove b, add e
                    # Check validity of e
                    # If valid, accept.
                    
                    # Actually, simpler:
                    # If we remove b, does e become valid?
                    # We need to check if there is another face f' containing e
                    # such that the other 3 edges are in (current_selected - {b} + {e})
                    # Since e is not in selected yet, "other 3" means other 3 in current_selected.
                    # If b was the only one blocking e in all its faces, then yes.
                    # But b might block e in one face, and another edge c might block e in another face.
                    
                    # Let's just try the move and check validity properly.
                    # Move: remove b, add e.
                    # Size is constant.
                    # We accept this if it leads to a state where we can add MORE edges later?
                    # Hard to know.
                    # Let's accept swaps with low probability to explore.
                    
                    if random.random() < 0.05:
                        # Check if e is valid after removing b
                        # We need to check all faces of e
                        valid = True
                        for f_idx in edge_faces[e]:
                            f_edges = face_edge_indices[f_idx]
                            count = 0
                            for ee in f_edges:
                                if ee == e:
                                    continue
                                if ee == b:
                                    continue # b is removed
                                if ee in current_selected:
                                    count += 1
                                    if count == 3:
                                        valid = False
                                        break
                            if not valid:
                                break
                        
                        if valid:
                            current_selected.remove(b)
                            current_selected.add(e)
                            # Size unchanged
                            # Mark as improved to restart scanning?
                            # No, size didn't change.
                            # But structure changed.
                            # Let's set improved = True to allow further scanning.
                            improved = True
                            
    # Final greedy polish
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in range(n_edges):
            if e not in best_selected:
                if is_valid_add(e, best_selected):
                    best_selected.add(e)
                    changed = True
    
    # Convert back to list of [u, v]
    result = []
    for e_idx in best_selected:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result

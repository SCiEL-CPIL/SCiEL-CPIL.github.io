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
                
    edge_faces = [[] for _ in range(n_edges)]
    faces = []
    
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
    all_edges_list = list(range(n_edges))
    
    def calculate_face_counts(removed_set):
        counts = [0] * n_faces
        for e in range(n_edges):
            if e not in removed_set:
                for f in edge_faces[e]:
                    counts[f] += 1
        return counts

    def is_valid_unremove(e, face_counts):
        for f in edge_faces[e]:
            if face_counts[f] == 3:
                return False
        return True

    def apply_unremove(e, removed_set, face_counts):
        removed_set.discard(e)
        for f in edge_faces[e]:
            face_counts[f] += 1

    def apply_remove(e, removed_set, face_counts):
        removed_set.add(e)
        for f in edge_faces[e]:
            face_counts[f] -= 1

    def local_search(removed_set, face_counts, start_time, timeout):
        current_removed = len(removed_set)
        best_removed = current_removed
        best_removed_set = set(removed_set)
        
        T = 1.0
        cooling_rate = 0.999
        last_improvement = time.time()
        
        while time.time() - start_time < timeout:
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to un-remove 1 edge (Improve by 1)
                e = random.choice(all_edges_list)
                if e in removed_set:
                    if is_valid_unremove(e, face_counts):
                        apply_unremove(e, removed_set, face_counts)
                        current_removed -= 1
                        if current_removed < best_removed:
                            best_removed = current_removed
                            best_removed_set = set(removed_set)
                            last_improvement = time.time()
                            T = max(T, 0.5)
                            
            elif move_type < 0.7:
                # Try Swap 1-for-1 (Neutral, for exploration)
                e_rm = random.choice(all_edges_list)
                if e_rm not in removed_set:
                    e_un = random.choice(all_edges_list)
                    if e_un in removed_set and e_un != e_rm:
                        # Check if removing e_rm allows un-removing e_un
                        # Temporarily remove e_rm
                        for f in edge_faces[e_rm]:
                            face_counts[f] -= 1
                        valid = is_valid_unremove(e_un, face_counts)
                        if valid:
                            # Apply
                            removed_set.add(e_rm)
                            removed_set.discard(e_un)
                            for f in edge_faces[e_un]:
                                face_counts[f] += 1
                            # Size unchanged
                        else:
                            # Revert
                            for f in edge_faces[e_rm]:
                                face_counts[f] += 1
                            
            elif move_type < 0.85:
                # Try Swap 1-for-2 (Improve by 1: Remove 1, Un-remove 2)
                # This is a "2-for-1" in terms of edges changed, but net -1 removed
                e_rm = random.choice(all_edges_list)
                if e_rm not in removed_set:
                    # Pick 2 random removed edges to try to un-remove
                    # To be efficient, pick from a small random subset of removed edges
                    # Or just pick 2 random edges and check if they are removed
                    candidates = []
                    for _ in range(5):
                        e_un = random.choice(all_edges_list)
                        if e_un in removed_set and e_un != e_rm:
                            candidates.append(e_un)
                        if len(candidates) >= 2:
                            break
                    
                    if len(candidates) >= 2:
                        e_un1, e_un2 = candidates[0], candidates[1]
                        # Temporarily remove e_rm
                        for f in edge_faces[e_rm]:
                            face_counts[f] -= 1
                        valid1 = is_valid_unremove(e_un1, face_counts)
                        if valid1:
                            # Temporarily un-remove e_un1 to check e_un2?
                            # Actually, we need to check if BOTH can be un-removed given e_rm is removed.
                            # They are independent if they don't share faces with each other in a conflicting way?
                            # No, un-removing e_un1 increases face counts.
                            # So we must check sequentially.
                            
                            # Apply e_un1 temporarily
                            for f in edge_faces[e_un1]:
                                face_counts[f] += 1
                            
                            valid2 = is_valid_unremove(e_un2, face_counts)
                            
                            if valid2:
                                # Apply e_un2
                                for f in edge_faces[e_un2]:
                                    face_counts[f] += 1
                                # Finalize: e_rm removed, e_un1 unremoved, e_un2 unremoved
                                removed_set.add(e_rm)
                                removed_set.discard(e_un1)
                                removed_set.discard(e_un2)
                                current_removed -= 1
                                if current_removed < best_removed:
                                    best_removed = current_removed
                                    best_removed_set = set(removed_set)
                                    last_improvement = time.time()
                                    T = max(T, 0.5)
                            else:
                                # Revert e_un1
                                for f in edge_faces[e_un1]:
                                    face_counts[f] -= 1
                        # Revert e_rm
                        for f in edge_faces[e_rm]:
                            face_counts[f] += 1

            else:
                # Try Swap 2-for-1 (Worsen by 1: Remove 2, Un-remove 1) - Exploration
                if random.random() < T: # Accept worse moves with probability T
                    e_rm1 = random.choice(all_edges_list)
                    e_rm2 = random.choice(all_edges_list)
                    if e_rm1 not in removed_set and e_rm2 not in removed_set and e_rm1 != e_rm2:
                        e_un = random.choice(all_edges_list)
                        if e_un in removed_set:
                            # Temporarily remove e_rm1, e_rm2
                            for f in edge_faces[e_rm1]:
                                face_counts[f] -= 1
                            for f in edge_faces[e_rm2]:
                                face_counts[f] -= 1
                            
                            valid = is_valid_unremove(e_un, face_counts)
                            if valid:
                                # Apply
                                removed_set.add(e_rm1)
                                removed_set.add(e_rm2)
                                removed_set.discard(e_un)
                                for f in edge_faces[e_un]:
                                    face_counts[f] += 1
                                current_removed += 1
                            else:
                                # Revert
                                for f in edge_faces[e_rm1]:
                                    face_counts[f] += 1
                                for f in edge_faces[e_rm2]:
                                    face_counts[f] += 1
                            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
            
            if time.time() - last_improvement > 5.0:
                # Reset to best and restart temperature
                removed_set = set(best_removed_set)
                face_counts = calculate_face_counts(removed_set)
                current_removed = best_removed
                T = 1.0
                last_improvement = time.time()
                
        return best_removed_set, best_removed

    start_time = time.time()
    timeout = 145
    
    best_overall_removed_set = None
    best_overall_removed = n_edges
    
    # Multi-start
    while time.time() - start_time < timeout:
        # Initialize randomly
        removed_set = set()
        # Greedy fill: add edges until conflict? Or start with all edges and remove?
        # Start with all edges, then remove to satisfy constraints
        for e in all_edges_list:
            removed_set.add(e)
        
        face_counts = calculate_face_counts(removed_set)
        
        # Greedy repair: remove edges from full faces
        changed = True
        while changed:
            changed = False
            for f in range(n_faces):
                if face_counts[f] == 4:
                    # Remove a random edge from this face
                    e_to_remove = random.choice(faces[f])
                    if e_to_remove not in removed_set:
                        removed_set.add(e_to_remove)
                        for ff in edge_faces[e_to_remove]:
                            face_counts[ff] -= 1
                    changed = True
                    break
        
        # Local Search
        current_timeout = min(10, timeout - (time.time() - start_time))
        if current_timeout < 1:
            break
            
        best_set, best_count = local_search(removed_set, face_counts, time.time(), current_timeout)
        
        if best_count < best_overall_removed:
            best_overall_removed = best_count
            best_overall_removed_set = best_set
            
    # Final Polish
    if best_overall_removed_set is None:
        best_overall_removed_set = set()
        best_overall_removed = n_edges
        
    face_counts = calculate_face_counts(best_overall_removed_set)
    improved = True
    while improved:
        improved = False
        for e in all_edges_list:
            if e in best_overall_removed_set:
                if is_valid_unremove(e, face_counts):
                    apply_unremove(e, best_overall_removed_set, face_counts)
                    best_overall_removed -= 1
                    improved = True

    result = []
    for i in range(n_edges):
        if i not in best_overall_removed_set:
            result.append(list(edges[i]))
            
    return result

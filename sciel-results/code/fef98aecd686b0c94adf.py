import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None
    
    # All edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Precompute face edge indices for fast lookup
    faces = []
    face_id = 0
    face_to_edges = {} # face_id -> set of 4 edge indices
    
    # Generate all faces: 2^(7-2) * C(7,2) = 32 * 21 = 672 faces
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                
                faces.append((idx1, idx2, idx3, idx4))
                face_id += 1

    num_faces = len(faces)
    
    # Map edge index to list of face indices it belongs to
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)
    
    # Precompute for each edge, the set of other edges that share a face with it
    edge_neighbors = [set() for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e1 in face:
            for e2 in face:
                if e1 != e2:
                    edge_neighbors[e1].add(e2)

    def try_greedy(order_seed=None, strategy="random"):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        face_counts = [0] * num_edges  # temp, will be replaced
        
        # Strategy 1: random order
        # Strategy 2: degree-based (prefer edges with fewer conflicts)
        # Strategy 3: weighted by current face saturation
        
        if strategy == "random":
            indices = list(range(num_edges))
            rng.shuffle(indices)
        elif strategy == "degree":
            # Start with edges that are in fewer faces (less constraining)
            indices = sorted(range(num_edges), key=lambda e: len(edge_to_faces[e]))
            rng.shuffle(indices)  # add some randomness
        else:
            indices = list(range(num_edges))
            rng.shuffle(indices)
        
        # face_counts[f] = number of chosen edges in face f
        face_counts = [0] * num_faces
        
        for idx in indices:
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    def local_search(chosen, face_counts, count, deadline, max_iter=5000):
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try all possible single-edge removals and see if we can add 2+ edges back
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue
                
                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
                
                # Find all unchosen edges that become valid (or remain valid)
                # An edge e is "blocked" if any face containing e has exactly 3 chosen edges
                # After removing r_idx, edges that were blocked only because of r_idx become valid
                # Also, edges that were already valid but not chosen remain valid
                
                # Collect candidate edges: unchosen edges that are now valid
                candidates = []
                for e_idx in range(num_edges):
                    if chosen[e_idx]:
                        continue
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)
                
                # Greedily add as many compatible edges as possible from candidates
                added = []
                remaining = list(candidates)
                while remaining and len(added) < 3:  # try to add up to 3 edges
                    best_e = None
                    best_score = -1
                    for e_idx in remaining:
                        # Score: number of faces where this edge would be the 4th (bad) vs enabling others
                        # Simpler: just try to add if compatible with already-added
                        compatible = True
                        for a_idx in added:
                            # Check if e_idx and a_idx share a face with 2+ other chosen edges
                            for f_idx in edge_to_faces[e_idx]:
                                if a_idx in faces[f_idx]:
                                    if face_counts[f_idx] >= 2:
                                        compatible = False
                                        break
                            if not compatible:
                                break
                        if compatible:
                            # Check against existing chosen
                            for f_idx in edge_to_faces[e_idx]:
                                if face_counts[f_idx] == 3:
                                    compatible = False
                                    break
                            if compatible:
                                best_e = e_idx
                                best_score = 1
                                break
                    if best_e is None:
                        break
                    added.append(best_e)
                    remaining.remove(best_e)
                    for f_idx in edge_to_faces[best_e]:
                        face_counts[f_idx] += 1
                    count += 1
                
                if len(added) >= 2:
                    # Commit: keep r_idx removed, keep added edges
                    improved = True
                    break
                else:
                    # Roll back
                    for e_idx in added:
                        chosen[e_idx] = False
                        count -= 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] -= 1
                    chosen[r_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    def kick(chosen, face_counts, count, rng, kick_size=3):
        """Randomly flip some edges to escape local optima"""
        # Remove kick_size random chosen edges
        to_remove = []
        for _ in range(kick_size):
            chosen_edges = [i for i in range(num_edges) if chosen[i]]
            if not chosen_edges:
                break
            r_idx = rng.choice(chosen_edges)
            to_remove.append(r_idx)
        
        for r_idx in to_remove:
            if chosen[r_idx]:
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1
        
        # Try to greedily add back some edges
        indices = list(range(num_edges))
        rng.shuffle(indices)
        for idx in indices:
            if chosen[idx]:
                continue
            can_add = True
            for f_idx in edge_to_faces[idx]:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in edge_to_faces[idx]:
                    face_counts[f_idx] += 1
        
        return count, chosen, face_counts

    attempt = 0
    strategies = ["random", "degree", "random"]
    
    while time.time() < deadline:
        # Use different strategies for different attempts
        strategy = strategies[attempt % len(strategies)]
        count, chosen, face_counts = try_greedy(attempt, strategy)
        
        # Local search
        count, chosen, face_counts = local_search(chosen, face_counts, count, deadline)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen[:]
        
        # If we have time, try a kick and restart local search
        if time.time() < deadline - 10:
            rng = random.Random(attempt + 1000)
            count_k, chosen_k, fc_k = kick(chosen, face_counts, count, rng, kick_size=2)
            count_k, chosen_k, fc_k = local_search(chosen_k, fc_k, count_k, deadline, max_iter=2000)
            if count_k > best_count:
                best_count = count_k
                best_chosen = chosen_k[:]
        
        attempt += 1
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out

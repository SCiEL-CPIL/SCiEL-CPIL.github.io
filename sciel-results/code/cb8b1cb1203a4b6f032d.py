import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None
    
    # All edges: (u, v, dim) where u < v
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)  # 448
    
    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Precompute face edge indices
    # For each edge, the set of faces (4-cycles) it belongs to
    # A face is determined by an edge (dim i) and another dimension j != i
    # The face contains 4 edges: the original, and 3 others
    # We store for each edge, the list of (i1, i2, i3) tuples where these are the other 3 edges in the face
    
    # Actually, let's precompute for each edge, which other edges share a face with it
    # And for local search, we need to know if adding an edge would complete a face
    
    # face_partners[idx] = list of sets, where each set contains the indices of the other 3 edges in a face with idx
    # But for the C4-free condition: a face is forbidden if all 4 edges are selected
    # So when considering adding edge idx, we need to check: for each face containing idx,
    # are the other 3 edges already selected?
    
    face_other_edges = [[] for _ in range(num_edges)]
    
    for idx, (u, v, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_tuple_to_idx[e1]
            idx2 = edge_tuple_to_idx[e2]
            idx3 = edge_tuple_to_idx[e3]
            face_other_edges[idx].append((idx1, idx2, idx3))
    
    # For local search, we need to quickly check if removing an edge unblocks others
    # And if adding an edge blocks others
    
    def try_greedy(order_seed=None):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        for idx in indices:
            can_add = True
            for i1, i2, i3 in face_other_edges[idx]:
                if chosen[i1] and chosen[i2] and chosen[i3]:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
        
        return count, chosen
    
    def local_search(chosen, count, rng, max_iterations=5000):
        """Local search: try to add unselected edges, remove selected edges that allow more additions."""
        chosen = list(chosen)
        count = count
        deadline_ls = time.time() + 2  # max 2 seconds per local search
        
        while time.time() < deadline_ls and max_iterations > 0:
            max_iterations -= 1
            improved = False
            
            # Try to add each unselected edge
            for idx in range(num_edges):
                if chosen[idx]:
                    continue
                can_add = True
                for i1, i2, i3 in face_other_edges[idx]:
                    if chosen[i1] and chosen[i2] and chosen[i3]:
                        can_add = False
                        break
                if can_add:
                    chosen[idx] = True
                    count += 1
                    improved = True
            
            # Try remove-1-add-1 swaps: remove a selected edge, see if we can add more
            if not improved:
                # Try removing each selected edge and see if net gain is positive
                for idx in range(num_edges):
                    if not chosen[idx]:
                        continue
                    # Temporarily remove
                    chosen[idx] = False
                    temp_count = count - 1
                    
                    # Try to add all unselected edges that are now possible
                    added = 0
                    newly_added = []
                    for jdx in range(num_edges):
                        if chosen[jdx]:
                            continue
                        can_add = True
                        for i1, i2, i3 in face_other_edges[jdx]:
                            if chosen[i1] and chosen[i2] and chosen[i3]:
                                can_add = False
                                break
                        if can_add:
                            chosen[jdx] = True
                            temp_count += 1
                            added += 1
                            newly_added.append(jdx)
                    
                    if temp_count > count:
                        # Keep the change
                        count = temp_count
                        improved = True
                    else:
                        # Revert
                        chosen[idx] = True
                        for jdx in newly_added:
                            chosen[jdx] = False
                    
                    if improved:
                        break  # Restart the loop with new state
        
        return count, chosen
    
    attempt = 0
    while time.time() < deadline:
        count, chosen = try_greedy(attempt)
        
        # Local search to improve
        rng = random.Random(attempt * 1000 + 7)
        count, chosen = local_search(chosen, count, rng)
        
        if count > best_count:
            best_count = count
            best_chosen = chosen
        
        attempt += 1
        
        # Occasionally do a larger perturbation
        if attempt % 5 == 0 and best_chosen is not None:
            rng = random.Random(attempt * 7919 + 13)
            perturbed = list(best_chosen)
            # Remove K random selected edges
            selected = [i for i in range(num_edges) if perturbed[i]]
            K = rng.randint(3, 15)
            if len(selected) >= K:
                to_remove = rng.sample(selected, K)
                for idx in to_remove:
                    perturbed[idx] = False
                # Greedy fill
                for idx in range(num_edges):
                    if perturbed[idx]:
                        continue
                    can_add = True
                    for i1, i2, i3 in face_other_edges[idx]:
                        if perturbed[i1] and perturbed[i2] and perturbed[i3]:
                            can_add = False
                            break
                    if can_add:
                        perturbed[idx] = True
                
                pcount = sum(perturbed)
                rng2 = random.Random(attempt * 31 + 3)
                pcount, perturbed = local_search(perturbed, pcount, rng2)
                
                if pcount > best_count:
                    best_count = pcount
                    best_chosen = perturbed
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out

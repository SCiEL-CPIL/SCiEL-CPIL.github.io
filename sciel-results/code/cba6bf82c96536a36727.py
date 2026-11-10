import random
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
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
    
    # Build adjacency: for each edge, which other edges share a face with it
    # This helps speed up local search by only checking relevant edges
    face_neighbors = [set() for _ in range(num_edges)]
    for idx in range(num_edges):
        for i1, i2, i3 in face_other_edges[idx]:
            face_neighbors[idx].add(i1)
            face_neighbors[idx].add(i2)
            face_neighbors[idx].add(i3)
    
    def can_add_edge(idx, chosen):
        """Check if edge idx can be added without completing a face."""
        for i1, i2, i3 in face_other_edges[idx]:
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    def try_greedy(order_seed=None):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        for idx in indices:
            if can_add_edge(idx, chosen):
                chosen[idx] = True
                count += 1
        
        return count, chosen
    
    def greedy_fill(chosen):
        """Try to add all unselected edges that can be added. Returns number added."""
        added = 0
        for idx in range(num_edges):
            if not chosen[idx] and can_add_edge(idx, chosen):
                chosen[idx] = True
                added += 1
        return added
    
    def local_search(chosen, count, rng, max_iterations=3000):
        """Local search: try to add unselected edges, remove selected edges that allow more additions."""
        chosen = list(chosen)
        count = count
        deadline_ls = time.time() + 5  # max 5 seconds per local search
        
        while time.time() < deadline_ls and max_iterations > 0:
            max_iterations -= 1
            improved = False
            
            # Try to add each unselected edge
            for idx in range(num_edges):
                if chosen[idx]:
                    continue
                if can_add_edge(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    improved = True
            
            # Try remove-1-add-1 swaps: remove a selected edge, see if we can add more
            if not improved:
                # Try removing each selected edge and see if net gain is positive
                selected_edges = [i for i in range(num_edges) if chosen[i]]
                rng.shuffle(selected_edges)
                
                for idx in selected_edges:
                    # Temporarily remove
                    chosen[idx] = False
                    temp_count = count - 1
                    
                    # Try to add all unselected edges that are now possible
                    # Only check edges that share a face with idx (they're the only ones that could be unblocked)
                    newly_added = []
                    for jdx in face_neighbors[idx]:
                        if not chosen[jdx] and can_add_edge(jdx, chosen):
                            chosen[jdx] = True
                            temp_count += 1
                            newly_added.append(jdx)
                    
                    # Also try adding other edges that might now be possible
                    # (edges that don't share a face with idx but were blocked by edges that shared a face)
                    # This is more complex, so we'll do a second pass
                    for jdx in range(num_edges):
                        if not chosen[jdx] and can_add_edge(jdx, chosen):
                            chosen[jdx] = True
                            temp_count += 1
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
            best_chosen = list(chosen)
        
        attempt += 1
        
        # Perturbation: remove K random selected edges, greedy fill, then local search
        if best_chosen is not None and attempt % 3 == 0:
            rng = random.Random(attempt * 7919 + 13)
            perturbed = list(best_chosen)
            
            # Use larger kick sizes (K=4-20) as per cell findings
            selected = [i for i in range(num_edges) if perturbed[i]]
            K = rng.randint(4, 20)
            if len(selected) >= K:
                to_remove = rng.sample(selected, K)
                for idx in to_remove:
                    perturbed[idx] = False
                
                # Greedy fill
                pcount = sum(perturbed)
                pcount += greedy_fill(perturbed)
                
                # Local search on perturbed state
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

import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    all_edges = []
    edge_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                edge_idx[(x, y)] = len(all_edges) - 1
    
    # Precompute faces: for each face, the 4 edges
    # A face is determined by vertex x and two distinct bit positions i < j
    faces = []
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                vi = x ^ (1 << i)
                vj = x ^ (1 << j)
                vij = x ^ (1 << i) ^ (1 << j)
                e1 = (min(x, vi), max(x, vi))
                e2 = (min(vi, vij), max(vi, vij))
                e3 = (min(vij, vj), max(vij, vj))
                e4 = (min(x, vj), max(x, vj))
                faces.append((edge_idx[e1], edge_idx[e2], edge_idx[e3], edge_idx[e4]))
    
    n_faces = len(faces)
    
    # For each edge, precompute which faces contain it
    edge_faces = [[] for _ in range(n_edges)]
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    # Precompute for each edge, the companion edges in each face
    # For edge e in face f, the other 3 edges in f are companions
    edge_companions = [[] for _ in range(n_edges)]
    for fi, face in enumerate(faces):
        for ei in face:
            for ej in face:
                if ei != ej:
                    edge_companions[ei].append((fi, ej))
    
    # Remove duplicates in edge_companions (each face appears once per edge)
    for ei in range(n_edges):
        edge_companions[ei] = list(set(edge_companions[ei]))
    
    def greedy_random_fast(rng):
        """One randomized greedy pass with fast conflict checks."""
        order = list(range(n_edges))
        rng.shuffle(order)
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        count = 0
        
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                selected[ei] = True
                count += 1
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        return selected, face_counts, count
    
    def greedy_degree_based(rng):
        """Greedy with degree-based ordering for diversity."""
        # Assign random weights to edges, sort by weight + degree heuristic
        weights = [rng.random() for _ in range(n_edges)]
        # Compute initial degree (number of faces each edge is in)
        degree = [len(edge_faces[ei]) for ei in range(n_edges)]
        
        # Sort by weight * degree (higher degree edges get more weight)
        order = sorted(range(n_edges), key=lambda ei: weights[ei] * (1.0 / (1 + degree[ei])), reverse=True)
        
        selected = [False] * n_edges
        face_counts = [0] * n_faces
        count = 0
        
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                selected[ei] = True
                count += 1
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        return selected, face_counts, count
    
    def local_search_additions(selected, face_counts, rng, time_limit):
        """Try adding random unselected edges that don't create conflicts."""
        n_selected = sum(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for _ in range(1000):
                ei = rng.randint(0, n_edges - 1)
                if selected[ei]:
                    continue
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                    n_selected += 1
                    improved = True
                    break
        return selected, face_counts, n_selected
    
    def local_search_swaps(selected, face_counts, rng, time_limit):
        """Try 1-for-1 swaps: remove one edge, add another."""
        n_selected = sum(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for _ in range(500):
                # Pick a random selected edge to remove
                ei_remove = rng.randint(0, n_edges - 1)
                if not selected[ei_remove]:
                    continue
                
                # Temporarily remove it
                selected[ei_remove] = False
                for fi in edge_faces[ei_remove]:
                    face_counts[fi] -= 1
                
                # Try to add a random unselected edge
                ei_add = rng.randint(0, n_edges - 1)
                if selected[ei_add]:
                    # Restore and try again
                    selected[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        face_counts[fi] += 1
                    continue
                
                can_add = True
                for fi in edge_faces[ei_add]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                
                if can_add:
                    selected[ei_add] = True
                    for fi in edge_faces[ei_add]:
                        face_counts[fi] += 1
                    # Keep the removal, accept the swap
                    improved = True
                    break
                else:
                    # Restore
                    selected[ei_remove] = True
                    for fi in edge_faces[ei_remove]:
                        face_counts[fi] += 1
        
        return selected, face_counts, n_selected
    
    def local_search_2swap(selected, face_counts, rng, time_limit):
        """Try removing 2 edges and adding up to 2 edges."""
        n_selected = sum(selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for _ in range(200):
                # Pick two random selected edges to remove
                ei1 = rng.randint(0, n_edges - 1)
                ei2 = rng.randint(0, n_edges - 1)
                if not selected[ei1] or ei1 == ei2 or not selected[ei2]:
                    continue
                
                # Temporarily remove both
                selected[ei1] = False
                selected[ei2] = False
                for fi in edge_faces[ei1]:
                    face_counts[fi] -= 1
                for fi in edge_faces[ei2]:
                    face_counts[fi] -= 1
                
                # Try to add two random unselected edges
                added = []
                for _ in range(2):
                    ei_add = rng.randint(0, n_edges - 1)
                    if selected[ei_add]:
                        continue
                    can_add = True
                    for fi in edge_faces[ei_add]:
                        if face_counts[fi] == 3:
                            can_add = False
                            break
                    if can_add:
                        selected[ei_add] = True
                        for fi in edge_faces[ei_add]:
                            face_counts[fi] += 1
                        added.append(ei_add)
                
                if len(added) == 2:
                    # Accept the 2-for-2 swap
                    improved = True
                    break
                else:
                    # Restore
                    for ei in added:
                        selected[ei] = False
                        for fi in edge_faces[ei]:
                            face_counts[fi] -= 1
                    selected[ei1] = True
                    selected[ei2] = True
                    for fi in edge_faces[ei1]:
                        face_counts[fi] += 1
                    for fi in edge_faces[ei2]:
                        face_counts[fi] += 1
        
        return selected, face_counts, n_selected
    
    def kick_and_optimize(selected, face_counts, rng, kick_size, time_limit):
        """Randomly perturb the solution and re-optimize."""
        n_selected = sum(selected)
        
        # Kick: randomly remove kick_size edges
        removed = []
        for _ in range(kick_size):
            ei = rng.randint(0, n_edges - 1)
            if selected[ei]:
                selected[ei] = False
                for fi in edge_faces[ei]:
                    face_counts[fi] -= 1
                removed.append(ei)
                n_selected -= 1
        
        # Re-optimize: add all possible edges
        for ei in range(n_edges):
            if not selected[ei]:
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                    n_selected += 1
        
        # Local search
        selected, face_counts, n_selected = local_search_additions(selected, face_counts, rng, time.time() + 1.0)
        selected, face_counts, n_selected = local_search_swaps(selected, face_counts, rng, time.time() + 1.0)
        
        return selected, face_counts, n_selected
    
    def final_polish(selected, face_counts):
        """Add any remaining valid edges."""
        n_selected = sum(selected)
        for ei in range(n_edges):
            if not selected[ei]:
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                    n_selected += 1
        return selected, face_counts, n_selected
    
    best_count = 0
    best_selected = None
    best_face_counts = None
    
    rng = random.Random(42)
    
    # Multiple restarts with different strategies
    n_restarts = 0
    while time.time() < deadline - 10.0:
        restart_start = time.time()
        remaining = deadline - 10.0 - time.time()
        if remaining < 0.5:
            break
        
        # Choose strategy based on restart number
        if n_restarts % 3 == 0:
            selected, face_counts, n_selected = greedy_random_fast(rng)
        elif n_restarts % 3 == 1:
            selected, face_counts, n_selected = greedy_degree_based(rng)
        else:
            selected, face_counts, n_selected = greedy_random_fast(rng)
            # Kick and optimize
            if time.time() < deadline - 10.0:
                selected, face_counts, n_selected = kick_and_optimize(
                    selected, face_counts, rng, 
                    kick_size=min(5, n_selected // 10),
                    time_limit=time.time() + min(remaining * 0.3, 3.0)
                )
        
        # Local search with time limit
        if time.time() < deadline - 10.0:
            local_time = min(remaining * 0.4, 5.0)
            selected, face_counts, n_selected = local_search_additions(
                selected, face_counts, rng, time.time() + local_time * 0.3
            )
            selected, face_counts, n_selected = local_search_swaps(
                selected, face_counts, rng, time.time() + local_time * 0.4
            )
            if time.time() < deadline - 10.0:
                selected, face_counts, n_selected = local_search_2swap(
                    selected, face_counts, rng, time.time() + local_time * 0.3
                )
        
        # Final polish
        selected, face_counts, n_selected = final_polish(selected, face_counts)
        
        if n_selected > best_count:
            best_count = n_selected
            best_selected = selected[:]
            best_face_counts = face_counts[:]
        
        n_restarts += 1
    
    # If we have time, do more aggressive search on the best solution
    if time.time() < deadline - 5.0 and best_selected is not None:
        # Try kick-and-optimize on the best solution
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            selected = best_selected[:]
            face_counts = best_face_counts[:]
            n_selected = sum(selected)
            
            selected, face_counts, n_selected = kick_and_optimize(
                selected, face_counts, rng,
                kick_size=3,
                time_limit=time.time() + 2.0
            )
            selected, face_counts, n_selected = final_polish(selected, face_counts)
            
            if n_selected > best_count:
                best_count = n_selected
                best_selected = selected[:]
                best_face_counts = face_counts[:]
    
    # Convert best_selected to list of edges
    result = []
    for ei in range(n_edges):
        if best_selected[ei]:
            result.append(list(all_edges[ei]))
    
    return result

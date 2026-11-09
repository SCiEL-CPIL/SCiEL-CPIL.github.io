import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 (7-dimensional hypercube).
    
    Strategy:
    1. Precompute face-edge incidences.
    2. Complementary deletion: Start with all edges, remove edges to break C4s using a heuristic.
    3. Local search: 
       - Greedy add edges.
       - Swap search (remove one, add one) to escape local optima.
       - Perturbation (kick) to jump to new basins.
    4. Multiple restarts with different seeds.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    d = 7
    n_vertices = 1 << d  # 128
    n_edges_total = d * (1 << (d - 1))  # 448
    
    # Precompute all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    n_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Build face list
    faces = []
    for base in range(n_vertices):
        for d1 in range(d):
            if (base >> d1) & 1:
                continue
            for d2 in range(d1 + 1, d):
                if (base >> d2) & 1:
                    continue
                v0 = base
                v1 = base ^ (1 << d1)
                v2 = base ^ (1 << d2)
                v3 = base ^ (1 << d1) ^ (1 << d2)
                
                e1 = (min(v0, v1), max(v0, v1))
                e2 = (min(v0, v2), max(v0, v2))
                e3 = (min(v1, v3), max(v1, v3))
                e4 = (min(v2, v3), max(v2, v3))
                
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)
    
    n_faces = len(faces)
    
    # For each edge, list of faces it belongs to
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    # Convert to numpy for potential vectorized checks if needed, 
    # but we will primarily use lists for the dynamic local search state.
    
    def time_left():
        return deadline - time.time()
    
    def count_edges(sol):
        return sum(1 for e in range(n_edges) if sol[e])
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        for fi in range(n_faces):
            if all(sol[e] for e in faces[fi]):
                return False
        return True

    def complementary_deletion(seed):
        """
        Start with all edges, remove edges to break all C4s.
        Heuristic: Process bad faces in random order. For a bad face, 
        pick an edge to remove. To improve, we can pick the edge that 
        is 'least useful' or just random. 
        """
        rng = random.Random(seed)
        sol = [True] * n_edges
        face_counts = [4] * n_faces
        bad_faces = set(range(n_faces))
        
        # Order faces randomly to diversify
        face_order = list(range(n_faces))
        rng.shuffle(face_order)
        
        for fi in face_order:
            if time_left() < 5:
                break
            if face_counts[fi] < 4:
                continue
            
            # Face is bad. Pick an edge to remove.
            # Strategy: Pick the edge in this face that belongs to the fewest 
            # OTHER bad faces (to minimize collateral damage to other potential solutions? 
            # Actually, we want to remove an edge that breaks this face. 
            # Removing an edge that is in many other bad faces might be good 
            # if it breaks those too, but those are already counted as bad.
            # Let's just pick randomly from the 4 edges to maintain diversity.
            # Or, pick the edge with the highest 'removal value'.
            
            candidates = faces[fi]
            # Random choice is usually sufficient for the initial phase
            ei = rng.choice(candidates)
            
            if sol[ei]:
                sol[ei] = False
                for f2 in edge_faces[ei]:
                    face_counts[f2] -= 1
                    if face_counts[f2] == 3:
                        bad_faces.discard(f2)
        
        return sol, face_counts

    def local_search(sol, face_counts, seed):
        """
        Improve solution via local search.
        1. Try to add any possible edge.
        2. If stuck, try swaps (remove one, add one).
        3. If stuck, perturb (remove k edges).
        """
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = list(face_counts)
        
        # Helper: can we add edge ei?
        def can_add(ei):
            if sol[ei]:
                return False
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    return False
            return True
            
        # Helper: can we swap ei_out with ei_in?
        def can_swap(ei_out, ei_in):
            if not sol[ei_out] or sol[ei_in]:
                return False
            # Simulate removal of ei_out
            # This decreases count for faces in edge_faces[ei_out]
            # We need to check if ei_in can be added in this new state
            # ei_in can be added if for all f in edge_faces[ei_in], 
            # count(f) - (1 if f in edge_faces[ei_out] else 0) < 3
            
            # Pre-check: if ei_out and ei_in share a face, the count drops by 1 for that face.
            # If they don't share a face, count stays same for faces of ei_in.
            
            shared = set(edge_faces[ei_out]) & set(edge_faces[ei_in])
            
            for fi in edge_faces[ei_in]:
                current_count = face_counts[fi]
                if fi in shared:
                    current_count -= 1
                if current_count >= 3:
                    return False
            return True

        # Helper: perform add
        def do_add(ei):
            sol[ei] = True
            for fi in edge_faces[ei]:
                face_counts[fi] += 1

        # Helper: perform remove
        def do_remove(ei):
            sol[ei] = False
            for fi in edge_faces[ei]:
                face_counts[fi] -= 1

        # Helper: perform swap
        def do_swap(ei_out, ei_in):
            sol[ei_out] = False
            sol[ei_in] = True
            # Update counts
            for fi in edge_faces[ei_out]:
                face_counts[fi] -= 1
            for fi in edge_faces[ei_in]:
                face_counts[fi] += 1

        max_iter = 5000
        iter_count = 0
        improved = True
        
        while improved and time_left() > 10 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Greedy Add
            addable = []
            for ei in range(n_edges):
                if can_add(ei):
                    addable.append(ei)
            
            if addable:
                rng.shuffle(addable)
                # Add as many as possible in this pass
                for ei in addable:
                    if can_add(ei):
                        do_add(ei)
                        improved = True
                # If we added anything, restart the loop to check for more
                continue
            
            # Phase 2: Swap Search
            # Find edges that can be swapped
            present = [e for e in range(n_edges) if sol[e]]
            missing = [e for e in range(n_edges) if not sol[e]]
            
            if present and missing:
                # To speed up, we can limit the number of swaps tried per iteration
                # Randomly sample a subset
                sample_present = rng.sample(present, min(len(present), 50))
                sample_missing = rng.sample(missing, min(len(missing), 50))
                
                swap_found = False
                for ei_out in sample_present:
                    if time_left() < 10:
                        break
                    for ei_in in sample_missing:
                        if can_swap(ei_out, ei_in):
                            do_swap(ei_out, ei_in)
                            improved = True
                            swap_found = True
                            break
                    if swap_found:
                        break
                
                if swap_found:
                    # After a swap, we might be able to add more edges
                    # Restart loop to do Phase 1 again
                    continue
            
            # Phase 3: Perturbation (Kick)
            # If stuck, remove a few random edges to break the local optimum
            if time_left() > 20:
                # Remove k random edges
                k = rng.choice([2, 3, 5])
                if len(present) > k:
                    to_remove = rng.sample(present, k)
                    for ei in to_remove:
                        do_remove(ei)
                    improved = True
                    # Restart loop
                    continue
            
            # If we get here, we are stuck and time is low or no kick allowed
            break
            
        return sol, face_counts

    best_count = 0
    best_solution = None
    
    # Initial deterministic greedy baseline (optional, but good for safety)
    # Let's rely on complementary deletion + local search
    
    attempts = 0
    # We will run multiple restarts
    while time_left() > 15:
        attempts += 1
        seed = attempts * 7919 + 42
        
        # 1. Complementary Deletion
        sol, face_counts = complementary_deletion(seed)
        
        # 2. Local Search
        sol, face_counts = local_search(sol, face_counts, seed)
        
        # 3. Verify and Count
        if is_valid(sol):
            cnt = count_edges(sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Optional: Second pass local search with different seed for perturbation
        if time_left() > 30 and best_solution:
             # Perturb best solution
             rng2 = random.Random(seed + 123)
             sol2 = list(best_solution)
             fc2 = [0] * n_faces
             for ei in range(n_edges):
                 if sol2[ei]:
                     for fi in edge_faces[ei]:
                         fc2[fi] += 1
                         
             # Remove 3 random edges
             present2 = [e for e in range(n_edges) if sol2[e]]
             if len(present2) > 3:
                 to_remove2 = rng2.sample(present2, 3)
                 for ei in to_remove2:
                     sol2[ei] = False
                     for fi in edge_faces[ei]:
                         fc2[fi] -= 1
             
             sol2, fc2 = local_search(sol2, fc2, seed + 456)
             if is_valid(sol2):
                 cnt2 = count_edges(sol2)
                 if cnt2 > best_count:
                     best_count = cnt2
                     best_solution = list(sol2)

    # Final fallback if something went wrong
    if best_solution is None:
        # Simple greedy
        sol = [False] * n_edges
        for ei in range(n_edges):
            ok = True
            for fi in edge_faces[ei]:
                # Check if other 3 edges are present
                cnt = 0
                for e2 in faces[fi]:
                    if e2 != ei and sol[e2]:
                        cnt += 1
                if cnt == 3:
                    ok = False
                    break
            if ok:
                sol[ei] = True
        best_solution = sol
        best_count = count_edges(sol)

    # Convert solution to list of edges
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result


import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 (7-dimensional hypercube).
    
    Strategy:
    1. Precompute face-edge incidences.
    2. Use a fast greedy construction with randomized orderings.
    3. Local search: repair bad faces, then add edges greedily.
    4. Multiple restarts with different random seeds.
    5. Time-bounded execution with safety margin.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (kill at 150s)
    
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
                e1 = (base, base ^ (1 << d1))
                e2 = (base, base ^ (1 << d2))
                e3 = (base ^ (1 << d1), base ^ (1 << d1) ^ (1 << d2))
                e4 = (base ^ (1 << d2), base ^ (1 << d1) ^ (1 << d2))
                
                e1 = (min(e1), max(e1))
                e2 = (min(e2), max(e2))
                e3 = (min(e3), max(e3))
                e4 = (min(e4), max(e4))
                
                face_edges = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                faces.append(face_edges)
    
    n_faces = len(faces)
    
    # For each edge, list of faces it belongs to
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for ei in face:
            edge_faces[ei].append(fi)
    
    # Convert to numpy for speed
    faces_np = np.array(faces, dtype=np.int32)
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        sol_arr = np.array(sol, dtype=np.bool_)
        for fi in range(n_faces):
            if np.all(sol_arr[faces_np[fi]]):
                return False
        return True
    
    def count_edges(sol):
        return sum(1 for e in range(n_edges) if sol[e])
    
    def greedy_initial(seed=None):
        """Generate a greedy initial solution with random edge order."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        
        order = list(range(n_edges))
        rng.shuffle(order)
        
        for ei in order:
            can_add = True
            for fi in edge_faces[ei]:
                if face_counts[fi] == 3:
                    can_add = False
                    break
            if can_add:
                sol[ei] = True
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        return sol
    
    def local_search(sol, seed=None, max_iter=10000):
        """Improve solution via local search with simulated annealing perturbation."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        # Find bad faces
        bad_faces = set()
        for fi in range(n_faces):
            if face_counts[fi] == 4:
                bad_faces.add(fi)
        
        improved = True
        iter_count = 0
        while improved and time_left() > 10 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces:
                # Remove an edge from a bad face - choose edge in most bad faces
                bad_face_list = list(bad_faces)
                rng.shuffle(bad_face_list)
                
                best_removal = None
                best_score = -1
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt = 0
                            for f2 in edge_faces[ei]:
                                if f2 in bad_faces:
                                    cnt += 1
                            if cnt > best_score:
                                best_score = cnt
                                best_removal = (fi, ei)
                
                if best_removal:
                    fi, ei = best_removal
                    sol[ei] = False
                    for f2 in edge_faces[ei]:
                        face_counts[f2] -= 1
                        if face_counts[f2] == 3 and f2 in bad_faces:
                            bad_faces.discard(f2)
                    if face_counts[fi] < 4:
                        bad_faces.discard(fi)
                    improved = True
            else:
                # No bad faces, try to add edges
                addable = []
                for ei in range(n_edges):
                    if not sol[ei]:
                        can_add = True
                        for fi in edge_faces[ei]:
                            if face_counts[fi] >= 3:
                                can_add = False
                                break
                        if can_add:
                            addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    for ei in addable:
                        if time_left() < 10:
                            break
                        can_add = True
                        for fi in edge_faces[ei]:
                            if face_counts[fi] >= 3:
                                can_add = False
                                break
                        if can_add:
                            sol[ei] = True
                            for fi in edge_faces[ei]:
                                face_counts[fi] += 1
                            improved = True
                else:
                    # Stuck - try perturbation: randomly remove a few edges and re-add
                    if time_left() > 20:
                        selected = [e for e in range(n_edges) if sol[e]]
                        if len(selected) > 10:
                            to_remove = rng.sample(selected, min(5, len(selected) // 10))
                            for ei in to_remove:
                                sol[ei] = False
                                for f2 in edge_faces[ei]:
                                    face_counts[f2] -= 1
                                    if f2 in bad_faces:
                                        bad_faces.discard(f2)
                            improved = True
    
        return sol
    
    def deterministic_greedy():
        """The original seed's deterministic greedy approach."""
        chosen = set()
        sol = [False] * n_edges
        for x in range(n_vertices):
            for i in range(d):
                y = x ^ (1 << i)
                if x < y:
                    ok = True
                    for j in range(d):
                        if j != i:
                            c = x ^ (1 << j)
                            e = y ^ (1 << j)
                            if all(p in chosen for p in ((min(x, c), max(x, c)), (min(y, e), max(y, e)), (min(c, e), max(c, e)))):
                                ok = False
                                break
                    if ok:
                        chosen.add((x, y))
                        ei = edge_to_idx[(x, y)]
                        sol[ei] = True
        return sol
    
    best_count = 0
    best_solution = None
    
    # First, run the deterministic greedy as a baseline
    sol = deterministic_greedy()
    cnt = count_edges(sol)
    if is_valid(sol):
        best_count = cnt
        best_solution = list(sol)
    
    # Then run multiple randomized restarts
    attempts = 0
    while time_left() > 15:
        attempts += 1
        seed = attempts * 7919
        
        # Generate initial solution
        sol = greedy_initial(seed)
        
        # Local search
        sol = local_search(sol, seed)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 30:
            break
    
    # Convert solution to list of edges
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result

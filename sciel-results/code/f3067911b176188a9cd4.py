import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Combined approach:
    1. Use the structural idea of the seed (deterministic greedy scan) as a strong baseline.
    2. Use the refinement machinery of the current program (local search with face-count tracking).
    3. Add a simulated-annealing-style random perturbation step to escape local optima.
    4. Multiple restarts with different seeds, keeping the best valid solution.
    5. Vectorized numpy operations for speed.
    6. Swap moves in local search (remove one, add another).
    7. Final polish pass with tighter convergence.
    """
    start_time = time.time()
    deadline = start_time + 138  # 138s safety margin (kill at 150s)
    
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
    edge_faces_np = [np.array(edge_faces[ei], dtype=np.int32) for ei in range(n_edges)]
    faces_np = np.array(faces, dtype=np.int32)
    
    # Precompute: for each face, the 4 edge indices as numpy array
    # faces_np is already shape (n_faces, 4)
    
    best_count = 0
    best_solution = None
    
    def time_left():
        return deadline - time.time()
    
    def is_valid(sol):
        """Check if solution has no face with all 4 edges."""
        sol_arr = np.array(sol, dtype=np.bool_)
        # Vectorized check: for each face, are all 4 edges selected?
        face_all = np.all(sol_arr[faces_np], axis=1)
        return not np.any(face_all)
    
    def count_edges(sol):
        return int(np.sum(np.array(sol, dtype=np.int8)))
    
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
    
    def greedy_initial(seed=None):
        """Generate a greedy initial solution with random edge order."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = np.zeros(n_faces, dtype=np.int8)
        
        order = list(range(n_edges))
        rng.shuffle(order)
        
        for ei in order:
            ef = edge_faces_np[ei]
            if np.any(face_counts[ef] >= 3):
                continue
            sol[ei] = True
            face_counts[ef] += 1
        
        return sol
    
    def compute_face_counts(sol):
        """Compute face counts from a solution using numpy."""
        sol_arr = np.array(sol, dtype=np.int8)
        # face_counts[fi] = sum of sol_arr over the 4 edges of face fi
        fc = np.sum(sol_arr[faces_np], axis=1)
        return fc
    
    def local_search(sol, seed=None, max_iter=10000):
        """Improve solution via local search with swap moves and perturbation."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = compute_face_counts(sol)
        
        # Find bad faces (count == 4)
        bad_faces_set = set(np.where(face_counts == 4)[0].tolist())
        
        improved = True
        iter_count = 0
        consecutive_no_improve = 0
        
        while improved and time_left() > 12 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces_set:
                # Remove an edge from a bad face - choose edge in most bad faces
                bad_face_list = list(bad_faces_set)
                rng.shuffle(bad_face_list)
                
                best_removal = None
                best_score = -1
                for fi in bad_face_list:
                    for ei in faces[fi]:
                        if sol[ei]:
                            cnt = 0
                            for f2 in edge_faces[ei]:
                                if f2 in bad_faces_set:
                                    cnt += 1
                            if cnt > best_score:
                                best_score = cnt
                                best_removal = ei
                
                if best_removal is not None:
                    ei = best_removal
                    sol[ei] = False
                    ef = edge_faces_np[ei]
                    face_counts[ef] -= 1
                    # Update bad faces
                    for f2 in ef:
                        if face_counts[f2] < 4:
                            bad_faces_set.discard(int(f2))
                    improved = True
                    consecutive_no_improve = 0
            else:
                # No bad faces, try to add edges
                sol_arr = np.array(sol, dtype=np.bool_)
                # Find addable edges: not selected and no face has count >= 3
                addable = []
                for ei in range(n_edges):
                    if sol[ei]:
                        continue
                    ef = edge_faces_np[ei]
                    if np.any(face_counts[ef] >= 3):
                        continue
                    addable.append(ei)
                
                if addable:
                    rng.shuffle(addable)
                    for ei in addable:
                        if time_left() < 12:
                            break
                        ef = edge_faces_np[ei]
                        if np.any(face_counts[ef] >= 3):
                            continue
                        sol[ei] = True
                        face_counts[ef] += 1
                        improved = True
                        consecutive_no_improve = 0
                else:
                    # Stuck - try swap moves first, then perturbation
                    if time_left() > 25:
                        # Try swap: remove one selected edge, add one unselected edge
                        selected = [e for e in range(n_edges) if sol[e]]
                        unselected = [e for e in range(n_edges) if not sol[e]]
                        
                        if selected and unselected:
                            # Try a few random swaps
                            swaps_tried = 0
                            max_swaps = min(20, len(selected) // 5)
                            for _ in range(max_swaps):
                                if time_left() < 12:
                                    break
                                ei_remove = rng.choice(selected)
                                ei_add = rng.choice(unselected)
                                if ei_remove == ei_add:
                                    continue
                                
                                # Check if swap is valid: after removing ei_remove, can we add ei_add?
                                ef_remove = edge_faces_np[ei_remove]
                                ef_add = edge_faces_np[ei_add]
                                
                                # Simulate: face_counts after removal
                                fc_sim = face_counts.copy()
                                fc_sim[ef_remove] -= 1
                                
                                # Check if ei_add can be added
                                if np.any(fc_sim[ef_add] >= 3):
                                    continue
                                
                                # Also check: does removing ei_remove create any new bad faces? No, it only decreases counts.
                                # Does adding ei_add create any bad faces?
                                if np.any(fc_sim[ef_add] == 3):
                                    # Would make a face have 4 edges
                                    continue
                                
                                # Perform swap
                                sol[ei_remove] = False
                                sol[ei_add] = True
                                face_counts[ef_remove] -= 1
                                face_counts[ef_add] += 1
                                
                                # Update bad faces
                                for f2 in ef_remove:
                                    if face_counts[f2] < 4:
                                        bad_faces_set.discard(int(f2))
                                for f2 in ef_add:
                                    if face_counts[f2] == 4:
                                        bad_faces_set.add(int(f2))
                                
                                # If swap created bad faces, undo
                                if bad_faces_set:
                                    sol[ei_add] = False
                                    sol[ei_remove] = True
                                    face_counts[ef_add] -= 1
                                    face_counts[ef_remove] += 1
                                    for f2 in ef_add:
                                        if face_counts[f2] < 4:
                                            bad_faces_set.discard(int(f2))
                                    for f2 in ef_remove:
                                        if face_counts[f2] == 4:
                                            bad_faces_set.add(int(f2))
                                    continue
                                
                                improved = True
                                consecutive_no_improve = 0
                                break
                            
                            if not improved:
                                # Perturbation: remove 3-6 random edges
                                if len(selected) > 15:
                                    n_remove = rng.randint(3, 6)
                                    to_remove = rng.sample(selected, min(n_remove, len(selected) // 8))
                                    for ei in to_remove:
                                        sol[ei] = False
                                        ef = edge_faces_np[ei]
                                        face_counts[ef] -= 1
                                        for f2 in ef:
                                            if face_counts[f2] < 4:
                                                bad_faces_set.discard(int(f2))
                                    improved = True
                                    consecutive_no_improve = 0
                                else:
                                    consecutive_no_improve += 1
                                    if consecutive_no_improve > 5:
                                        break
                        else:
                            consecutive_no_improve += 1
                            if consecutive_no_improve > 5:
                                break
    
        return sol
    
    def polish(sol, seed=None):
        """Final high-precision polish: try all single-edge additions and removals."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = compute_face_counts(sol)
        
        # Try to add any edge that can be added
        changed = True
        while changed and time_left() > 15:
            changed = False
            for ei in range(n_edges):
                if sol[ei]:
                    continue
                ef = edge_faces_np[ei]
                if np.any(face_counts[ef] >= 3):
                    continue
                sol[ei] = True
                face_counts[ef] += 1
                changed = True
        
        return sol
    
    # First, get the deterministic greedy baseline
    sol = deterministic_greedy()
    cnt = count_edges(sol)
    if is_valid(sol):
        best_count = cnt
        best_solution = list(sol)
    
    # Multiple restarts with different seeds
    attempts = 0
    while time_left() > 20:
        attempts += 1
        seed = attempts * 7919 + 42
        
        # Alternate between different strategies
        if attempts % 5 == 1:
            sol = greedy_initial(seed)
        elif attempts % 5 == 2:
            sol = greedy_initial(seed + 100000)
        elif attempts % 5 == 3:
            # Perturbed best solution
            if best_solution:
                sol = list(best_solution)
                rng = random.Random(seed)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(4, len(selected) // 8))
                    for ei in to_remove:
                        sol[ei] = False
            else:
                sol = greedy_initial(seed)
        elif attempts % 5 == 4:
            # Perturbed best with more removals
            if best_solution:
                sol = list(best_solution)
                rng = random.Random(seed + 777)
                selected = [e for e in range(n_edges) if sol[e]]
                if selected:
                    to_remove = rng.sample(selected, min(6, len(selected) // 6))
                    for ei in to_remove:
                        sol[ei] = False
            else:
                sol = greedy_initial(seed)
        else:
            # Hybrid: start from greedy, then perturb
            sol = greedy_initial(seed + 200000)
            rng = random.Random(seed + 300000)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(3, len(selected) // 10))
                for ei in to_remove:
                    sol[ei] = False
        
        # Local search
        sol = local_search(sol, seed)
        
        # Polish
        sol = polish(sol, seed + 55555)
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 35:
            break
    
    # Final fallback: if we haven't found anything good, use the deterministic greedy
    if best_solution is None or best_count < 250:
        chosen = set()
        out = []
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
                        out.append([x, y])
        return out
    
    # Convert solution to list of edges
    result = []
    for ei in range(n_edges):
        if best_solution[ei]:
            result.append(list(edges[ei]))
    
    return result

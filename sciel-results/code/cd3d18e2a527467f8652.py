import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 (7-dimensional hypercube).
    
    Strategy:
    1. Precompute face-edge incidences using numpy for speed.
    2. Fast greedy construction with randomized orderings.
    3. Local search: repair bad faces, then add edges greedily.
    4. Multiple restarts with different random seeds and perturbation strategies.
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
    
    # Build face list - each face is a 4-cycle with 4 edges
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
    
    def greedy_initial(seed=None, strategy='random'):
        """Generate a greedy initial solution with various strategies."""
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        
        if strategy == 'random':
            order = list(range(n_edges))
            rng.shuffle(order)
        elif strategy == 'bit_priority':
            # Process edges in order of bit dimension, with random tie-breaking
            by_bit = defaultdict(list)
            for ei, (u, v) in enumerate(edges):
                xor = u ^ v
                bit = xor.bit_length() - 1
                by_bit[bit].append(ei)
            order = []
            for bit in range(d):
                by_bit[bit].sort(key=lambda e: rng.random())
                order.extend(by_bit[bit])
        elif strategy == 'parity_weighted':
            # Weight edges by parity of vertex
            order = list(range(n_edges))
            weights = []
            for ei, (u, v) in enumerate(edges):
                parity = (bin(u).count('1') + bin(v).count('1')) % 2
                weights.append(rng.random() + parity * 0.1)
            order.sort(key=lambda e: weights[e], reverse=True)
        elif strategy == 'clique_breaking':
            # Avoid creating dense local structures
            order = list(range(n_edges))
            rng.shuffle(order)
            # Second pass: prioritize edges that break potential cliques
            for _ in range(3):
                rng.shuffle(order)
        elif strategy == 'complementary':
            # Start with all edges, remove to break C4s
            sol = [True] * n_edges
            for fi in range(n_faces):
                face_counts[fi] = 4
            
            # Remove edges to break all faces
            order = list(range(n_edges))
            rng.shuffle(order)
            for ei in order:
                # Check if this edge is in any face with count 4
                in_bad_face = False
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 4:
                        in_bad_face = True
                        break
                if not in_bad_face:
                    continue
                # Remove this edge
                sol[ei] = False
                for fi in edge_faces[ei]:
                    face_counts[fi] -= 1
            return sol
        else:
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
    
    def local_search(sol, seed=None, max_iter=50000):
        """Improve solution via local search with various perturbation strategies."""
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
        stuck_count = 0
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
                            # Add small random tie-breaker
                            score = cnt + rng.random() * 0.1
                            if score > best_score:
                                best_score = score
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
                    stuck_count = 0
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
                            stuck_count = 0
                else:
                    # Stuck - try perturbation: randomly remove a few edges and re-add
                    if time_left() > 20:
                        selected = [e for e in range(n_edges) if sol[e]]
                        if len(selected) > 10:
                            # Try different perturbation sizes
                            pert_size = rng.choice([3, 5, 7, 10, 12, 15])
                            to_remove = rng.sample(selected, min(pert_size, max(2, len(selected) // 8)))
                            for ei in to_remove:
                                sol[ei] = False
                                for f2 in edge_faces[ei]:
                                    face_counts[f2] -= 1
                                    if f2 in bad_faces:
                                        bad_faces.discard(f2)
                            improved = True
                            stuck_count += 1
                            # If stuck too many times, do a bigger perturbation
                            if stuck_count > 5:
                                extra_size = rng.choice([5, 8, 12])
                                selected = [e for e in range(n_edges) if sol[e]]
                                if len(selected) > 20:
                                    extra_remove = rng.sample(selected, min(extra_size, len(selected) // 6))
                                    for ei in extra_remove:
                                        sol[ei] = False
                                        for f2 in edge_faces[ei]:
                                            face_counts[f2] -= 1
                                            if f2 in bad_faces:
                                                bad_faces.discard(f2)
                                stuck_count = 0
                    else:
                        break
    
        return sol
    
    def swap_search(sol, seed=None, max_iter=10000):
        """Try to improve by swapping edges: remove one, add another."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        improved = True
        iter_count = 0
        while improved and time_left() > 15 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Find edges that can be removed (not part of any bad face or part of exactly one bad face)
            removable = []
            for ei in range(n_edges):
                if not sol[ei]:
                    continue
                bad_cnt = 0
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 4:
                        bad_cnt += 1
                if bad_cnt <= 1:
                    removable.append(ei)
            
            if not removable:
                break
            
            # Find edges that can be added
            addable = []
            for ei in range(n_edges):
                if sol[ei]:
                    continue
                can_add = True
                for fi in edge_faces[ei]:
                    if face_counts[fi] >= 3:
                        can_add = False
                        break
                if can_add:
                    addable.append(ei)
            
            if not addable:
                # Try to remove an edge to make space
                if time_left() > 20 and removable:
                    ei = rng.choice(removable)
                    sol[ei] = False
                    for f2 in edge_faces[ei]:
                        face_counts[f2] -= 1
                    improved = True
                continue
            
            # Try swaps: remove one, add another
            rng.shuffle(removable)
            rng.shuffle(addable)
            
            for ei_rem in removable[:25]:  # Limit tries
                if time_left() < 15:
                    break
                for ei_add in addable[:25]:
                    # Check if removing ei_rem and adding ei_add keeps validity
                    # Simulate removal
                    temp_face_counts = face_counts.copy()
                    for f2 in edge_faces[ei_rem]:
                        temp_face_counts[f2] -= 1
                    
                    # Check if add is valid
                    valid_add = True
                    for f2 in edge_faces[ei_add]:
                        if temp_face_counts[f2] >= 3:
                            valid_add = False
                            break
                        temp_face_counts[f2] += 1
                    for f2 in edge_faces[ei_rem]:
                        temp_face_counts[f2] -= 1
                    
                    if valid_add:
                        # Perform swap
                        sol[ei_rem] = False
                        sol[ei_add] = True
                        for f2 in edge_faces[ei_rem]:
                            face_counts[f2] -= 1
                        for f2 in edge_faces[ei_add]:
                            face_counts[f2] += 1
                        improved = True
                        break
                if improved:
                    break
    
        return sol
    
    def simulated_annealing(sol, seed=None, max_iter=30000):
        """Simulated annealing to escape local optima."""
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
        
        current_count = count_edges(sol)
        best_count = current_count
        best_solution = list(sol)
        
        T = 5.0  # Initial temperature
        cooling = 0.995
        
        iter_count = 0
        while time_left() > 20 and iter_count < max_iter:
            iter_count += 1
            
            # Choose a random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add an edge
                candidates = [ei for ei in range(n_edges) if not sol[ei]]
                if not candidates:
                    continue
                ei = rng.choice(candidates)
                
                # Check if this creates a bad face
                new_bad = 0
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 3:
                        new_bad += 1
                
                delta = 1 - new_bad  # Gain 1 edge, lose new_bad faces
                if new_bad == 0:
                    # Always accept
                    sol[ei] = True
                    for fi in edge_faces[ei]:
                        face_counts[fi] += 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_solution = list(sol)
                else:
                    # Accept with probability exp(delta/T)
                    if rng.random() < np.exp(delta / T):
                        sol[ei] = True
                        for fi in edge_faces[ei]:
                            face_counts[fi] += 1
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_solution = list(sol)
                        # Update bad faces
                        for fi in edge_faces[ei]:
                            if face_counts[fi] == 4:
                                bad_faces.add(fi)
            
            elif move_type < 0.6:
                # Try to remove an edge
                candidates = [ei for ei in range(n_edges) if sol[ei]]
                if not candidates:
                    continue
                ei = rng.choice(candidates)
                
                # Check how many bad faces this fixes
                fixes_bad = 0
                for fi in edge_faces[ei]:
                    if face_counts[fi] == 4:
                        fixes_bad += 1
                
                delta = -1 + fixes_bad  # Lose 1 edge, gain fixes_bad faces
                if fixes_bad > 0:
                    # Always accept if it fixes bad faces
                    sol[ei] = False
                    for fi in edge_faces[ei]:
                        face_counts[fi] -= 1
                        if face_counts[fi] == 3 and fi in bad_faces:
                            bad_faces.discard(fi)
                    current_count -= 1
                else:
                    # Accept with probability exp(delta/T)
                    if rng.random() < np.exp(delta / T):
                        sol[ei] = False
                        for fi in edge_faces[ei]:
                            face_counts[fi] -= 1
                        current_count -= 1
            
            else:
                # Try a swap: remove one, add another
                rem_candidates = [ei for ei in range(n_edges) if sol[ei]]
                add_candidates = [ei for ei in range(n_edges) if not sol[ei]]
                if not rem_candidates or not add_candidates:
                    continue
                
                ei_rem = rng.choice(rem_candidates)
                ei_add = rng.choice(add_candidates)
                
                if ei_rem == ei_add:
                    continue
                
                # Simulate
                temp_fc = face_counts.copy()
                for f2 in edge_faces[ei_rem]:
                    temp_fc[f2] -= 1
                
                new_bad = 0
                for f2 in edge_faces[ei_add]:
                    if temp_fc[f2] == 3:
                        new_bad += 1
                    temp_fc[f2] += 1
                for f2 in edge_faces[ei_rem]:
                    temp_fc[f2] -= 1
                
                delta = 1 - new_bad
                if new_bad == 0:
                    # Always accept
                    sol[ei_rem] = False
                    sol[ei_add] = True
                    for f2 in edge_faces[ei_rem]:
                        face_counts[f2] -= 1
                        if face_counts[f2] == 3 and f2 in bad_faces:
                            bad_faces.discard(f2)
                    for f2 in edge_faces[ei_add]:
                        face_counts[f2] += 1
                        if face_counts[f2] == 4:
                            bad_faces.add(f2)
                    if current_count > best_count:
                        best_count = current_count
                        best_solution = list(sol)
                else:
                    if rng.random() < np.exp(delta / T):
                        sol[ei_rem] = False
                        sol[ei_add] = True
                        for f2 in edge_faces[ei_rem]:
                            face_counts[f2] -= 1
                            if face_counts[f2] == 3 and f2 in bad_faces:
                                bad_faces.discard(f2)
                        for f2 in edge_faces[ei_add]:
                            face_counts[f2] += 1
                            if face_counts[f2] == 4:
                                bad_faces.add(f2)
                        if current_count > best_count:
                            best_count = current_count
                            best_solution = list(sol)
            
            T *= cooling
        
        return best_solution
    
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
    
    # First, get the deterministic greedy baseline
    sol = deterministic_greedy()
    cnt = count_edges(sol)
    if is_valid(sol):
        best_count = cnt
        best_solution = list(sol)
    
    # Try complementary deletion as another starting point
    sol_comp = greedy_initial(42, 'complementary')
    cnt_comp = count_edges(sol_comp)
    if is_valid(sol_comp):
        if cnt_comp > best_count:
            best_count = cnt_comp
            best_solution = list(sol_comp)
    
    # Multiple restarts with different seeds and strategies
    attempts = 0
    strategies = ['random', 'bit_priority', 'parity_weighted', 'clique_breaking', 'complementary', 'random']
    
    while time_left() > 20:
        attempts += 1
        seed = attempts * 7919 + 42
        strategy = strategies[attempts % len(strategies)]
        
        # Alternate between different starting strategies
        if attempts % 7 == 1:
            sol = greedy_initial(seed, strategy)
        elif attempts % 7 == 2:
            sol = list(best_solution) if best_solution else greedy_initial(seed, strategy)
            # Perturb: remove a few random edges
            rng = random.Random(seed)
            selected = [e for e in range(n_edges) if sol[e]]
            if selected:
                to_remove = rng.sample(selected, min(5, len(selected) // 8))
                for ei in to_remove:
                    sol[ei] = False
        elif attempts % 7 == 3:
            sol = greedy_initial(seed + 12345, 'random')
        elif attempts % 7 == 4:
            sol = greedy_initial(seed + 99999, 'bit_priority')
        elif attempts % 7 == 5:
            sol = greedy_initial(seed + 55555, 'parity_weighted')
        elif attempts % 7 == 6:
            sol = greedy_initial(seed + 77777, 'complementary')
        else:
            sol = greedy_initial(seed + 33333, 'clique_breaking')
        
        # Local search
        sol = local_search(sol, seed)
        
        # Additional swap search
        if time_left() > 35:
            sol = swap_search(sol, seed + 777)
        
        # Simulated annealing for further improvement
        if time_left() > 50 and count_edges(sol) >= best_count - 5:
            sol_sa = simulated_annealing(sol, seed + 1234)
            cnt_sa = count_edges(sol_sa)
            if is_valid(sol_sa) and cnt_sa > count_edges(sol):
                sol = sol_sa
        
        # Verify and count
        cnt = count_edges(sol)
        if is_valid(sol):
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
        
        # Check time
        if time_left() < 35:
            break
    
    # Final polish: do a few more local search passes on the best solution
    if best_solution and time_left() > 25:
        for polish_iter in range(3):
            if time_left() < 25:
                break
            sol = list(best_solution)
            sol = local_search(sol, 99999 + polish_iter, max_iter=20000)
            cnt = count_edges(sol)
            if is_valid(sol) and cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
    
    # Final fallback
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

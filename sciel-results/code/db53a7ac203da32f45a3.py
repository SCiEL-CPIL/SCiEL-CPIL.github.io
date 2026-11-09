import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Maximize C4-free edge set in Q7 (7-dimensional hypercube).
    
    Strategy:
    1. Precompute face-edge incidences and conflict graph.
    2. Build conflict graph: two edges conflict if they are in a common face
       and selecting both would create a C4 (actually: they conflict if they
       are opposite edges in a face, since selecting all 4 creates C4,
       but the real constraint is: a face has C4 iff all 4 edges are selected.
       So the conflict is: for each face, we cannot select all 4 edges.
       This is a 4-uniform hypergraph constraint, not pairwise.
       
       Actually, let me reconsider. The constraint is: no face has all 4 edges.
       This means for each face (e1,e2,e3,e4), we need NOT(all selected).
       This is equivalent to: for each face, at least one edge is NOT selected.
       
       For branch-and-bound, we can model this as: we want to maximize |S|
       subject to: for each face F, S does not contain all of F.
       
       A better formulation: the complement. Let T = all edges \\ S.
       We want to minimize |T| subject to: for each face F, T intersects F.
       This is the minimum edge cover of the hypergraph (each face must be hit).
       This is a set cover problem!
       
       Actually it's: minimum hitting set where the sets are the faces (4-element sets)
       and the universe is the edges. We want the minimum number of edges to remove
       so that every face has at least one removed edge.
       
       So: minimize |T| such that for all faces F, T ∩ F ≠ ∅.
       This is the minimum hitting set / transversal problem.
       
       For 448 edges and 392 faces (7*6/2 * 128/4... let me compute),
       this is a set cover problem that can be solved with branch-and-bound.
    """
    start_time = time.time()
    deadline = start_time + 143  # 143s safety margin
    
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
    
    # Greedy initial solution
    def greedy_initial(seed=None, strategy='random'):
        rng = random.Random(seed)
        sol = [False] * n_edges
        face_counts = [0] * n_faces
        
        if strategy == 'random':
            order = list(range(n_edges))
            rng.shuffle(order)
        elif strategy == 'bit_priority':
            by_bit = defaultdict(list)
            for ei, (u, v) in enumerate(edges):
                xor = u ^ v
                bit = xor.bit_length() - 1
                by_bit[bit].append(ei)
            order = []
            for bit in range(d):
                by_bit[bit].sort(key=lambda e: rng.random())
                order.extend(by_bit[bit])
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
    
    def local_search(sol, seed=None, max_iter=30000):
        """Improve solution via local search."""
        rng = random.Random(seed)
        sol = list(sol)
        face_counts = [0] * n_faces
        for ei in range(n_edges):
            if sol[ei]:
                for fi in edge_faces[ei]:
                    face_counts[fi] += 1
        
        bad_faces = set()
        for fi in range(n_faces):
            if face_counts[fi] == 4:
                bad_faces.add(fi)
        
        improved = True
        iter_count = 0
        while improved and time_left() > 5 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            if bad_faces:
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
            else:
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
                        if time_left() < 5:
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
                    if time_left() > 10:
                        selected = [e for e in range(n_edges) if sol[e]]
                        if len(selected) > 10:
                            pert_size = rng.choice([3, 5, 7])
                            to_remove = rng.sample(selected, min(pert_size, len(selected) // 10))
                            for ei in to_remove:
                                sol[ei] = False
                                for f2 in edge_faces[ei]:
                                    face_counts[f2] -= 1
                            improved = True
                    else:
                        break
        
        return sol
    
    # Branch and bound for the hitting set formulation
    # We want to minimize |T| such that every face is hit by T.
    # T is the set of edges to REMOVE.
    # The answer is n_edges - min_hitting_set_size.
    
    def branch_and_bound_hitting_set(initial_sol, seed=None, max_nodes=500000):
        """
        Branch and bound to find minimum hitting set.
        State: (edges_considered, current_hitting_set, uncovered_faces)
        We process edges in some order, deciding include/exclude.
        
        Better: recursive with memoization-like pruning.
        """
        rng = random.Random(seed)
        
        # Start from the complement of initial_sol
        # initial_sol[i] = True means edge i is SELECTED (kept)
        # T = edges to remove = {i : not initial_sol[i]}
        T = set()
        for i in range(n_edges):
            if not initial_sol[i]:
                T.add(i)
        
        # Which faces are currently hit by T?
        face_hit = [False] * n_faces
        for ei in T:
            for fi in edge_faces[ei]:
                face_hit[fi] = True
        
        uncovered = [fi for fi in range(n_faces) if not face_hit[fi]]
        
        if not uncovered:
            return T  # Already a valid hitting set
        
        # We need to add more edges to T to cover all uncovered faces.
        # This is a set cover problem: sets = {edge_faces[ei] for ei not in T},
        # universe = uncovered faces.
        
        # Greedy set cover to get an initial upper bound
        greedy_T = set(T)
        greedy_face_hit = [False] * n_faces
        for ei in greedy_T:
            for fi in edge_faces[ei]:
                greedy_face_hit[fi] = True
        
        greedy_uncovered = set(fi for fi in range(n_faces) if not greedy_face_hit[fi])
        
        # Greedy: repeatedly pick the edge that covers the most uncovered faces
        available_edges = [ei for ei in range(n_edges) if ei not in greedy_T]
        
        while greedy_uncovered and available_edges:
            if time_left() < 5:
                break
            best_ei = None
            best_count = 0
            for ei in available_edges:
                cnt = 0
                for fi in edge_faces[ei]:
                    if fi in greedy_uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_ei = ei
            if best_ei is None or best_count == 0:
                break
            greedy_T.add(best_ei)
            available_edges.remove(best_ei)
            for fi in edge_faces[best_ei]:
                if fi in greedy_uncovered:
                    greedy_uncovered.discard(fi)
        
        upper_bound = len(greedy_T)
        
        # Now branch and bound
        # State: (current_T, uncovered_faces, lower_bound)
        # Lower bound: max over uncovered faces of 1 (each needs at least one edge)
        # Better lower bound: for each uncovered face, we need at least one edge from that face.
        # If two uncovered faces share no common edge, we need at least 2 edges.
        # Use a simple lower bound: number of uncovered faces / max faces per edge
        
        # For each edge, how many uncovered faces it covers
        # We'll use a recursive search with pruning
        
        best_T = set(greedy_T)
        best_size = upper_bound
        
        # Sort edges by how many uncovered faces they cover (descending)
        # for better branching
        
        def compute_lower_bound(uncovered_set):
            """Lower bound on additional edges needed."""
            if not uncovered_set:
                return 0
            # Each edge covers at most 6 faces (each edge is in C(6,1)=6 faces? No)
            # Each edge is in exactly (d-1) choose 1... actually each edge is in 
            # the number of faces that contain it. In Q7, each edge is in 
            # C(6,1) = 6 faces? No.
            # An edge connects vertices differing in bit i. The faces containing it
            # are those where the other dimension is j != i, and the base vertex
            # has bit i = 0 and bit j = 0. So for a given edge, there are 6 choices
            # for j, and for each j, there's exactly one face. So 6 faces per edge.
            # So max coverage per edge is 6.
            return (len(uncovered_set) + 5) // 6  # ceil division
        
        # Recursive B&B
        # To keep it tractable, we limit the depth and use time limits
        
        def bb_search(current_T, uncovered_list, depth=0):
            nonlocal best_T, best_size
            
            if time_left() < 3:
                return
            
            if not uncovered_list:
                if len(current_T) < best_size:
                    best_size = len(current_T)
                    best_T = set(current_T)
                return
            
            # Lower bound
            lb = len(current_T) + compute_lower_bound(set(uncovered_list))
            if lb >= best_size:
                return
            
            if depth > 20:  # Limit depth
                return
            
            # Choose an uncovered face to branch on
            # Pick the face with the fewest candidate edges (not in current_T)
            # that can cover it
            best_face = None
            best_candidates = None
            best_count = float('inf')
            
            for fi in uncovered_list[:50]:  # Limit faces considered
                candidates = [ei for ei in faces[fi] if ei not in current_T]
                if len(candidates) < best_count:
                    best_count = len(candidates)
                    best_face = fi
                    best_candidates = candidates
            
            if best_face is None:
                return
            
            # Branch: for each edge in best_candidates, add it to current_T
            # and recurse
            # Order by coverage
            def coverage(ei):
                cnt = 0
                for fi in edge_faces[ei]:
                    if fi in set(uncovered_list):
                        cnt += 1
                return cnt
            
            best_candidates.sort(key=coverage, reverse=True)
            
            for ei in best_candidates:
                if time_left() < 3:
                    return
                if len(current_T) + 1 >= best_size:
                    break
                
                new_T = set(current_T)
                new_T.add(ei)
                
                new_uncovered = []
                for fi in uncovered_list:
                    if fi in edge_faces[ei]:
                        continue
                    new_uncovered.append(fi)
                
                if not new_uncovered:
                    if len(new_T) < best_size:
                        best_size = len(new_T)
                        best_T = new_T
                else:
                    bb_search(new_T, new_uncovered, depth + 1)
        
        # Initialize uncovered_list
        uncovered_list = [fi for fi in range(n_faces) if not face_hit[fi]]
        
        bb_search(T, uncovered_list, 0)
        
        return best_T
    
    # Main algorithm
    best_count = 0
    best_solution = None
    
    # Get a good initial solution via greedy + local search
    for attempt in range(5):
        seed = attempt * 12345 + 67890
        sol = greedy_initial(seed, 'random' if attempt % 2 == 0 else 'bit_priority')
        sol = local_search(sol, seed)
        
        if is_valid(sol):
            cnt = count_edges(sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = list(sol)
    
    # Try branch and bound from the best solution
    if best_solution and time_left() > 20:
        # The hitting set T = {i : not best_solution[i]}
        # B&B tries to find a smaller T
        T = branch_and_bound_hitting_set(best_solution, seed=42)
        
        if T is not None:
            # Convert T back to solution
            bb_sol = [True] * n_edges
            for ei in T:
                bb_sol[ei] = False
            
            if is_valid(bb_sol):
                cnt = count_edges(bb_sol)
                if cnt > best_count:
                    best_count = cnt
                    best_solution = bb_sol
    
    # Final local search polish
    if best_solution and time_left() > 10:
        final_sol = local_search(best_solution, seed=99999, max_iter=10000)
        if is_valid(final_sol):
            cnt = count_edges(final_sol)
            if cnt > best_count:
                best_count = cnt
                best_solution = final_sol
    
    # Fallback
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

import random
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = n_vertices * d // 2

    # Precompute all edges
    all_edges = []
    edge_to_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_index[e] = len(all_edges)
                all_edges.append(e)

    # Precompute faces
    faces = []
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v = u ^ (1 << i)
                w = u ^ (1 << j)
                z = u ^ (1 << i) ^ (1 << j)

                e1 = (min(u, v), max(u, v))
                e2 = (min(v, z), max(v, z))
                e3 = (min(z, w), max(z, w))
                e4 = (min(w, u), max(w, u))

                face_edges = [e1, e2, e3, e4]
                faces.append(face_edges)

    # Optimization: Store faces as tuples of edge indices
    faces_idx = []
    for f in faces:
        idxs = tuple(edge_to_index[e] for e in f)
        faces_idx.append(idxs)

    # Map edge index to list of face indices
    edge_to_faces_idx = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            edge_to_faces_idx[e_idx].append(f_idx)

    # Precompute for each edge, the list of (face_idx, other_edge_indices) for fast validity checks
    edge_face_others = [[] for _ in range(total_edges)]
    for f_idx, f in enumerate(faces_idx):
        for e_idx in f:
            others = [idx for idx in f if idx != e_idx]
            edge_face_others[e_idx].append((f_idx, tuple(others)))

    def is_valid_addition(chosen, e_idx):
        """
        Check if adding edge e_idx to chosen set is valid.
        chosen is a set of edge indices.
        """
        for f_idx, others in edge_face_others[e_idx]:
            count = 0
            for idx in others:
                if idx in chosen:
                    count += 1
                    if count == 3:
                        return False
        return True

    def greedy_random():
        chosen = set()
        indices = list(range(total_edges))
        random.shuffle(indices)
        for e_idx in indices:
            if is_valid_addition(chosen, e_idx):
                chosen.add(e_idx)
        return chosen

    def local_search(chosen, time_limit):
        """
        Improved local search with 1-for-1, 2-for-1 swaps and perturbations.
        Uses precomputed edge_face_others for faster validity checks.
        """
        start_time = time.time()
        current_size = len(chosen)
        
        # Helper to try adding any valid missing edge
        def try_add_all():
            nonlocal current_size
            added = False
            for e_idx in range(total_edges):
                if e_idx not in chosen and is_valid_addition(chosen, e_idx):
                    chosen.add(e_idx)
                    current_size += 1
                    added = True
            return added

        # Initial repair
        try_add_all()

        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any missing edge
            if try_add_all():
                improved = True
                continue

            # 2. Try 2-for-1 swaps
            # For each missing edge e, find which single edge b in S would unblock it
            single_unblockers = {}
            missing_edges = [i for i in range(total_edges) if i not in chosen]
            
            for e in missing_edges:
                bad_faces = []
                for f_idx, others in edge_face_others[e]:
                    count_in_s = 0
                    for idx in others:
                        if idx in chosen:
                            count_in_s += 1
                    if count_in_s == 3:
                        bad_faces.append(others)
                
                if not bad_faces:
                    continue
                
                if len(bad_faces) == 1:
                    single_unblockers[e] = set(bad_faces[0])
                else:
                    common = set(bad_faces[0])
                    for bf in bad_faces[1:]:
                        common &= set(bf)
                    if common:
                        single_unblockers[e] = common
                    else:
                        single_unblockers[e] = set()

            # Map b -> list of e's it unblocks
            unblocks = {b: [] for b in chosen}
            for e, blockers in single_unblockers.items():
                for b in blockers:
                    if b in unblocks:
                        unblocks[b].append(e)

            # Try to find a b that unblocks >= 2 edges
            found_swap = False
            b_list = list(chosen)
            random.shuffle(b_list)
            
            for b in b_list:
                if time.time() - start_time > time_limit:
                    break
                if b not in chosen:
                    continue
                candidates = unblocks.get(b, [])
                if len(candidates) >= 2:
                    # Try all pairs from candidates (limited to first 10 for speed)
                    candidates = candidates[:10]
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            e1, e2 = candidates[i], candidates[j]
                            # Check validity by temporarily removing b
                            chosen.discard(b)
                            valid1 = is_valid_addition(chosen, e1)
                            if valid1:
                                chosen.add(e1)
                                valid2 = is_valid_addition(chosen, e2)
                                if valid2:
                                    chosen.add(e2)
                                    current_size += 1
                                    improved = True
                                    found_swap = True
                                    break
                                else:
                                    chosen.discard(e1)
                            chosen.add(b)
                            if found_swap:
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if found_swap:
                try_add_all()
                continue

            # 3. Try 1-for-1 swaps
            if missing_edges:
                e = random.choice(missing_edges)
                blockers = single_unblockers.get(e, set())
                if blockers:
                    b = random.choice(list(blockers))
                    chosen.discard(b)
                    chosen.add(e)
                    improved = True

            if not improved:
                # Perturbation: remove random edge, greedy repair
                if chosen:
                    b_to_remove = random.choice(list(chosen))
                    chosen.discard(b_to_remove)
                    current_size -= 1
                    try_add_all()

        # Final greedy fill
        try_add_all()
        return chosen

    best_set = set()
    best_size = 0
    
    start = time.time()
    deadline = start + 110 # 10s safety margin
    
    # Phase 1: Random Greedy Restarts
    while time.time() < deadline - 15:
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
    # Phase 2: Local Search on the best set
    if time.time() < deadline - 3:
        improved_set = local_search(best_set, deadline - time.time() - 3)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Phase 3: Final greedy fill to ensure maximality
    for e_idx in range(total_edges):
        if e_idx not in best_set and is_valid_addition(best_set, e_idx):
            best_set.add(e_idx)
            
    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])
        
    return result

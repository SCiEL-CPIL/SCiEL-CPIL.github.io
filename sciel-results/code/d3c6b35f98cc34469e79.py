def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 115

    d = 7
    nverts = 1 << d  # 128

    # 1. Build edges
    edges = []
    edge_map = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx

    num_edges = len(edges)  # 448

    # 2. Build faces
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    face_vertices_set = set()

    for i in range(d):
        for j in range(i + 1, d):
            for u in range(nverts):
                if (u & (1 << i)) != 0:
                    continue
                if (u & (1 << j)) != 0:
                    continue

                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)

                face_key = tuple(sorted([v1, v2, v3, v4]))
                if face_key in face_vertices_set:
                    continue

                face_vertices_set.add(face_key)
                f_idx = len(face_edges_list)

                e1 = edge_map[(min(v1, v2), max(v1, v2))]
                e2 = edge_map[(min(v1, v3), max(v1, v3))]
                e3 = edge_map[(min(v2, v4), max(v2, v4))]
                e4 = edge_map[(min(v3, v4), max(v3, v4))]

                face_edges_list.append((e1, e2, e3, e4))

                edge_faces[e1].append(f_idx)
                edge_faces[e2].append(f_idx)
                edge_faces[e3].append(f_idx)
                edge_faces[e4].append(f_idx)

    num_faces = len(face_edges_list)

    # 3. Build conflict graph using bitsets
    # Two edges conflict if they share a face (i.e., cannot both be in a C4-free set
    # if all 4 edges of that face are selected - actually, the constraint is that
    # no face has all 4 edges selected)
    # Wait, the constraint is: no 2-dimensional face may have all four edges selected.
    # This is NOT a simple pairwise conflict. It's a 4-uniform hypergraph constraint.
    # So we can't directly model this as maximum independent set in a graph.
    #
    # However, we CAN model it differently. Let's think about what the constraint means.
    # For each face (4 edges), we cannot have all 4 selected.
    # This is equivalent to: for each face, at least one edge is NOT selected.
    #
    # This is a hitting set / set cover type problem, or equivalently, we want to
    # maximize selected edges subject to: for each face, not all 4 are selected.
    #
    # Let's stick with the local search approach but improve it.
    # Actually, let me reconsider. The constraint is on faces, not on pairs.
    # So the "conflict graph" idea doesn't directly apply.
    #
    # Let's improve the local search with better heuristics.

    # Precompute face edge sets for faster lookup
    face_edges_set = [set(f) for f in face_edges_list]

    # 4. Greedy Initialization
    def random_greedy(seed=None):
        if seed is not None:
            random.seed(seed)

        selected = [False] * num_edges
        face_count = [0] * num_faces
        order = list(range(num_edges))
        random.shuffle(order)

        count = 0
        for e in order:
            ok = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for f in edge_faces[e]:
                    face_count[f] += 1

        return selected, face_count, count

    # Try a few random starts to find a good baseline
    best_selected, best_face_count, best_count = None, None, 0
    t0 = time.time()
    while time.time() - t0 < 2.0:  # Spend 2s finding a good start
        s, fc, c = random_greedy()
        if c > best_count:
            best_count = c
            best_selected = s
            best_face_count = fc

    selected = best_selected[:]
    face_count = best_face_count[:]
    count = best_count
    best_found = count
    best_selected_copy = selected[:]
    best_face_count_copy = face_count[:]

    # 5. Improved local search with multiple strategies
    def try_add_edge(e):
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        selected[e] = True
        count += 1
        for f in edge_faces[e]:
            face_count[f] += 1
        if count > best_found:
            best_found = count
            best_selected_copy = selected[:]
            best_face_count_copy = face_count[:]
        return True

    def remove_edge(e):
        nonlocal selected, face_count, count
        selected[e] = False
        count -= 1
        for f in edge_faces[e]:
            face_count[f] -= 1
        return True

    def get_available_edges():
        avail = []
        for e in range(num_edges):
            if selected[e]:
                continue
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                avail.append(e)
        return avail

    def find_blocking_faces(e):
        b_faces = []
        for f in edge_faces[e]:
            if face_count[f] == 3:
                b_faces.append(f)
        return b_faces

    def try_2swap(e_add, e_rem1, e_rem2):
        """Remove 2 edges, add 1 edge. Returns True if successful."""
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        # Check that removing both and adding e_add is valid
        # Temporarily remove both
        for f in edge_faces[e_rem1]:
            face_count[f] -= 1
        for f in edge_faces[e_rem2]:
            face_count[f] -= 1
        selected[e_rem1] = False
        selected[e_rem2] = False
        count -= 2
        
        # Try to add e_add
        can_add = True
        for f in edge_faces[e_add]:
            if face_count[f] == 3:
                can_add = False
                break
        
        if can_add:
            selected[e_add] = True
            count += 1
            for f in edge_faces[e_add]:
                face_count[f] += 1
            if count > best_found:
                best_found = count
                best_selected_copy = selected[:]
                best_face_count_copy = face_count[:]
            return True
        else:
            # Revert
            selected[e_rem1] = True
            selected[e_rem2] = True
            count += 2
            for f in edge_faces[e_rem1]:
                face_count[f] += 1
            for f in edge_faces[e_rem2]:
                face_count[f] += 1
            return False

    def try_3swap(e_add, e_rem1, e_rem2, e_rem3):
        """Remove 3 edges, add 1 edge. Returns True if successful."""
        nonlocal selected, face_count, count, best_found, best_selected_copy, best_face_count_copy
        for f in edge_faces[e_rem1]:
            face_count[f] -= 1
        for f in edge_faces[e_rem2]:
            face_count[f] -= 1
        for f in edge_faces[e_rem3]:
            face_count[f] -= 1
        selected[e_rem1] = False
        selected[e_rem2] = False
        selected[e_rem3] = False
        count -= 3
        
        can_add = True
        for f in edge_faces[e_add]:
            if face_count[f] == 3:
                can_add = False
                break
        
        if can_add:
            selected[e_add] = True
            count += 1
            for f in edge_faces[e_add]:
                face_count[f] += 1
            if count > best_found:
                best_found = count
                best_selected_copy = selected[:]
                best_face_count_copy = face_count[:]
            return True
        else:
            selected[e_rem1] = True
            selected[e_rem2] = True
            selected[e_rem3] = True
            count += 3
            for f in edge_faces[e_rem1]:
                face_count[f] += 1
            for f in edge_faces[e_rem2]:
                face_count[f] += 1
            for f in edge_faces[e_rem3]:
                face_count[f] += 1
            return False

    # Main loop with multiple phases
    neutral_swap_count = 0
    max_neutral_swaps = 300
    kick_attempts = 0
    max_kick_attempts = 50
    polish_phase = False
    polish_start_time = None
    last_improvement_time = time.time()

    while time.time() < deadline:
        # Check if we should enter polish phase (no improvement for a while)
        if not polish_phase and time.time() - last_improvement_time > 10:
            polish_phase = True
            polish_start_time = time.time()
        
        # Phase 1: Greedy Add - add any available edge
        avail = get_available_edges()
        if avail:
            e = random.choice(avail)
            if try_add_edge(e):
                last_improvement_time = time.time()
                neutral_swap_count = 0
                kick_attempts = 0
            continue

        # Phase 2: Try 1-for-1 swaps
        e = random.randint(0, num_edges - 1)
        if selected[e]:
            continue

        b_faces = find_blocking_faces(e)
        if not b_faces:
            if try_add_edge(e):
                last_improvement_time = time.time()
                neutral_swap_count = 0
                kick_attempts = 0
            continue

        # Find candidate edges to remove
        candidate_removes = []
        first_f = b_faces[0]
        for r in face_edges_list[first_f]:
            if selected[r]:
                # Check if removing r unblocks all blocking faces
                ok = True
                for f in b_faces[1:]:
                    if r not in face_edges_set[f]:
                        ok = False
                        break
                if ok:
                    candidate_removes.append(r)

        if candidate_removes:
            r = random.choice(candidate_removes)
            # Try the swap
            remove_edge(r)
            if try_add_edge(e):
                last_improvement_time = time.time()
                neutral_swap_count = 0
                kick_attempts = 0
            else:
                # Revert
                remove_edge(r)
                try_add_edge(r)
                neutral_swap_count += 1
        else:
            neutral_swap_count += 1

        # Phase 3: If stuck, try 2-for-1 swaps
        if neutral_swap_count > max_neutral_swaps:
            # Try a 2-for-1 swap
            success = False
            for _ in range(5):
                e_add = random.randint(0, num_edges - 1)
                if selected[e_add]:
                    continue
                b_faces = find_blocking_faces(e_add)
                if not b_faces:
                    if try_add_edge(e_add):
                        last_improvement_time = time.time()
                        neutral_swap_count = 0
                        kick_attempts = 0
                        success = True
                        break
                    continue
                
                # Find two edges to remove that together unblock e_add
                # Collect all edges in blocking faces
                rem_candidates = set()
                for f in b_faces:
                    for r in face_edges_list[f]:
                        if selected[r]:
                            rem_candidates.add(r)
                
                rem_list = list(rem_candidates)
                if len(rem_list) >= 2:
                    # Try random pairs
                    for _ in range(10):
                        i, j = random.sample(range(len(rem_list)), 2)
                        if try_2swap(e_add, rem_list[i], rem_list[j]):
                            last_improvement_time = time.time()
                            neutral_swap_count = 0
                            kick_attempts = 0
                            success = True
                            break
                if success:
                    break

            if not success:
                kick_attempts += 1
                if kick_attempts > max_kick_attempts:
                    # Restart with new greedy
                    t0 = time.time()
                    while time.time() - t0 < 1.0:
                        s, fc, c = random_greedy()
                        if c > best_count:
                            best_count = c
                            best_selected = s
                            best_face_count = fc
                    selected = best_selected[:]
                    face_count = best_face_count[:]
                    count = best_count
                    neutral_swap_count = 0
                    kick_attempts = 0
                    polish_phase = False
                    last_improvement_time = time.time()

    # Final polish pass: try to squeeze out more edges
    if time.time() < deadline - 5:
        # Do a more aggressive search with larger swaps
        polish_deadline = deadline - 5
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            avail = get_available_edges()
            if avail:
                for e in random.sample(avail, min(len(avail), 5)):
                    try_add_edge(e)
            
            # Try some 2-for-1 and 3-for-1 swaps
            for _ in range(20):
                if time.time() > polish_deadline:
                    break
                e_add = random.randint(0, num_edges - 1)
                if selected[e_add]:
                    continue
                b_faces = find_blocking_faces(e_add)
                if not b_faces:
                    try_add_edge(e_add)
                    continue
                
                rem_candidates = set()
                for f in b_faces:
                    for r in face_edges_list[f]:
                        if selected[r]:
                            rem_candidates.add(r)
                
                rem_list = list(rem_candidates)
                if len(rem_list) >= 2:
                    for _ in range(5):
                        i, j = random.sample(range(len(rem_list)), 2)
                        if try_2swap(e_add, rem_list[i], rem_list[j]):
                            break
                if len(rem_list) >= 3:
                    for _ in range(3):
                        i, j, k = random.sample(range(len(rem_list)), 3)
                        if try_3swap(e_add, rem_list[i], rem_list[j], rem_list[k]):
                            break

    # Return best found
    result = []
    for i in range(num_edges):
        if best_selected_copy[i]:
            result.append(list(edges[i]))

    return result

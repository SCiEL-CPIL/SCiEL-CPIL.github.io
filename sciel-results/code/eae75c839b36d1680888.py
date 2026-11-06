def construct():
    import time
    import random

    start_time = time.time()
    deadline = start_time + 112.0

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

    num_faces = len(face_edges_list)  # 336 faces
    face_edges_set = [set(f) for f in face_edges_list]

    # State: removed[e] = True means edge e is in R (removed from S)
    # S = E \ R, |S| = 448 - |R|
    # Constraint: for every face, at least 1 edge is in R (face_removed_count[f] >= 1)

    def random_greedy_remove(seed=None):
        """Build R greedily: start with R = {}, add edges to R to break all intact faces."""
        if seed is not None:
            random.seed(seed)

        removed = [False] * num_edges
        face_removed_count = [0] * num_faces
        intact_faces = [True] * num_faces
        num_intact = num_faces

        count_removed = 0
        while num_intact > 0:
            best_e = -1
            best_break = 0
            for e in range(num_edges):
                if removed[e]:
                    continue
                brk = 0
                for f in edge_faces[e]:
                    if intact_faces[f]:
                        brk += 1
                if brk > best_break:
                    best_break = brk
                    best_e = e
                if best_break >= 2:
                    break

            if best_e == -1:
                break

            removed[best_e] = True
            count_removed += 1
            for f in edge_faces[best_e]:
                if intact_faces[f]:
                    intact_faces[f] = False
                    num_intact -= 1
                face_removed_count[f] += 1

        return removed, face_removed_count, count_removed

    def greedy_remove_by_priority(seed=None):
        """Greedy with random tie-breaking for speed."""
        if seed is not None:
            random.seed(seed)

        removed = [False] * num_edges
        face_removed_count = [0] * num_faces
        intact_faces = [True] * num_faces
        num_intact = num_faces

        count_removed = 0
        while num_intact > 0:
            best_e = -1
            best_score = 0
            candidates = list(range(num_edges))
            random.shuffle(candidates)
            for e in candidates[:80]:
                if removed[e]:
                    continue
                score = 0
                for f in edge_faces[e]:
                    if intact_faces[f]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_e = e
                if best_score >= 2:
                    break

            if best_e == -1:
                for e in range(num_edges):
                    if not removed[e]:
                        best_e = e
                        break
                if best_e == -1:
                    break

            removed[best_e] = True
            count_removed += 1
            for f in edge_faces[best_e]:
                if intact_faces[f]:
                    intact_faces[f] = False
                    num_intact -= 1
                face_removed_count[f] += 1

        return removed, face_removed_count, count_removed

    # Try multiple random starts
    best_removed = None
    best_face_rc = None
    best_count_removed = num_edges

    t0 = time.time()
    while time.time() - t0 < 4.0:
        r, frc, c = random_greedy_remove()
        if c < best_count_removed:
            best_count_removed = c
            best_removed = r
            best_face_rc = frc

    for _ in range(5):
        r, frc, c = greedy_remove_by_priority()
        if c < best_count_removed:
            best_count_removed = c
            best_removed = r
            best_face_rc = frc

    removed = best_removed[:]
    face_removed_count = best_face_rc[:]
    count_removed = best_count_removed
    best_count_removed_found = count_removed
    best_removed_copy = removed[:]

    # Local search on R
    no_improve_count = 0
    max_no_improve = 5000
    stagnation_count = 0
    max_stagnation = 3000

    while time.time() < deadline:
        # Phase 1: Try to unremove any edge from R (add back to S)
        added_back = False
        for e in range(num_edges):
            if not removed[e]:
                continue
            # Check if unremoving e is safe: for all faces containing e, face_removed_count[f] >= 2
            safe = True
            for f in edge_faces[e]:
                if face_removed_count[f] < 2:
                    safe = False
                    break
            if safe:
                removed[e] = False
                count_removed -= 1
                for f in edge_faces[e]:
                    face_removed_count[f] -= 1
                if count_removed < best_count_removed_found:
                    best_count_removed_found = count_removed
                    best_removed_copy = removed[:]
                added_back = True
                no_improve_count = 0
                stagnation_count = 0
                break

        if added_back:
            continue

        # Phase 2: Try 1-for-1 swaps in R
        r_edges = [e for e in range(num_edges) if removed[e]]
        if not r_edges:
            break

        r = random.choice(r_edges)
        unremoved_edges = [e for e in range(num_edges) if not removed[e]]
        if not unremoved_edges:
            break
        e = random.choice(unremoved_edges)

        # Check if swapping r (out) with e (in) is valid
        # For faces containing r but not e: face_removed_count[f] must be >= 2
        swap_valid = True
        for f in edge_faces[r]:
            if e in face_edges_set[f]:
                continue
            if face_removed_count[f] < 2:
                swap_valid = False
                break

        if swap_valid:
            # Perform swap
            removed[r] = False
            count_removed -= 1
            for f in edge_faces[r]:
                face_removed_count[f] -= 1

            removed[e] = True
            count_removed += 1
            for f in edge_faces[e]:
                face_removed_count[f] += 1

            if count_removed < best_count_removed_found:
                best_count_removed_found = count_removed
                best_removed_copy = removed[:]
            no_improve_count = 0
            stagnation_count = 0
        else:
            stagnation_count += 1

        if count_removed >= best_count_removed_found:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if no_improve_count > max_no_improve or stagnation_count > max_stagnation:
            t0 = time.time()
            while time.time() - t0 < 1.5:
                r, frc, c = random_greedy_remove()
                if c < best_count_removed:
                    best_count_removed = c
                    best_removed = r
                    best_face_rc = frc

            removed = best_removed[:]
            face_removed_count = best_face_rc[:]
            count_removed = best_count_removed
            no_improve_count = 0
            stagnation_count = 0

    # Final polish
    for e in range(num_edges):
        if not removed[e]:
            continue
        safe = True
        for f in edge_faces[e]:
            if face_removed_count[f] < 2:
                safe = False
                break
        if safe:
            removed[e] = False
            count_removed -= 1
            for f in edge_faces[e]:
                face_removed_count[f] -= 1
            if count_removed < best_count_removed_found:
                best_count_removed_found = count_removed
                best_removed_copy = removed[:]

    # Reconstruct S = E \ R
    result = []
    for i in range(num_edges):
        if not best_removed_copy[i]:
            result.append(list(edges[i]))

    return result

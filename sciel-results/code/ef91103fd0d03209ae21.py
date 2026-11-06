def construct():
    import time
    import random
    import numpy as np

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
    # A face is a 4-cycle in the hypercube.
    # There are C(7,2) * 2^(7-2) = 21 * 16 = 336 faces.
    face_edges_list = []
    edge_faces = [[] for _ in range(num_edges)]
    face_vertices_set = set()

    for i in range(d):
        for j in range(i + 1, d):
            # Iterate through all vertices, but only pick one base per face
            # A face is defined by fixing bits other than i and j.
            # To avoid duplicates, we can iterate u where bit i=0 and bit j=0.
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

                # Edges are: (v1,v2), (v1,v3), (v2,v4), (v3,v4)
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

    # Precompute face edge sets for faster lookup
    face_edges_set = [set(f) for f in face_edges_list]

    # 3. Greedy Initialization
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
    best_selected = selected[:]
    best_face_count = face_count[:]

    # 4. Local Search with Simulated Annealing-like swaps
    # We maintain:
    # selected: list of bools
    # face_count: list of ints (count of selected edges in each face)

    def try_add_edge(e):
        nonlocal selected, face_count, count, best_found, best_selected, best_face_count
        for f in edge_faces[e]:
            if face_count[f] == 3:
                return False
        selected[e] = True
        count += 1
        for f in edge_faces[e]:
            face_count[f] += 1
        if count > best_found:
            best_found = count
            best_selected = selected[:]
            best_face_count = face_count[:]
        return True

    def remove_edge(e):
        nonlocal selected, face_count, count
        selected[e] = False
        count -= 1
        for f in edge_faces[e]:
            face_count[f] -= 1
        return True

    # Main loop
    # We will alternate between:
    # 1. Greedy Add: Find any edge that can be added, add it.
    # 2. Swap: If no add is possible, try a swap to escape local optimum.
    #    A swap removes one edge and adds one. We want to find a swap that
    #    allows a future add, or at least maintains the count while changing structure.
    #    However, a simple swap that maintains count doesn't help directly unless it
    #    enables a subsequent add.
    #    Better strategy: Randomly pick an unselected edge e. If it's blocked,
    #    find a set of edges to remove that unblocks it.
    #    Since we want to maximize size, we prefer swaps that result in a net gain?
    #    No, a swap is -1 + 1 = 0. But it might unlock an add.
    #    So:
    #    - If add possible, do it.
    #    - If not, do a random "move" that might help.
    #      Move: Pick random unselected e. If blocked, try to remove one edge r
    #      that is in ALL blocking faces of e. If such r exists, remove r and add e.
    #      This is a neutral swap. It changes the configuration.
    #      We might get stuck in a loop of neutral swaps.
    #      To avoid this, we can use a "temperature" or just accept neutral swaps
    #      with some probability, or keep track of recent states.
    #      Given the small size, maybe just do random neutral swaps and hope to hit
    #      a state where an add is possible.
    #      Actually, if no add is possible, the set is maximal.
    #      Any maximal C4-free set has size at least some lower bound.
    #      We want the maximum size.
    #      So we need to escape local optima.
    #      Simulated Annealing: Accept neutral swaps with prob exp(-1/T)?
    #      Or just accept them always for a while, then restart?
    #      Let's try:
    #      - Try to add.
    #      - If fail, do a random neutral swap (if possible).
    #      - If no neutral swap possible (isolated?), restart with new greedy.

    # To speed up "Try to add", we can maintain a list of "available" edges?
    # An edge is available if it's not selected and no face containing it has count 3.
    # When we add an edge, it might make other edges unavailable.
    # When we remove an edge, it might make other edges available.
    # This bookkeeping is complex. With 448 edges, scanning is fast enough.

    # Let's use a simple loop.
    # To prevent infinite loops of neutral swaps, we can limit the number of
    # consecutive neutral swaps before forcing a restart or a "kick" (remove 2, add 1? No, size drops).
    # Maybe remove 1, add 1 is the only safe move.
    # If we do many neutral swaps, we might cycle.
    # Let's just do them and hope. The state space is huge.

    # Optimization: Precompute for each edge the list of faces. (Done: edge_faces)
    # Optimization: Precompute for each face the list of edges. (Done: face_edges_list)

    # Let's implement the loop.
    neutral_swap_count = 0
    max_neutral_swaps = 1000

    while time.time() < deadline:
        # Phase 1: Greedy Add
        added = False
        # Randomize the order of checking edges to avoid bias?
        # Or just scan 0..447.
        # Scanning is deterministic. If we always pick the first available,
        # we might get stuck.
        # Let's pick a random unselected edge to check?
        # No, if ANY edge is addable, we should add it.
        # Which one? It matters.
        # Greedy: pick the one that "hurts" the least?
        # "Hurt" = number of faces that become count 3 (blocking future adds)?
        # Let's just pick the first one found in a random order.
        # Generate random order once? No, state changes.
        # Let's just scan 0..447. It's fast.
        for e in range(num_edges):
            if selected[e]:
                continue
            can_add = True
            for f in edge_faces[e]:
                if face_count[f] == 3:
                    can_add = False
                    break
            if can_add:
                try_add_edge(e)
                added = True
                neutral_swap_count = 0
                break

        if added:
            continue

        # Phase 2: Swap (if no add possible)
        # Pick a random unselected edge e
        e = random.randint(0, num_edges - 1)
        if selected[e]:
            continue

        # Check if e is blocked
        b_faces = []
        for f in edge_faces[e]:
            if face_count[f] == 3:
                b_faces.append(f)

        if not b_faces:
            # Should have been added in Phase 1.
            # Maybe face counts changed? No, we didn't change anything.
            # Just add it.
            try_add_edge(e)
            neutral_swap_count = 0
            continue

        # Find a valid r to remove
        # r must be in S and in all b_faces.
        # Intersection of edges in b_faces that are selected.
        # Since b_faces is small (max 6), we can compute intersection.
        # Start with edges of first b_face.
        candidate_r = None
        if b_faces:
            first_f = b_faces[0]
            # Get selected edges in first_f
            sel_in_first = [re for re in face_edges_list[first_f] if selected[re]]
            for r in sel_in_first:
                ok = True
                for f in b_faces[1:]:
                    if r not in face_edges_set[f]:
                        ok = False
                        break
                if ok:
                    candidate_r = r
                    break

        if candidate_r is None:
            # No single edge removal can unblock e.
            # We could try double removal (remove 2, add 1 -> size -1). Bad.
            # Or remove 2, add 2?
            # Let's just skip this e and try another.
            # To avoid getting stuck, we count this as a "failed swap attempt".
            neutral_swap_count += 1
            if neutral_swap_count > max_neutral_swaps:
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
            continue

        # Perform Swap
        remove_edge(candidate_r)
        # Now e should be addable
        # Verify
        can_add = True
        for f in edge_faces[e]:
            if face_count[f] == 3:
                can_add = False
                break
        if can_add:
            try_add_edge(e)
            # This was a neutral swap (remove 1, add 1)
            # It didn't increase count, but changed structure.
            neutral_swap_count += 1
            if neutral_swap_count > max_neutral_swaps:
                # Restart
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
        else:
            # Should not happen if logic is correct.
            # Restore state?
            # If we can't add e, we have lost an edge.
            # We should undo the removal.
            # But we don't have an "undo" function.
            # We can just add candidate_r back?
            # But candidate_r might be blocked now?
            # Actually, if we removed candidate_r, face counts decreased.
            # Adding candidate_r back should be valid because it was valid before.
            # Let's just add it back to restore size.
            try_add_edge(candidate_r)
            neutral_swap_count += 1

    # Return best found
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append(list(edges[i]))

    return result

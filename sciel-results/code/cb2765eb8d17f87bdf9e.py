import random
import time
import sys

def construct():
    d = 7
    n_v = 1 << d
    deadline = time.time() + 140  # safety margin before 150s kill

    # Build edges
    edges = []
    edge_idx = {}
    for x in range(n_v):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_idx[(x, y)] = idx

    n_edges = len(edges)  # 448

    # Build faces: each face is a set of 4 edges.
    # A 2D face is determined by vertex x and two dimensions (i, j) with i<j.
    # The 4 vertices: x, x^1<<i, x^1<<j, x^(1<<i)^(1<<j)
    # The 4 edges: (x, x^1<<i), (x, x^1<<j), (x^1<<i, x^(1<<i)^(1<<j)), (x^1<<j, x^(1<<i)^(1<<j))
    # To avoid duplicates, use x as the "minimum" vertex in the face? 
    # Actually, a face is uniquely determined by its 4 vertices. Let's use the min vertex x_min and the two dims.
    # But x must be such that x has 0 in both dims i and j? No, that's not right either.
    # A face is a subcube of dimension 2. It's determined by fixing 5 bits and leaving 2 free.
    # The 4 vertices are: base, base^1<<i, base^1<<j, base^(1<<i)^(1<<j) where base has 0 in bits i and j.
    # So for each pair (i,j) and each assignment of the other 5 bits, we get a face.
    # Total: C(7,2) * 2^5 = 21 * 32 = 672 faces. Correct.

    faces = []  # list of list of 4 edge indices
    edge_to_faces = [[] for _ in range(n_edges)]

    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all assignments of the other d-2 bits
            for mask in range(1 << (d - 2)):
                # Reconstruct base: bits i and j are 0
                base = 0
                bit_pos = 0
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if mask & (1 << bit_pos):
                        base |= (1 << k)
                    bit_pos += 1
                # 4 vertices
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ (1 << i) ^ (1 << j)
                # 4 edges
                e01 = edge_idx[(min(v0, v1), max(v0, v1))]
                e02 = edge_idx[(min(v0, v2), max(v0, v2))]
                e13 = edge_idx[(min(v1, v3), max(v1, v3))]
                e23 = edge_idx[(min(v2, v3), max(v2, v3))]
                face_edges = [e01, e02, e13, e23]
                faces.append(face_edges)
                for e in face_edges:
                    edge_to_faces[e].append(len(faces) - 1)

    n_faces = len(faces)  # 672

    def greedy_random_order(order, time_limit):
        """Run greedy with given edge order. Returns (count, selected_set)."""
        selected = [False] * n_edges
        face_count = [0] * n_faces
        count = 0
        for e in order:
            # Check if adding e creates a C4
            can_add = True
            for f in edge_to_faces[e]:
                if face_count[f] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                count += 1
                for f in edge_to_faces[e]:
                    face_count[f] += 1
        return count, selected

    def local_search(selected, face_count, count, time_limit):
        """Try to improve via swaps. Returns (count, selected, face_count)."""
        # Build list of non-selected edges
        non_selected = [e for e in range(n_edges) if not selected[e]]
        random.shuffle(non_selected)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for e_new in non_selected:
                if time.time() >= time_limit:
                    break
                # Check if e_new can be added
                blocking_faces = []
                for f in edge_to_faces[e_new]:
                    if face_count[f] >= 3:
                        blocking_faces.append(f)
                if not blocking_faces:
                    # Can add directly
                    selected[e_new] = True
                    count += 1
                    for f in edge_to_faces[e_new]:
                        face_count[f] += 1
                    non_selected.remove(e_new)
                    improved = True
                else:
                    # Try swaps: remove one edge from a blocking face, then add e_new
                    # For each blocking face, find a selected edge in it (other than e_new which isn't selected)
                    for f in blocking_faces:
                        for e_old in faces[f]:
                            if not selected[e_old]:
                                continue
                            # Check if removing e_old and adding e_new is valid
                            # After removing e_old, check if e_new can be added
                            # Simulate
                            ok = True
                            # For each face containing e_old, decrement count
                            # For each face containing e_new, check if count (after removal) < 4
                            for f2 in edge_to_faces[e_old]:
                                # face_count[f2] will decrease by 1
                                pass
                            for f2 in edge_to_faces[e_new]:
                                # face_count[f2] will increase by 1
                                # If e_old is also in f2, net change is 0
                                # If not, net change is +1
                                if e_old in faces[f2]:
                                    # net change 0, so need face_count[f2] < 4
                                    if face_count[f2] >= 4:
                                        ok = False
                                        break
                                else:
                                    # net change +1, so need face_count[f2] < 3
                                    if face_count[f2] >= 3:
                                        ok = False
                                        break
                            if not ok:
                                continue
                            # Perform swap
                            selected[e_old] = False
                            selected[e_new] = True
                            for f2 in edge_to_faces[e_old]:
                                face_count[f2] -= 1
                            for f2 in edge_to_faces[e_new]:
                                face_count[f2] += 1
                            non_selected.remove(e_new)
                            improved = True
                            break
                        if improved:
                            break
        return count, selected, face_count

    best_count = 0
    best_selected = None

    # Run multiple random greedy + local search
    seed = 42
    rng = random.Random(seed)
    
    for trial in range(50):  # up to 50 trials
        if time.time() >= deadline:
            break
        order = list(range(n_edges))
        rng.shuffle(order)
        
        # Time limit for greedy: small
        greedy_time = time.time() + 5
        count, selected = greedy_random_order(order, greedy_time)
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
        
        # Local search with remaining time
        if time.time() >= deadline:
            break
        ls_time = deadline - 5  # leave 5s margin
        face_count = [0] * n_faces
        for e in range(n_edges):
            if selected[e]:
                for f in edge_to_faces[e]:
                    face_count[f] += 1
        
        count2, selected2, face_count2 = local_search(selected, face_count, count, ls_time)
        
        if count2 > best_count:
            best_count = count2
            best_selected = selected2[:]
        
        if time.time() >= deadline:
            break

    # Build output
    out = []
    for e in range(n_edges):
        if best_selected[e]:
            x, y, i = edges[e]
            out.append([x, y])
    
    return out

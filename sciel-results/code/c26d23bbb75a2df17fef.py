import time
import random
import numpy as np
from collections import deque

def construct():
    d = 7
    n = 2 ** d
    m = d * (n // 2)  # 448
    deadline = time.time() + 115.0

    # Generate edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    # Generate faces and edge-face incidence
    faces = []
    edge_faces = [set() for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)

                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v1, v3), max(v1, v3))]
                e2 = edge_idx[(min(v2, v3), max(v2, v3))]
                e3 = edge_idx[(min(v0, v2), max(v0, v2))]

                face_edges = frozenset([e0, e1, e2, e3])
                faces.append(face_edges)
                f_idx = len(faces) - 1
                for e in face_edges:
                    edge_faces[e].add(f_idx)

    F = len(faces)  # 672

    # Precompute face edge lists for fast access
    face_edge_lists = [list(f) for f in faces]

    def greedy_hitting_set(rng):
        """Greedy hitting set: repeatedly pick edge covering most unhit faces."""
        removed = set()
        unhit = set(range(F))

        while unhit:
            best_e = -1
            max_cov = -1
            for e in range(m):
                cov = 0
                for f in edge_faces[e]:
                    if f in unhit:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_e = e

            if best_e == -1:
                break

            removed.add(best_e)
            for f in edge_faces[best_e]:
                if f in unhit:
                    unhit.remove(f)

        return removed

    def local_search(removed, rng, time_limit):
        """Local search with deletions, swaps, and 2-swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # 1. Try deletions
            for e in list(removed):
                redundant = True
                for f in edge_faces[e]:
                    count = sum(1 for ee in face_edge_lists[f] if ee in removed)
                    if count == 1:
                        redundant = False
                        break
                if redundant:
                    removed.discard(e)
                    improved = True

            if time.time() >= time_limit:
                break

            # 2. Try single swaps (exhaustive over removed, random candidate)
            if removed:
                for e in list(removed):
                    critical_faces = []
                    for f in edge_faces[e]:
                        count = sum(1 for ee in face_edge_lists[f] if ee in removed)
                        if count == 1:
                            critical_faces.append(f)

                    if not critical_faces:
                        continue

                    inter = set(face_edge_lists[critical_faces[0]])
                    for f in critical_faces[1:]:
                        inter.intersection_update(face_edge_lists[f])

                    candidates = inter - removed
                    if candidates:
                        e_new = rng.choice(list(candidates))
                        removed.discard(e)
                        removed.add(e_new)
                        improved = True
                        break  # Restart after a swap

            if time.time() >= time_limit:
                break

            # 3. Try 2-swap moves: remove e1, e2 and add e3, e4
            # This helps escape deeper local minima
            if len(removed) >= 2:
                removed_list = list(removed)
                # Try a few random 2-swap attempts
                for _ in range(5):
                    if time.time() >= time_limit:
                        break
                    e1, e2 = rng.sample(removed_list, 2)

                    # Check if removing e1 and e2 keeps all faces hit
                    # i.e., for every face containing e1 or e2, there's another removed edge in that face
                    ok = True
                    for f in edge_faces[e1] | edge_faces[e2]:
                        count = sum(1 for ee in face_edge_lists[f] if ee in removed)
                        if count <= 2 and (e1 in face_edge_lists[f] or e2 in face_edge_lists[f]):
                            # Need at least one other removed edge in this face besides e1/e2
                            if count == 2:
                                ok = False
                                break
                            elif count == 1:
                                ok = False
                                break

                    if not ok:
                        continue

                    # Find candidate edges to add
                    # We want edges that help cover faces that might become uncovered
                    # For simplicity, just try random non-removed edges
                    non_removed = [e for e in range(m) if e not in removed]
                    if len(non_removed) < 2:
                        continue

                    e3, e4 = rng.sample(non_removed, 2)

                    # Tentatively apply
                    test_removed = removed - {e1, e2} | {e3, e4}

                    # Verify all faces are still hit
                    valid = True
                    for f in range(F):
                        count = sum(1 for ee in face_edge_lists[f] if ee in test_removed)
                        if count == 0:
                            valid = False
                            break

                    if valid:
                        removed = test_removed
                        improved = True
                        break

        return removed

    def solve_once(seed):
        rng = random.Random(seed)
        removed = greedy_hitting_set(rng)
        removed = local_search(removed, rng, deadline - 5.0)
        return removed

    # Run multiple restarts
    best_removed = None
    best_size = m

    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        removed = solve_once(i)
        if len(removed) < best_size:
            best_size = len(removed)
            best_removed = removed.copy()

    # Final polish: try to further reduce the best solution
    if best_removed is not None:
        rng = random.Random(99999)
        best_removed = local_search(best_removed, rng, deadline - 2.0)
        best_size = len(best_removed)

    kept_edges = [edges[i] for i in range(m) if i not in best_removed]
    return kept_edges

"""C4-free edge set in Q7 using greedy + local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    d = 7
    n = 1 << d
    total_edges = d * (1 << (d - 1))

    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx

    m = len(edges)

    face_edges = {}
    edge_faces = [[] for _ in range(m)]
    face_id = 0
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for base in range(n):
                if base & mask != 0:
                    continue
                v0 = base
                v1 = base ^ (1 << i)
                v2 = base ^ (1 << j)
                v3 = base ^ mask
                e0 = edge_index[(min(v0, v1), max(v0, v1))]
                e1 = edge_index[(min(v0, v2), max(v0, v2))]
                e2 = edge_index[(min(v1, v3), max(v1, v3))]
                e3 = edge_index[(min(v2, v3), max(v2, v3))]
                face_edges[face_id] = [e0, e1, e2, e3]
                for e in [e0, e1, e2, e3]:
                    edge_faces[e].append(face_id)
                face_id += 1

    num_faces = face_id

    def greedy_construct(seed):
        rng = random.Random(seed)
        edge_order = list(range(m))
        rng.shuffle(edge_order)

        selected = np.zeros(m, dtype=bool)
        face_count = np.zeros(num_faces, dtype=np.int8)
        count = 0

        for e in edge_order:
            ok = True
            for fid in edge_faces[e]:
                if face_count[fid] == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
                for fid in edge_faces[e]:
                    face_count[fid] += 1

        return selected, count, face_count

    def local_search(selected, count, face_count, rng, time_limit=5.0):
        ls_start = time.time()
        improved = True
        while improved and time.time() < ls_start + time_limit and time.time() < deadline:
            improved = False
            unselected = [e for e in range(m) if not selected[e]]
            rng.shuffle(unselected)
            for e in unselected:
                if time.time() > deadline:
                    break
                blocking_faces = []
                for fid in edge_faces[e]:
                    if face_count[fid] == 3:
                        blocking_faces.append(fid)
                if not blocking_faces:
                    selected[e] = True
                    count += 1
                    for fid in edge_faces[e]:
                        face_count[fid] += 1
                    improved = True
                    break
                for fid in blocking_faces:
                    if time.time() > deadline:
                        break
                    face_e = face_edges[fid]
                    others = [fe for fe in face_e if fe != e]
                    for oe in others:
                        if not selected[oe]:
                            continue
                        still_blocked = False
                        for bf in blocking_faces:
                            if face_count[bf] == 3:
                                if oe in face_edges[bf]:
                                    if face_count[bf] - 1 >= 3:
                                        still_blocked = True
                                        break
                                else:
                                    still_blocked = True
                                    break
                        if not still_blocked:
                            selected[oe] = False
                            for bf in edge_faces[oe]:
                                face_count[bf] -= 1
                            selected[e] = True
                            count += 1
                            for bf in edge_faces[e]:
                                face_count[bf] += 1
                            improved = True
                            break
                    if improved:
                        break
        return selected, count, face_count

    best_selected = None
    best_count = 0
    best_face_count = None

    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() > deadline - 15:
            break
        seed = attempt * 12345 + 67890
        selected, count, face_count = greedy_construct(seed)
        rng = random.Random(seed + 1)
        selected, count, face_count = local_search(selected, count, face_count, rng, time_limit=3.0)
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
            best_face_count = face_count.copy()

    if time.time() < deadline - 5 and best_selected is not None:
        rng = random.Random(42)
        best_selected, best_count, best_face_count = local_search(
            best_selected, best_count, best_face_count, rng, time_limit=30.0
        )

    if best_selected is None:
        return []

    out = []
    for e in range(m):
        if best_selected[e]:
            out.append(list(edges[e]))

    return out

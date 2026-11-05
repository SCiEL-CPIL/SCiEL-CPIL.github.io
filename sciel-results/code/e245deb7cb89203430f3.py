# EVOLVE-BLOCK-START
import time
import random
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0   # 10s safety margin before 150s kill

    d = 7
    n = 2 ** d                       # 128 vertices
    n_edges = d * 2 ** (d - 1)       # 448

    # ---- Precompute edges and faces (identical to current program) ----
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx

    # A 2-face is a 4-cycle; it is "full" (forbidden) only when all 4 edges are selected.
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e02 = edge_to_idx[(min(v0, v2), max(v0, v2))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    e23 = edge_to_idx[(min(v2, v3), max(v2, v3))]
                    faces.append((e01, e02, e13, e23))

    n_faces = len(faces)

    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)

    def can_add(selected, face_count, e):
        for f in edge_faces[e]:
            if face_count[f] >= 3:
                return False
        return True

    # ---- Randomized greedy initializer (lazy max-heap for speed) ----
    def greedy_init(rng):
        selected = [False] * n_edges
        face_count = [0] * n_faces
        remaining = [len(edge_faces[e]) for e in range(n_edges)]
        heap = []
        for e in range(n_edges):
            heapq.heappush(heap, (-remaining[e], e))
        count = 0
        while heap:
            if time.time() > deadline - 20.0:
                break
            neg, e = heapq.heappop(heap)
            if selected[e] or remaining[e] == 0:
                continue
            if not can_add(selected, face_count, e):
                continue
            selected[e] = True
            count += 1
            for f in edge_faces[e]:
                if face_count[f] == 0:
                    face_count[f] = 1
                    for e2 in faces[f]:
                        if not selected[e2]:
                            remaining[e2] -= 1
                            if remaining[e2] > 0:
                                heapq.heappush(heap, (-remaining[e2], e2))
        return selected, face_count, count

    # ---- Redundancy removal: drop any edge that is not the sole protector of a face ----
    def minimize(selected, face_count, rng):
        count = 0
        for e in range(n_edges):
            if selected[e]:
                count += 1
        order = [e for e in range(n_edges) if selected[e]]
        rng.shuffle(order)
        for e in order:
            if not selected[e]:
                continue
            sole = False
            for f in edge_faces[e]:
                if face_count[f] == 1:
                    sole = True
                    break
            if not sole:
                selected[e] = False
                count -= 1
                for f in edge_faces[e]:
                    face_count[f] -= 1
        return count

    # ---- Repair: greedily add edges to clear any face that currently has all 4 edges ----
    def repair(selected, face_count, rng):
        count = 0
        for e in range(n_edges):
            if selected[e]:
                count += 1
        while True:
            if time.time() > deadline - 5.0:
                break
            full = [f for f in range(n_faces) if face_count[f] >= 4]
            if not full:
                break
            f = rng.choice(full)
            best_e = -1
            best_gain = -1
            for e in faces[f]:
                if not selected[e]:
                    gain = 0
                    for f2 in edge_faces[e]:
                        if face_count[f2] == 3:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_e = e
            if best_e < 0:
                for e in faces[f]:
                    if not selected[e]:
                        best_e = e
                        break
            if best_e < 0:
                break
            selected[best_e] = True
            count += 1
            for f2 in edge_faces[best_e]:
                face_count[f2] += 1
        return count

    # ---- Add every edge that is currently addable, then minimize ----
    def add_all_and_min(selected, face_count, rng):
        changed = True
        while changed:
            changed = False
            for e in range(n_edges):
                if not selected[e] and can_add(selected, face_count, e):
                    selected[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    changed = True
        count = 0
        for e in range(n_edges):
            if selected[e]:
                count += 1
        count = minimize(selected, face_count, rng)
        return count

    # ---- Main loop ----
    best_selected = None
    best_count = -1
    rng = random.Random(12345)

    # Multiple randomized greedy starts, keep the largest
    while time.time() < deadline - 30.0:
        sel, fc, cnt = greedy_init(rng)
        cnt = minimize(sel, fc, rng)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel[:]
            best_fc = fc[:]

    # Destroy-repair simulated annealing on the edge set
    sel = best_selected[:]
    fc = best_fc[:]
    cur = best_count
    T = 4.0
    T_min = 0.15
    cooling = 0.99995
    while time.time() < deadline - 8.0:
        if T < T_min:
            T = 4.0
        # destroy: remove a random handful of selected edges
        k = rng.randint(2, 8)
        for _ in range(k):
            while True:
                e = rng.randrange(n_edges)
                if sel[e]:
                    break
            sel[e] = False
            cur -= 1
            for f in edge_faces[e]:
                fc[f] -= 1
        # repair + minimize
        cur = repair(sel, fc, rng)
        cur = minimize(sel, fc, rng)
        delta = cur - best_count
        if delta >= 0 or rng.random() < math.exp(delta / T):
            if cur > best_count:
                best_count = cur
                best_selected = sel[:]
                best_fc = fc[:]
        T *= cooling

    # Final polish: add all addable edges, then remove redundancies
    sel = best_selected[:]
    fc = best_fc[:]
    final = add_all_and_min(sel, fc, rng)
    if final > best_count:
        best_count = final
        best_selected = sel[:]

    result = []
    for e in range(n_edges):
        if best_selected[e]:
            result.append(list(edges[e]))
    return result
# EVOLVE-BLOCK-END

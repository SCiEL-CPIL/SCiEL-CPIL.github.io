import time

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    d = 7
    N = 1 << d
    # Build edges
    edges = []
    edge_index = {}
    for x in range(N):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                key = (x, y)
                edge_index[key] = len(edges)
                edges.append(key)
    E = len(edges)  # 448

    # Build faces: a face is (x, i, j) with i<j, x has bits i,j = 0
    # face -> list of 4 edge indices
    face_edges = {}   # face_id -> [e0,e1,e2,e3]
    edge_faces = [[] for _ in range(E)]  # edge -> list of face ids
    fid = 0
    for x in range(N):
        for i in range(d):
            if x & (1 << i):
                continue
            for j in range(i + 1, d):
                if x & (1 << j):
                    continue
                a = x
                b = x ^ (1 << i)
                c = x ^ (1 << j)
                e2 = x ^ (1 << i) ^ (1 << j)
                f = [edge_index[(min(a,b),max(a,b))],
                     edge_index[(min(a,c),max(a,c))],
                     edge_index[(min(b,e2),max(b,e2))],
                     edge_index[(min(c,e2),max(c,e2))]]
                face_edges[fid] = f
                for e in f:
                    edge_faces[e].append(fid)
                fid += 1
    F = fid

    # present set as bool array
    present = [False] * E
    face_count = [0] * F  # number of present edges in face

    def add_edge(e):
        present[e] = True
        for f in edge_faces[e]:
            face_count[f] += 1

    def remove_edge(e):
        present[e] = False
        for f in edge_faces[e]:
            face_count[f] -= 1

    # Greedy init: random-ish order to avoid bad local minima
    import random
    rng = random.Random(12345)
    order = list(range(E))
    rng.shuffle(order)
    for e in order:
        f = edge_faces[e]
        ok = True
        for fid2 in f:
            if face_count[fid2] == 3:
                ok = False
                break
        if ok:
            add_edge(e)

    count = sum(present)

    def time_left():
        return DEADLINE - (time.time() - t0) > 2.0

    # Local search
    # (a) add all non-conflicting missing edges repeatedly
    while time_left():
        progress = False
        for e in range(E):
            if present[e]:
                continue
            f = edge_faces[e]
            ok = True
            for fid2 in f:
                if face_count[fid2] == 3:
                    ok = False
                    break
            if ok:
                add_edge(e)
                count += 1
                progress = True
        if not progress:
            break

    # (b) 1-remove-k-add moves
    # For each missing edge, compute blocking present edges
    def compute_blockers():
        # returns dict missing_edge -> list of present edges that block it
        # m is addable iff for all faces f of m, face_count[f] < 3 (since m not present)
        # Actually m addable iff no face of m has count 3 (the other 3 present)
        # blocked by: for each face f of m with count==3, the 3 present edges in f
        blockers = {}
        for e in range(E):
            if present[e]:
                continue
            bl = set()
            for fid2 in edge_faces[e]:
                if face_count[fid2] == 3:
                    for e2 in face_edges[fid2]:
                        if present[e2]:
                            bl.add(e2)
            if bl:
                blockers[e] = bl
        return blockers

    while time_left():
        improved = False
        blockers = compute_blockers()
        # For each present edge, count how many missing edges it is the sole blocker of
        # An edge e blocks missing m if e in blockers[m]. m becomes addable when
        # all its blockers are removed. For a 1-removal move removing e, m becomes
        # addable iff blockers[m] == {e}.
        sole = {}
        for m, bl in blockers.items():
            if len(bl) == 1:
                e = next(iter(bl))
                sole.setdefault(e, []).append(m)
        for e, ms in sole.items():
            if not present[e]:
                continue
            if len(ms) >= 2:
                # remove e, add all in ms that become addable
                remove_edge(e)
                count -= 1
                added = 0
                for m in ms:
                    f = edge_faces[m]
                    ok = True
                    for fid2 in f:
                        if face_count[fid2] == 3:
                            ok = False
                            break
                    if ok:
                        add_edge(m)
                        count += 1
                        added += 1
                if added > 1:
                    improved = True
                # also try to add more non-conflicting
                # (loop (a) will catch them next iteration)
        if not improved:
            break

    # final sweep
    while time_left():
        progress = False
        for e in range(E):
            if present[e]:
                continue
            f = edge_faces[e]
            ok = True
            for fid2 in f:
                if face_count[fid2] == 3:
                    ok = False
                    break
            if ok:
                add_edge(e)
                count += 1
                progress = True
        if not progress:
            break

    out = [[edges[e][0], edges[e][1]] for e in range(E) if present[e]]
    return out

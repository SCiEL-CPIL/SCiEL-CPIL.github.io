"""C4-free edge set of Q7 via local search (remove-then-readd)."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute global data once at import time.
_D = 7
_N = 1 << _D  # 128
_E = _D * _N // 2  # 448

# Edge index: (u, v) with u < v -> idx. Build via bit-position.
# Edge (u, u ^ (1<<i)) for i in 0..6, u < u^(1<<i).
_EDGE_U = np.empty(_E, dtype=np.int32)
_EDGE_V = np.empty(_E, dtype=np.int32)
_EDGE_BIT = np.empty(_E, dtype=np.int32)
_EDGE_IDX = {}
_k = 0
for u in range(_N):
    for i in range(_D):
        v = u ^ (1 << i)
        if u < v:
            _EDGE_U[_k] = u
            _EDGE_V[_k] = v
            _EDGE_BIT[_k] = i
            _EDGE_IDX[(u, v)] = _k
            _k += 1

# Faces: for each (i<j) and base x with bits i,j == 0.
# Face = (u00, u10, u01, u11) where u10=u00^ei, u01=u00^ej, u11=u00^ei^ej.
# Edges: e01=(u00,u10) bit i, e02=(u00,u01) bit j, e13=(u10,u11) bit j, e23=(u01,u11) bit i
_FACE_EDGES = []  # list of 4-tuples of edge indices
for i in range(_D):
    for j in range(i + 1, _D):
        ei, ej = 1 << i, 1 << j
        for x in range(_N):
            if (x & ei) == 0 and (x & ej) == 0:
                u00 = x
                u10 = x ^ ei
                u01 = x ^ ej
                u11 = x ^ ei ^ ej
                f = (
                    _EDGE_IDX[(min(u00, u10), max(u00, u10))],
                    _EDGE_IDX[(min(u00, u01), max(u00, u01))],
                    _EDGE_IDX[(min(u10, u11), max(u10, u11))],
                    _EDGE_IDX[(min(u01, u11), max(u01, u11))],
                )
                _FACE_EDGES.append(f)
_FACES = np.array(_FACE_EDGES, dtype=np.int32)  # shape (672, 4)
_NFACES = _FACES.shape[0]

# For each edge, list of faces containing it (for fast violation updates).
_EDGE_FACES = [[] for _ in range(_E)]
for fi, f in enumerate(_FACE_EDGES):
    for e in f:
        _EDGE_FACES[e].append(fi)
_EDGE_FACES = [np.array(lst, dtype=np.int32) for lst in _EDGE_FACES]


def _greedy_initial(seed):
    """Randomized greedy: pick a random edge order, keep if no C4 completed."""
    rng = random.Random(seed)
    order = list(range(_E))
    rng.shuffle(order)
    in_set = np.zeros(_E, dtype=bool)
    face_count = np.zeros(_NFACES, dtype=np.int32)  # how many of the 4 edges present
    for e in order:
        faces = _EDGE_FACES[e]
        if np.any(face_count[faces] == 3):
            continue
        in_set[e] = True
        face_count[faces] += 1
    return in_set


def _local_search(in_set, seed, deadline):
    """Hill-climb: repeatedly remove an edge from a violated face, then re-add safe edges."""
    rng = random.Random(seed)
    in_set = in_set.copy()
    face_count = np.zeros(_NFACES, dtype=np.int32)
    for e in range(_E):
        if in_set[e]:
            face_count[_EDGE_FACES[e]] += 1
    best = int(in_set.sum())

    # Repeated sweeps
    while time.time() < deadline:
        improved = False
        # Phase A: remove one edge from each violated face until none violated.
        # We process faces; for each fully-present face, remove a random one of its 4 edges.
        for fi in range(_NFACES):
            if face_count[fi] == 4:
                f = _FACES[fi]
                e = int(f[rng.randrange(4)])
                if in_set[e]:
                    in_set[e] = False
                    fc = _EDGE_FACES[e]
                    face_count[fc] -= 1
                    improved = True
        # Phase B: re-add any edge that is absent and whose faces are not at 3.
        # Build the set of candidate edges to try: those absent.
        absent = np.where(~in_set)[0]
        if absent.size:
            # For each absent edge, check if any of its faces has count==3.
            # Vectorize: for each absent edge, we need max face_count over its faces.
            # Do it in chunks for memory safety.
            chunk = 512
            for s in range(0, len(absent), chunk):
                sub = absent[s:s + chunk]
                # gather face indices
                fidx = np.concatenate([_EDGE_FACES[int(e)] for e in sub])
                fcounts = face_count[fidx]
                # split per edge
                starts = np.concatenate([[0], np.cumsum([len(_EDGE_FACES[int(e)]) for e in sub])])
                # max per edge
                maxc = np.empty(len(sub), dtype=np.int32)
                for k, e in enumerate(sub):
                    sl = slice(starts[k], starts[k + 1])
                    maxc[k] = fcounts[sl].max()
                can_add = maxc < 3
                for k, e in enumerate(sub):
                    if can_add[k]:
                        in_set[int(e)] = True
                        fc = _EDGE_FACES[int(e)]
                        face_count[fc] += 1
                        improved = True
        cur = int(in_set.sum())
        if cur > best:
            best = cur
        if not improved:
            break
    return in_set, best


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    best_in_set = None
    best_count = -1

    # First, a deterministic greedy baseline (seed 0) to guarantee a valid set.
    base = _greedy_initial(0)
    base, cnt = _local_search(base, 0, min(deadline, time.time() + 20.0))
    best_in_set, best_count = base, cnt

    # Random restarts within budget.
    trial = 1
    while time.time() < deadline - 5.0:
        # Reserve a bit of time per trial; stop if too little remains.
        if time.time() > deadline - 8.0:
            break
        init = _greedy_initial(trial)
        in_set, cnt = _local_search(init, 1000 + trial, deadline)
        if cnt > best_count:
            best_count = cnt
            best_in_set = in_set
        trial += 1
        if trial > 2000:  # hard cap on restarts
            break

    # Convert to list of edges.
    out = []
    for e in range(_E):
        if best_in_set[e]:
            out.append([int(_EDGE_U[e]), int(_EDGE_V[e])])
    return out

# EVOLVE-BLOCK-END

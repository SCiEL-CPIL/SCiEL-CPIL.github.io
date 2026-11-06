"""Max-SAT-inspired greedy: start from K16, enumerate all Q3 embeddings,
remove edges that break the most remaining Q3s until Q3-free."""
import itertools

def _cube_edges():
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                edges.append((a, b))
    return edges

_CUBE = _cube_edges()  # 12 edges

def _all_embeddings():
    # Canonical labeling: lexicographically smallest permutation of the 8-set
    # assigned to cube corners 0..7. Any bijection is a valid cube labeling.
    verts = range(16)
    embeddings = []
    for combo in itertools.combinations(verts, 8):
        best = None
        for perm in itertools.permutations(combo):
            if best is None or perm < best:
                best = perm
        embeddings.append(best)
    return embeddings

def construct():
    V = 16
    all_edges = [(i, j) for i in range(V) for j in range(i + 1, V)]
    edge_index = {e: k for k, e in enumerate(all_edges)}
    n_edges = len(all_edges)

    emb_edges = []
    for labeling in _all_embeddings():
        ee = []
        for (a, b) in _CUBE:
            u, v = labeling[a], labeling[b]
            if u > v:
                u, v = v, u
            ee.append(edge_index[(u, v)])
        emb_edges.append(ee)
    n_emb = len(emb_edges)

    present = [1] * n_edges
    emb_count = [12] * n_emb
    edge_embs = [[] for _ in range(n_edges)]
    for e_idx, ee in enumerate(emb_edges):
        for ei in ee:
            edge_embs[ei].append(e_idx)
    edge_full = [len(edge_embs[ei]) for ei in range(n_edges)]

    full_embs = set(range(n_emb))

    while full_embs:
        emb = next(iter(full_embs))
        best_e = None
        best_val = -1
        for ei in emb_edges[emb]:
            if present[ei]:
                if edge_full[ei] > best_val:
                    best_val = edge_full[ei]
                    best_e = ei
        if best_e is None:
            break
        present[best_e] = 0
        for e2 in edge_embs[best_e]:
            if emb_count[e2] == 12:
                emb_count[e2] = 11
                full_embs.discard(e2)
                for ei2 in emb_edges[e2]:
                    if ei2 != best_e:
                        edge_full[ei2] -= 1
            else:
                emb_count[e2] -= 1
        edge_full[best_e] = 0

    result = []
    for k, (u, v) in enumerate(all_edges):
        if present[k]:
            result.append([u, v])
    return result

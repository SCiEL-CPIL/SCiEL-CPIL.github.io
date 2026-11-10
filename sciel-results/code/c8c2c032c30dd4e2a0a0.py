import random
import time

def construct():
    d = 7
    n_vertices = 1 << d

    # Generate all 448 edges of Q7
    all_edges = []
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v))

    # Precompute, for each edge, the 6 other edges that share a 2-face with it.
    # Adding edge e creates a C4 iff any of its 6 face-neighbours are already
    # selected.  (6 = 6 other coordinate axes.)
    face_neighbors = {}
    for u, v in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        nbrs = set()
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            nbrs.add((min(u, w1), max(u, w1)))
            nbrs.add((min(v, w2), max(v, w2)))
            nbrs.add((min(w1, w2), max(w1, w2)))
        face_neighbors[(u, v)] = nbrs

    def can_add(e, selected):
        return not any(n in selected for n in face_neighbors[e])

    def greedy_fill(seed, order):
        """Start from a seed set (must be C4-free) and greedily add every edge
        in `order` that does not complete a C4.  Returns a C4-free set."""
        s = set(seed)
        for e in order:
            if e in s:
                continue
            if can_add(e, s):
                s.add(e)
        return s

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin vs the 150s kill

    # --- Phase 0: a couple of quick pure greedy runs to get a warm start ---
    best_set = set()
    for _ in range(2):
        if time.time() > deadline:
            break
        random.shuffle(all_edges)
        s = greedy_fill(set(), all_edges)
        if len(s) > len(best_set):
            best_set = s

    # --- Phase 1: randomized greedy (repeated) ---
    while time.time() < deadline:
        random.shuffle(all_edges)
        s = greedy_fill(set(), all_edges)
        if len(s) > len(best_set):
            best_set = s

    # --- Phase 2: local search via multi-removal + greedy refill ---
    # Remove a small random batch of edges from the best set, then greedily
    # refill the freed space.  This is a cheap way to escape local optima that
    # a single swap cannot, and each iteration is O(448 * 6) membership tests.
    while time.time() < deadline:
        k = random.randint(1, 3)
        cur = set(best_set)
        removed = random.sample(list(cur), k)
        for e in removed:
            cur.discard(e)
        random.shuffle(all_edges)
        s = greedy_fill(cur, all_edges)
        if len(s) > len(best_set):
            best_set = s

    return [[u, v] for u, v in best_set]

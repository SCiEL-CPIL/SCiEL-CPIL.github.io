import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    d = 7
    n_vertices = 1 << d
    m = n_vertices * d // 2  # 448

    # Precompute edges: index -> (u, v), and (u,v) -> index
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_map[(u, v)] = idx
                idx += 1

    # Precompute faces: each face = (e0, e1, e2, e3)
    # Also for each edge, list of (face_id, the other 3 edges in that face)
    face_edges = []
    edge_face_info = [[] for _ in range(m)]  # edge -> list of (face_id, other3)

    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                if ((x >> i) & 1) == 0 and ((x >> j) & 1) == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    dd = x ^ (1 << i) ^ (1 << j)

                    e_ab = edge_map[(min(a, b), max(a, b))]
                    e_bd = edge_map[(min(b, dd), max(b, dd))]
                    e_dc = edge_map[(min(dd, c), max(dd, c))]
                    e_ca = edge_map[(min(c, a), max(c, a))]

                    fid = len(face_edges)
                    face_edges.append((e_ab, e_bd, e_dc, e_ca))

                    # For each edge in this face, record the other 3
                    for pos, e in enumerate((e_ab, e_bd, e_dc, e_ca)):
                        others = [face_edges[fid][k] for k in range(4) if k != pos]
                        edge_face_info[e].append((fid, tuple(others)))

    n_faces = len(face_edges)

    def is_valid_add(edge_idx, selected):
        """Check if adding edge_idx to selected keeps the set C4-free."""
        for (fid, others) in edge_face_info[edge_idx]:
            # If all 3 others are in selected, adding edge_idx completes a C4
            if all(o in selected for o in others):
                return False
        return True

    def greedy_fill(selected, candidate_order):
        """Add as many edges as possible from candidate_order (in that order)."""
        for idx in candidate_order:
            if idx not in selected and is_valid_add(idx, selected):
                selected.add(idx)
        return selected

    def removal_gain(edge_idx, selected):
        """Number of currently-forbidden edges that would become addable if edge_idx were removed.
        For each face containing edge_idx, if exactly 2 of the other 3 are in selected,
        the 3rd (not in selected) is blocked by edge_idx (among others). Count it."""
        gain = 0
        for (fid, others) in edge_face_info[edge_idx]:
            in_count = 0
            missing = None
            for o in others:
                if o in selected:
                    in_count += 1
                else:
                    missing = o
            if in_count == 2:
                # The missing edge is blocked; removing edge_idx unblocks it
                # (it may also be blocked by other edges, but we count potential)
                gain += 1
        return gain

    def compute_best_removal(selected, sample_size=None):
        """Find the edge in selected whose removal gives the highest gain.
        If sample_size is set, only consider a random sample of that size."""
        if not selected:
            return None, 0
        candidates = list(selected)
        if sample_size is not None and len(candidates) > sample_size:
            candidates = random.sample(candidates, sample_size)
        best_e = None
        best_g = -1
        for e in candidates:
            g = removal_gain(e, selected)
            if g > best_g:
                best_g = g
                best_e = e
        return best_e, best_g

    all_indices = list(range(m))

    best_set = set()
    best_size = 0

    # Phase 1: Multiple random greedy fills to find a good starting point
    for seed in [42, 123, 456, 789, 1024, 2048, 333, 777, 999, 111, 2222, 3333]:
        if time.time() - start_time > time_limit:
            break
        random.seed(seed)
        current = set()
        order = list(all_indices)
        random.shuffle(order)
        greedy_fill(current, order)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)

    # Phase 2: Local search with purposeful removals and SA
    current = set(best_set)
    cur_size = len(current)

    # SA parameters
    T0 = 2.0
    T_min = 0.05
    cooling = 0.995

    T = T0
    no_improve = 0
    max_no_improve = 2000

    while time.time() - start_time < time_limit:
        if not current:
            break

        # Save state
        saved = set(current)
        saved_size = len(current)

        # Choose an edge to remove: prefer high-gain, with some randomness
        # Sample up to 60 candidates for speed
        e_remove, g = compute_best_removal(current, sample_size=60)
        if e_remove is None:
            e_remove = random.choice(list(current))

        current.discard(e_remove)

        # Greedily add back: try all edges not in current in random order
        # But that's O(m * check). To speed up, first try edges that were "unblocked"
        # Simpler: shuffle all not-in-current and greedily add
        remaining = [i for i in all_indices if i not in current]
        random.shuffle(remaining)
        greedy_fill(current, remaining)

        new_size = len(current)

        # SA accept/reject
        if new_size > cur_size:
            cur_size = new_size
            no_improve = 0
            if cur_size > best_size:
                best_size = cur_size
                best_set = set(current)
        elif new_size == cur_size:
            # Accept with moderate probability (random walk)
            if random.random() < 0.5:
                cur_size = new_size
            else:
                current = set(saved)
        else:
            # new_size < cur_size: accept with probability exp(-(cur_size - new_size)/T)
            delta = cur_size - new_size
            prob = min(1.0, 2.718281828 ** (-delta / max(T, 1e-6)))
            if random.random() < prob:
                cur_size = new_size
            else:
                current = set(saved)

        # Cooling
        T = max(T_min, T * cooling)

        # If stuck for too long, do a kick: remove a few random edges and refill
        no_improve += 1
        if no_improve > max_no_improve:
            # Kick: remove 3 random edges, refill
            for _ in range(3):
                if current:
                    current.discard(random.choice(list(current)))
            remaining = [i for i in all_indices if i not in current]
            random.shuffle(remaining)
            greedy_fill(current, remaining)
            cur_size = len(current)
            no_improve = 0
            T = max(T, T0 * 0.5)  # reheat a bit

    # Final polish: try to add any remaining edge
    remaining = [i for i in all_indices if i not in best_set]
    random.shuffle(remaining)
    greedy_fill(best_set, remaining)

    # Convert to result
    result = []
    for idx in best_set:
        result.append(list(edges[idx]))

    return result

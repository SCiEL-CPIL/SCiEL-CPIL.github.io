import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for verification

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    # Q3 template edges on vertices 0..7
    q3_template_edges = []
    for v in range(8):
        for bit, mask in [(0, 4), (1, 2), (2, 1)]:
            other = v ^ mask
            if other > v:
                q3_template_edges.append((v, other))
    assert len(q3_template_edges) == 12

    # Generate sampled Q3 embeddings
    random.seed(12345)
    seen_edge_sets = set()
    q3_embeddings = []
    num_q3_target = 500000
    attempts = 0
    max_attempts = num_q3_target * 5
    while len(q3_embeddings) < num_q3_target and attempts < max_attempts:
        attempts += 1
        target = random.sample(range(16), 8)
        edge_set = set()
        for u, v in q3_template_edges:
            a, b = target[u], target[v]
            if a > b:
                a, b = b, a
            edge_set.add((a, b))
        edge_tuple = tuple(sorted(edge_set))
        if edge_tuple not in seen_edge_sets:
            seen_edge_sets.add(edge_tuple)
            idx_tuple = tuple(sorted(edge_index[e] for e in edge_set))
            q3_embeddings.append(idx_tuple)

    num_q3 = len(q3_embeddings)

    # Edge -> Q3 index incidence
    edge_to_q3 = [[] for _ in range(num_edges)]
    for q3_idx, edge_indices in enumerate(q3_embeddings):
        for ei in edge_indices:
            edge_to_q3[ei].append(q3_idx)

    # SA with combined objective: minimize violations, maximize edges
    # score = num_edges - 10 * violations (we want high score)
    # Equivalent: minimize (10*violations - num_edges)

    random.seed(777)
    present = [False] * num_edges
    for idx in range(num_edges):
        if random.random() < 0.7:
            present[idx] = True

    def compute_violations(present_arr):
        v = 0
        for edge_indices in q3_embeddings:
            all_p = True
            for ei in edge_indices:
                if not present_arr[ei]:
                    all_p = False
                    break
            if all_p:
                v += 1
        return v

    current_violations = compute_violations(present)
    current_edges = sum(present)
    best_score = current_edges - 10 * current_violations
    best_present = list(present)

    T = 3.0
    cooling = 0.9998
    min_T = 0.005

    def check_best(present_arr, violations, edges):
        nonlocal best_score, best_present
        score = edges - 10 * violations
        if score > best_score:
            best_score = score
            best_present = list(present_arr)

    check_best(present, current_violations, current_edges)

    step = 0
    while time.time() < deadline:
        T *= cooling
        if T < min_T:
            T = min_T

        ei = random.randint(0, num_edges - 1)
        if present[ei]:
            # Remove
            de_viol = 0
            for q3_idx in edge_to_q3[ei]:
                all_others = True
                for oe in q3_embeddings[q3_idx]:
                    if oe != ei and not present[oe]:
                        all_others = False
                        break
                if all_others:
                    de_viol += 1
            new_viol = current_violations - de_viol
            new_edges = current_edges - 1
            new_score = new_edges - 10 * new_viol
            cur_score = current_edges - 10 * current_violations
            if new_score >= cur_score or random.random() < pow(2.71828, (cur_score - new_score) / T):
                present[ei] = False
                current_violations = new_viol
                current_edges = new_edges
                check_best(present, current_violations, current_edges)
        else:
            # Add
            de_viol = 0
            for q3_idx in edge_to_q3[ei]:
                all_others = True
                for oe in q3_embeddings[q3_idx]:
                    if oe != ei and not present[oe]:
                        all_others = False
                        break
                if all_others:
                    de_viol += 1
            new_viol = current_violations + de_viol
            new_edges = current_edges + 1
            new_score = new_edges - 10 * new_viol
            cur_score = current_edges - 10 * current_violations
            if new_score >= cur_score or random.random() < pow(2.71828, (cur_score - new_score) / T):
                present[ei] = True
                current_violations = new_viol
                current_edges = new_edges
                check_best(present, current_violations, current_edges)

        step += 1

    # Use best solution found
    best_set = set(i for i in range(num_edges) if best_present[i])

    # Verification: ensure no Q3 in the final graph using full check
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for ei in best_set:
        u, v = all_edges[ei]
        adj[u].add(v)
        adj[v].add(u)

    def find_q3(adj):
        for v in range(n):
            nbrs = list(adj[v])
            if len(nbrs) < 3:
                continue
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    for k in range(j + 1, len(nbrs)):
                        a, b, c = nbrs[i], nbrs[j], nbrs[k]
                        if b in adj[a] or c in adj[a] or c in adj[b]:
                            continue
                        forbidden = {v, a, b, c}
                        xs = (adj[a] & adj[b]) - forbidden
                        ys = (adj[a] & adj[c]) - forbidden
                        zs = (adj[b] & adj[c]) - forbidden
                        for x in xs:
                            for y in ys:
                                for z in zs:
                                    if x == y or y == z or x == z:
                                        continue
                                    if y in adj[x] or z in adj[x] or z in adj[y]:
                                        continue
                                    ws = (adj[x] & adj[y] & adj[z]) - forbidden - {x, y, z}
                                    for w in ws:
                                        if a in adj[w] or b in adj[w] or c in adj[w]:
                                            continue
                                        return [(v, a), (v, b), (v, c),
                                                (a, x), (b, x),
                                                (a, y), (c, y),
                                                (b, z), (c, z),
                                                (x, w), (y, w), (z, w)]
        return None

    # Remove edges until no Q3
    guard = 0
    while guard < 200:
        if time.time() > deadline:
            break
        q3 = find_q3(adj)
        if q3 is None:
            break
        # Remove random edge from Q3
        eu, ev = random.choice(q3)
        e = (min(eu, ev), max(eu, ev))
        if e in best_set:
            best_set.remove(edge_index[e])
            adj[eu].discard(ev)
            adj[ev].discard(eu)
        guard += 1

    result = [list(all_edges[i]) for i in sorted(best_set)]
    return result

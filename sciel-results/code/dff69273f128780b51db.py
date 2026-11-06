import random
import time

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65

    # ---------- bitmask helpers ----------
    def score_graph(adj_mask, deg):
        """Return (max_sum, worst_triangle). worst_triangle = (u,v,w) or None."""
        max_sum = 0
        worst = None
        for u in range(n):
            mu = adj_mask[u]
            for v in range(u + 1, n):
                if not (mu >> v) & 1:
                    continue
                common = mu & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)  # only w > v
                while common:
                    w = common.bit_length() - 1
                    common ^= (1 << w)
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                        worst = (u, v, w)
        return max_sum, worst

    def edges_to_mask(edges):
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg

    def mask_to_edges(adj_mask):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edges.append([u, v])
        return edges

    def non_edges(adj_mask):
        out = []
        for u in range(n):
            m = adj_mask[u]
            for v in range(u + 1, n):
                if not (m >> v) & 1:
                    out.append((u, v))
        return out

    def edge_list(adj_mask):
        out = []
        for u in range(n):
            m = adj_mask[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    out.append((u, v))
        return out

    # ---------- swap evaluation ----------
    def do_swap(adj_mask, deg, add, rem):
        (u, v) = add
        (x, y) = rem
        adj_mask[u] |= (1 << v); adj_mask[v] |= (1 << u)
        adj_mask[x] &= ~(1 << y); adj_mask[y] &= ~(1 << x)
        deg[u] += 1; deg[v] += 1
        deg[x] -= 1; deg[y] -= 1

    def undo_swap(adj_mask, deg, add, rem):
        (u, v) = add
        (x, y) = rem
        adj_mask[u] &= ~(1 << v); adj_mask[v] &= ~(1 << u)
        adj_mask[x] |= (1 << y); adj_mask[y] |= (1 << x)
        deg[u] -= 1; deg[v] -= 1
        deg[x] += 1; deg[y] += 1

    # ---------- local search ----------
    def local_search(adj_mask, deg, edges, current_score, max_rounds=200000):
        rounds = 0
        while time.time() < deadline and rounds < max_rounds:
            rounds += 1
            # full scan for best improving swap
            best_gain = 0
            best_add = None
            best_rem = None
            ne = non_edges(adj_mask)
            el = edge_list(adj_mask)
            # degree-ordered: try non-edges involving low-degree vertices first
            ne_sorted = sorted(ne, key=lambda t: deg[t[0]] + deg[t[1]])
            # remove high-degree-sum edges first
            el_sorted = sorted(el, key=lambda t: -(deg[t[0]] + deg[t[1]]))
            for add in ne_sorted:
                u, v = add
                for rem in el_sorted:
                    x, y = rem
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    do_swap(adj_mask, deg, add, rem)
                    ns, _ = score_graph(adj_mask, deg)
                    undo_swap(adj_mask, deg, add, rem)
                    gain = current_score - ns
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add
                        best_rem = rem
                        if gain >= 3:  # big jump, take it
                            break
                if best_gain >= 3:
                    break
            if best_add is None:
                break
            do_swap(adj_mask, deg, best_add, best_rem)
            current_score -= best_gain
            # update edges
            (x, y) = best_rem
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append(list(best_add))
        return adj_mask, deg, edges, current_score

    # ---------- targeted worst-triangle reduction ----------
    def targeted_reduce(adj_mask, deg, edges, current_score, max_iters=200000):
        iters = 0
        while time.time() < deadline and iters < max_iters:
            iters += 1
            _, worst = score_graph(adj_mask, deg)
            if worst is None:
                break
            u, v, w = worst
            worst_sum = deg[u] + deg[v] + deg[w]
            # candidate removals: the three edges of the worst triangle
            # (removing any one breaks this triangle)
            cand_rems = [(u, v), (u, w), (v, w)]
            # candidate adds: non-edges with minimal degree sum
            ne = non_edges(adj_mask)
            ne_sorted = sorted(ne, key=lambda t: deg[t[0]] + deg[t[1]])
            best_gain = 0
            best_add = None
            best_rem = None
            for rem in cand_rems:
                x, y = rem
                for add in ne_sorted[:40]:  # only low-degree non-edges
                    a, b = add
                    if (a == x and b == y) or (a == y and b == x):
                        continue
                    do_swap(adj_mask, deg, add, rem)
                    ns, _ = score_graph(adj_mask, deg)
                    undo_swap(adj_mask, deg, add, rem)
                    gain = worst_sum - ns
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add
                        best_rem = rem
            if best_add is None:
                break
            do_swap(adj_mask, deg, best_add, best_rem)
            current_score -= best_gain
            (x, y) = best_rem
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append(list(best_add))
        return adj_mask, deg, edges, current_score

    # ---------- initial state: K_{8,8} + one edge ----------
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj_mask, deg = edges_to_mask(init_edges)
    edges = [e[:] for e in init_edges]
    current_score, _ = score_graph(adj_mask, deg)

    best_score = current_score
    best_adj = adj_mask[:]
    best_deg = deg[:]
    best_edges = [e[:] for e in edges]

    # alternating local search + targeted reduction
    while time.time() < deadline:
        adj_mask, deg, edges, current_score = local_search(
            adj_mask, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj_mask[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        if time.time() > deadline:
            break
        adj_mask, deg, edges, current_score = targeted_reduce(
            adj_mask, deg, edges, current_score, max_iters=30000)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj_mask[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]

        # random perturbation to escape local minima
        if time.time() > deadline:
            break
        adj_p = best_adj[:]
        deg_p = best_deg[:]
        edges_p = [e[:] for e in best_edges]
        for _ in range(4):
            ne = non_edges(adj_p)
            el = edge_list(adj_p)
            if not ne or not el:
                break
            add = random.choice(ne)
            rem = random.choice(el)
            if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
                continue
            do_swap(adj_p, deg_p, add, rem)
            for i, e in enumerate(edges_p):
                if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                    edges_p.pop(i)
                    break
            edges_p.append(list(add))
        adj_p, deg_p, edges_p, sc_p = local_search(adj_p, deg_p, edges_p, score_graph(adj_p, deg_p)[0], max_rounds=15000)
        adj_p, deg_p, edges_p, sc_p = targeted_reduce(adj_p, deg_p, edges_p, sc_p, max_iters=15000)
        if sc_p < best_score:
            best_score = sc_p
            best_adj = adj_p[:]
            best_deg = deg_p[:]
            best_edges = [e[:] for e in edges_p]

    # final validation
    final_adj, final_deg = edges_to_mask(best_edges)
    assert len(best_edges) == target_edges, f"edge count {len(best_edges)} != {target_edges}"
    return {'n': 16, 'edges': best_edges}

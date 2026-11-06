import numpy as np
import random
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    n = 20
    deadline = time.time() + 110

    def heuristic_fallback():
        best_edges = -1
        best_A = None

        def is_valid(A):
            K = A @ A
            for u in range(n):
                for v in range(u + 1, n):
                    if K[u, v] >= 3:
                        return False
            return True

        def try_add_edges(A, K, count):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for u in range(n):
                    if time.time() >= deadline - 5:
                        return A, K, count
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            ok = True
                            for x in range(n):
                                if A[u, x] == 1 and x != v:
                                    if K[x, v] >= 2:
                                        ok = False
                                        break
                            if ok:
                                for y in range(n):
                                    if A[v, y] == 1 and y != u:
                                        if K[u, y] >= 2:
                                            ok = False
                                            break
                            if ok:
                                A[u, v] = 1
                                A[v, u] = 1
                                for x in range(n):
                                    if A[u, x] == 1 and x != v:
                                        K[x, v] += 1
                                        K[v, x] += 1
                                for y in range(n):
                                    if A[v, y] == 1 and y != u:
                                        K[u, y] += 1
                                        K[y, u] += 1
                                count += 1
                                changed = True
                                break
            return A, K, count

        def remove_to_fix(A, K, count):
            while time.time() < deadline - 5:
                viol = False
                for u in range(n):
                    if time.time() >= deadline - 5:
                        break
                    for v in range(u + 1, n):
                        if K[u, v] >= 3:
                            viol = True
                            zs = [z for z in range(n) if A[u, z] == 1 and A[v, z] == 1]
                            random.shuffle(zs)
                            for z in zs:
                                A[u, z] = 0
                                A[z, u] = 0
                                for w in range(n):
                                    if A[u, w] == 1 and w != z:
                                        K[w, z] -= 1
                                        K[z, w] -= 1
                                for w in range(n):
                                    if A[z, w] == 1 and w != u:
                                        K[u, w] -= 1
                                        K[w, u] -= 1
                                count -= 1
                                break
                            break
                    if viol:
                        break
                if not viol:
                    break
            return A, K, count

        for restart in range(100):
            if time.time() >= deadline - 5:
                break
            A = np.zeros((n, n), dtype=int)
            for a in range(10):
                for b in range(a + 1, 10):
                    A[a, b] = 1
                    A[b, a] = 1
            for a in range(10, 20):
                for b in range(a + 1, 20):
                    A[a, b] = 1
                    A[b, a] = 1

            perm = list(range(n))
            random.shuffle(perm)
            A2 = np.zeros((n, n), dtype=int)
            for u in range(n):
                for v in range(n):
                    if A[u, v] == 1:
                        A2[perm[u], perm[v]] = 1
            A = A2
            K = A @ A
            count = 90

            A, K, count = remove_to_fix(A, K, count)
            A, K, count = try_add_edges(A, K, count)

            if count > best_edges:
                best_edges = count
                best_A = A.copy()

        if best_A is None:
            best_A = np.zeros((n, n), dtype=int)
            for k in range(5):
                for a in range(4):
                    for b in range(a + 1, 4):
                        u = 4 * k + a
                        v = 4 * k + b
                        best_A[u, v] = 1
                        best_A[v, u] = 1
        return best_A

    # MILP Formulation
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edges_list.append((u, v))
    num_edges = len(edges_list)

    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    num_aux_per_edge = n - 2
    total_aux = num_edges * num_aux_per_edge
    total_vars = num_edges + total_aux

    c = np.zeros(total_vars)
    c[:num_edges] = -1.0

    rows = []
    cols = []
    vals = []
    b_ub = []
    b_lb = []

    for i, (u, v) in enumerate(edges_list):
        edge_idx = i
        base_aux = num_edges + edge_idx * num_aux_per_edge

        # Constraint 4: sum y <= 2
        # Row: sum y_uvw <= 2
        row_idx = len(b_ub)
        w_count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            aux_idx = base_aux + w_count
            rows.append(row_idx)
            cols.append(aux_idx)
            vals.append(1)
            w_count += 1
        b_ub.append(2)
        b_lb.append(-np.inf)

        # Constraints 1, 2, 3 for each w
        w_count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            aux_idx = base_aux + w_count
            w_count += 1

            idx_uw = edge_index[(u, w)] if u < w else edge_index[(w, u)]
            idx_vw = edge_index[(v, w)] if v < w else edge_index[(w, v)]

            # C1: y_uvw - x_uw <= 0
            row_idx = len(b_ub)
            rows.append(row_idx)
            cols.append(aux_idx)
            vals.append(1)
            rows.append(row_idx)
            cols.append(idx_uw)
            vals.append(-1)
            b_ub.append(0)
            b_lb.append(-np.inf)

            # C2: y_uvw - x_vw <= 0
            row_idx = len(b_ub)
            rows.append(row_idx)
            cols.append(aux_idx)
            vals.append(1)
            rows.append(row_idx)
            cols.append(idx_vw)
            vals.append(-1)
            b_ub.append(0)
            b_lb.append(-np.inf)

            # C3: -y_uvw + x_uw + x_vw <= 1
            row_idx = len(b_ub)
            rows.append(row_idx)
            cols.append(aux_idx)
            vals.append(-1)
            rows.append(row_idx)
            cols.append(idx_uw)
            vals.append(1)
            rows.append(row_idx)
            cols.append(idx_vw)
            vals.append(1)
            b_ub.append(1)
            b_lb.append(-np.inf)

    A = csr_matrix((vals, (rows, cols)), shape=(len(b_ub), total_vars))
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(total_vars)

    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb=b_lb, ub=b_ub), integrality=integrality, bounds=bounds, options={"time_limit": 100})
    except Exception:
        res = None

    if res is not None and res.status == 0:
        x = res.x
        best_A = np.zeros((n, n), dtype=int)
        for i, (u, v) in enumerate(edges_list):
            if x[i] > 0.5:
                best_A[u, v] = 1
                best_A[v, u] = 1
        K = best_A @ best_A
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if K[u, v] >= 3:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_A[u, v] == 1:
                        edges.append([u, v])
            return edges

    best_A = heuristic_fallback()
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges

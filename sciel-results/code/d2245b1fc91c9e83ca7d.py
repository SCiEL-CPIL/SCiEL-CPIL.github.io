"""Structural approach: start with complete graph, use group-based coloring, repair via local search."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    N = 20
    C = 20

    def get_color(adj, u, v):
        if u in adj and v in adj[u]:
            return adj[u][v]
        if v in adj and u in adj[v]:
            return adj[v][u]
        return None

    def set_color(adj, u, v, c):
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c

    def remove_edge(adj, u, v):
        if u in adj and v in adj[u]:
            del adj[u][v]
        if v in adj and u in adj[v]:
            del adj[v][u]

    def find_violations(adj):
        """Find all 4-cycles that are NOT rainbow."""
        violations = []
        for a in range(N):
            for b in range(a + 1, N):
                if get_color(adj, a, b) is None:
                    continue
                for x in adj.get(a, {}):
                    if x == b:
                        continue
                    for y in adj.get(b, {}):
                        if y == a or y == x:
                            continue
                        if get_color(adj, x, y) is not None:
                            c_ab = get_color(adj, a, b)
                            c_bx = get_color(adj, b, x)
                            c_xy = get_color(adj, x, y)
                            c_ya = get_color(adj, y, a)
                            if None in (c_ab, c_bx, c_xy, c_ya):
                                continue
                            if len({c_ab, c_bx, c_xy, c_ya}) < 4:
                                violations.append((a, b, x, y, c_ab, c_bx, c_xy, c_ya))
        return violations

    def full_validate(adj):
        """Check all 4-cycles are rainbow."""
        return len(find_violations(adj)) == 0

    def adj_to_result(adj):
        result = []
        for u in range(N):
            for v in adj.get(u, {}):
                if v > u:
                    result.append([u, v, adj[u][v]])
        return result

    def edge_count(adj):
        return sum(len(v) for v in adj.values()) // 2

    def check_single_edge_valid(adj, u, v, c):
        """Check if adding edge (u,v) with colour c maintains rainbow 4-cycles."""
        for x in adj.get(u, {}):
            if x == v:
                continue
            for y in adj.get(v, {}):
                if y == u or y == x:
                    continue
                if get_color(adj, x, y) is not None:
                    c_vy = get_color(adj, v, y)
                    c_xy = get_color(adj, x, y)
                    c_xu = get_color(adj, x, u)
                    if c_vy is not None and c_xy is not None and c_xu is not None:
                        if len({c, c_vy, c_xy, c_xu}) < 4:
                            return False
        return True

    random.seed(42)
    best_result = []
    best_count = 0

    # Strategy 1: Try complete graph with various structured colorings
    # Use different algebraic structures for coloring

    colorings = []

    # Coloring 1: (u + v) mod C
    def coloring1(u, v):
        return (u + v) % C

    # Coloring 2: (u * v) mod C (careful with 0)
    def coloring2(u, v):
        return (u * v + 1) % C if u * v % C == 0 else (u * v) % C

    # Coloring 3: (u - v) mod C
    def coloring3(u, v):
        return (u - v) % C

    # Coloring 4: XOR-based
    def coloring4(u, v):
        return (u ^ v) % C

    # Coloring 5: (u + 2*v) mod C
    def coloring5(u, v):
        return (u + 2 * v) % C

    # Coloring 6: (2*u + v) mod C
    def coloring6(u, v):
        return (2 * u + v) % C

    # Coloring 7: (u * (v + 1)) mod C
    def coloring7(u, v):
        return (u * (v + 1)) % C

    # Coloring 8: (u + v * v) mod C
    def coloring8(u, v):
        return (u + v * v) % C

    colorings = [coloring1, coloring2, coloring3, coloring4, coloring5, coloring6, coloring7, coloring8]

    for coloring_func in colorings:
        if time.time() - start_time > DEADLINE - 30:
            break

        adj = {}
        for u in range(N):
            for v in range(u + 1, N):
                c = coloring_func(u, v)
                set_color(adj, u, v, c)

        # Check if this coloring works for the complete graph
        violations = find_violations(adj)
        if len(violations) == 0:
            result = adj_to_result(adj)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result
            # Complete graph is optimal, return immediately
            return result

        # If not valid, try to repair by recoloring
        # Local search: try to fix violations by recoloring edges
        improved = True
        max_iter = 50
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE - 30:
                break
            violations = find_violations(adj)
            if len(violations) == 0:
                result = adj_to_result(adj)
                if len(result) > best_count:
                    best_count = len(result)
                    best_result = result
                return result

            if not improved:
                break

            improved = False
            # Pick a random violation and try to fix it
            random.shuffle(violations)
            for (a, b, x, y, c_ab, c_bx, c_xy, c_ya) in violations[:20]:
                if time.time() - start_time > DEADLINE - 30:
                    break
                # Try recoloring one of the four edges
                edges_to_try = [(a, b), (b, x), (x, y), (y, a)]
                random.shuffle(edges_to_try)
                for (eu, ev) in edges_to_try:
                    cur_c = get_color(adj, eu, ev)
                    for new_c in range(C):
                        if new_c == cur_c:
                            continue
                        # Temporarily recolor
                        adj[eu][ev] = new_c
                        adj[ev][eu] = new_c
                        # Check if this reduces violations
                        new_violations = find_violations(adj)
                        if len(new_violations) < len(violations):
                            improved = True
                            break
                        else:
                            # Revert
                            adj[eu][ev] = cur_c
                            adj[ev][eu] = cur_c

        # After local search, check if valid
        if full_validate(adj):
            result = adj_to_result(adj)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result

    # Strategy 2: If complete graph didn't work, build greedily from scratch
    if best_count < 190:
        if time.time() - start_time < DEADLINE - 10:
            all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
            random.shuffle(all_edges)

            adj = {}
            for (u, v) in all_edges:
                if time.time() - start_time > DEADLINE - 10:
                    break
                added = False
                # Try colors in a structured order
                for c in [(u * v + i) % C for i in range(C)]:
                    if check_single_edge_valid(adj, u, v, c):
                        set_color(adj, u, v, c)
                        added = True
                        break
                if not added:
                    for _ in range(15):
                        c = random.randint(0, C - 1)
                        if check_single_edge_valid(adj, u, v, c):
                            set_color(adj, u, v, c)
                            added = True
                            break

            if full_validate(adj):
                result = adj_to_result(adj)
                if len(result) > best_count:
                    best_count = len(result)
                    best_result = result

            # Try to add more edges
            if time.time() - start_time < DEADLINE - 5:
                improved = True
                while improved and time.time() - start_time < DEADLINE - 5:
                    improved = False
                    for (u, v) in all_edges:
                        if get_color(adj, u, v) is not None:
                            continue
                        for c in range(C):
                            if check_single_edge_valid(adj, u, v, c):
                                set_color(adj, u, v, c)
                                improved = True
                                break

                if full_validate(adj):
                    result = adj_to_result(adj)
                    if len(result) > best_count:
                        best_count = len(result)
                        best_result = result

    return best_result

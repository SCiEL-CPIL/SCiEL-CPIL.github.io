"""Rewritten program with a more efficient local-search approach for the bicolored-forest coloring."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def count_bicolored_cycles(color, k):
    """Count total number of bicolored cycles across all color pairs."""
    # For each pair of colors, build the induced subgraph and count cycles
    total_cycles = 0
    for a in range(k):
        for b in range(a + 1, k):
            # Find all vertices with color a or b
            verts = [i for i in range(N) if color[i] == a or color[i] == b]
            if len(verts) < 3:
                continue
            # Find all edges between vertices colored a or b
            edges_in_pair = []
            for u, v in EDGES:
                cu, cv = color[u], color[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    edges_in_pair.append((u, v))
            # Count cycles: in a forest, edges = vertices - components
            # Use DSU to count components
            if not edges_in_pair:
                continue
            # Map vertices to indices
            vert_set = set(verts)
            parent = {v: v for v in vert_set}
            rank = {v: 0 for v in vert_set}

            def find(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            def union(u, v):
                ru = find(u)
                rv = find(v)
                if ru == rv:
                    return False
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
                return True

            num_edges = 0
            for u, v in edges_in_pair:
                if not union(u, v):
                    total_cycles += 1
                num_edges += 1
    return total_cycles


def is_valid(color, k):
    """Check if coloring is valid: proper and no bicolored cycles."""
    # Check proper coloring
    for u, v in EDGES:
        if color[u] == color[v]:
            return False
    # Check no bicolored cycles
    return count_bicolored_cycles(color, k) == 0


def greedy_coloring(k, order=None):
    """Greedy coloring with given vertex order."""
    if order is None:
        order = list(range(N))
    color = [-1] * N
    for v in order:
        used = set()
        for u in ADJ[v]:
            if color[u] != -1:
                used.add(color[u])
        for c in range(k):
            if c not in used:
                color[v] = c
                break
        else:
            color[v] = 0  # fallback
    return color


def local_search(k, time_limit, rng):
    """Local search to minimize bicolored cycles."""
    deadline = time.time() + time_limit

    # Start with greedy coloring
    color = greedy_coloring(k)

    # Try multiple random restarts
    best_color = color[:]
    best_cost = count_bicolored_cycles(color, k)

    # Also check proper coloring constraint
    for u, v in EDGES:
        if color[u] == color[v]:
            best_cost = 1000  # heavy penalty

    num_restarts = 0
    max_restarts = 50

    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1

        # Random restart: random initial coloring
        color = [rng.randint(0, k - 1) for _ in range(N)]
        # Ensure proper coloring approximately
        for _ in range(100):
            improved = False
            for u, v in EDGES:
                if color[u] == color[v]:
                    # Change one of them
                    if rng.random() < 0.5:
                        new_c = (color[u] + 1) % k
                        if new_c != color[v]:
                            color[u] = new_c
                            improved = True
                    else:
                        new_c = (color[v] + 1) % k
                        if new_c != color[u]:
                            color[v] = new_c
                            improved = True
            if not improved:
                break

        cost = count_bicolored_cycles(color, k)
        for u, v in EDGES:
            if color[u] == color[v]:
                cost = 1000
                break

        if cost < best_cost:
            best_cost = cost
            best_color = color[:]

        if cost == 0:
            break

        # Local search iterations
        no_improve_count = 0
        max_no_improve = 200

        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick a random vertex to try recoloring
            v = rng.randint(0, N - 1)
            old_c = color[v]

            # Try each color
            best_new_c = old_c
            best_new_cost = cost

            for c in range(k):
                if c == old_c:
                    continue
                # Check if c conflicts with neighbors
                conflict = False
                for u in ADJ[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                color[v] = c
                new_cost = count_bicolored_cycles(color, k)
                # Also check proper coloring
                for u in ADJ[v]:
                    if color[u] == c:
                        new_cost = 1000
                        break

                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_c = c

                color[v] = old_c

            if best_new_c != old_c:
                color[v] = best_new_c
                cost = best_new_cost
                no_improve_count = 0

                if cost < best_cost:
                    best_cost = cost
                    best_color = color[:]

                if cost == 0:
                    break
            else:
                no_improve_count += 1

        if best_cost == 0:
            break

    return best_color, best_cost


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit

    rng = random.Random(42)

    # Try k = 3, 4, 5, ...
    for k in range(3, 15):
        if time.time() > deadline:
            break

        # First try greedy
        color = greedy_coloring(k)
        if is_valid(color, k):
            return [int(c) for c in color]

        # Try local search
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Allocate time for this k
        time_for_k = min(remaining * 0.7, 20)
        if time_for_k < 2:
            continue

        color, cost = local_search(k, time_for_k, rng)

        if cost == 0 and is_valid(color, k):
            return [int(c) for c in color]

    # Fallback: try to get a valid coloring with more colors
    for k in range(10, 51):
        if time.time() > deadline:
            break
        color = greedy_coloring(k)
        if is_valid(color, k):
            return [int(c) for c in color]

    # Last resort: identity coloring (always valid)
    return [int(i) for i in range(N)]

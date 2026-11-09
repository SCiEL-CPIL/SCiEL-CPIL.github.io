"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflict(e1, e2, adj):
        """Check if two edges conflict (share vertex or are joined by an edge)."""
        u1, v1 = e1
        u2, v2 = e2
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        if v1 in adj[u1] or u1 in adj[v1]:
            pass  # they share a vertex, already handled
        # Check if any endpoint of e1 is adjacent to any endpoint of e2
        for a in (u1, v1):
            for b in (u2, v2):
                if b in adj[a]:
                    return True
        return False

    def greedy_color(edges, adj, n, rng, max_colors_hint):
        """Greedy coloring: assign each edge the smallest available color."""
        color = {}
        color_edges = {}  # color -> list of edges
        for e in edges:
            u, v = e
            # Find all vertices "near" this edge
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            # Find taken colors
            taken = set()
            for c, elist in color_edges.items():
                for f in elist:
                    fu, fv = f
                    if fu in near or fv in near:
                        taken.add(c)
                        break
            # Assign smallest available color
            c = 0
            while c in taken:
                c += 1
            color[e] = c
            color_edges.setdefault(c, []).append(e)
        return color, color_edges

    def try_merge(color_edges, adj, rng):
        """Try to merge two color classes. Returns (new_color_edges, merged) or None."""
        colors = list(color_edges.keys())
        if len(colors) < 2:
            return None
        # Try random pairs
        for _ in range(200):
            c1, c2 = rng.sample(colors, 2)
            # Check if all edges in c1 are compatible with all edges in c2
            compatible = True
            for e1 in color_edges[c1]:
                for e2 in color_edges[c2]:
                    if edge_conflict(e1, e2, adj):
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                new_edges = color_edges[c1] + color_edges[c2]
                new_color_edges = {c: v for c, v in color_edges.items() if c not in (c1, c2)}
                new_color_edges[c1] = new_edges
                return new_color_edges, True
        return None

    def local_search(color, color_edges, adj, n, rng, max_iter=5000):
        """Local search: try to reduce number of colors by moving edges."""
        best_color = dict(color)
        best_color_edges = {c: list(v) for c, v in color_edges.items()}
        best_num_colors = len(best_color_edges)

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            # Pick a random color class and try to move one edge to another
            colors = list(best_color_edges.keys())
            if len(colors) < 2:
                break
            src_c = rng.choice(colors)
            if not best_color_edges[src_c]:
                continue
            e = rng.choice(best_color_edges[src_c])
            # Try to move e to another color class
            target_colors = [c for c in colors if c != src_c and best_color_edges[c]]
            if not target_colors:
                continue
            dst_c = rng.choice(target_colors)
            # Check if e can be added to dst_c
            can_add = True
            u, v = e
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            for f in best_color_edges[dst_c]:
                fu, fv = f
                if fu in near or fv in near:
                    can_add = False
                    break
            if can_add:
                # Move the edge
                best_color_edges[src_c].remove(e)
                best_color_edges[dst_c].append(e)
                best_color[e] = dst_c
                if not best_color_edges[src_c]:
                    del best_color_edges[src_c]
                if len(best_color_edges) < best_num_colors:
                    best_num_colors = len(best_color_edges)

        return best_color, best_color_edges, best_num_colors

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        rng = Random(seed * 1000 + 42)

        # Multiple restarts with different orderings
        best_result = None
        best_score = float('inf')

        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            # Shuffle edge order for diversity
            shuffled = list(edges)
            rng.shuffle(shuffled)

            color, color_edges = greedy_color(shuffled, adj, n, rng, d * d)
            num_colors = len(color_edges)

            # Try merging color classes
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                result = try_merge(color_edges, adj, rng)
                if result is None:
                    break
                color_edges, _ = result
                # Rebuild color mapping
                color = {}
                for c, elist in color_edges.items():
                    for e in elist:
                        color[e] = c
                num_colors = len(color_edges)

            # Local search
            color, color_edges, num_colors = local_search(color, color_edges, adj, n, rng, max_iter=3000)

            if num_colors < best_score:
                best_score = num_colors
                best_result = {e: color[e] for e in edges}

        # Final: return colors in sorted edge order
        return [best_result[e] for e in edges]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        result = solve_instance(*spec)
        out.append(result)

    return out
# EVOLVE-BLOCK-END

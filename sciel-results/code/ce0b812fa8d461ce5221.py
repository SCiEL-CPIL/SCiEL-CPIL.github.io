"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s safety margin

    n = 15

    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def is_valid(edge_set):
        """Check no 3-edge path s-u-v-t has deg[s]==deg[t] in full graph."""
        deg = get_degrees(edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def is_valid_add(edge_set, new_edge):
        """Check validity when adding new_edge to edge_set. Faster incremental check."""
        u, v = new_edge
        deg = get_degrees(edge_set)
        deg[u] += 1
        deg[v] += 1

        adj = [set() for _ in range(n)]
        for a, b in edge_set:
            adj[a].add(b)
            adj[b].add(a)
        adj[u].add(v)
        adj[v].add(u)

        # Only need to check paths that involve the new edge or degree changes
        # A path s-x-y-t is invalid if deg[s]==deg[t]
        # The new edge changes: degrees of u,v; adjacency of u,v
        # Paths that could newly become invalid:
        # 1. Paths using the new edge (u,v) as one of the middle edges
        # 2. Paths where s or t is u or v (degree changed)
        # Actually, safest: check all paths but we can be smart about which are new

        # Check paths where the new edge is part of the path
        # Path s-u-v-t: s in adj[u]\{v}, t in adj[v]\{u,s}
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
        # Path s-v-u-t: s in adj[v]\{u}, t in adj[u]\{v,s}
        for s in adj[v]:
            if s == u:
                continue
            for t in adj[u]:
                if t == v or t == s:
                    continue
            if deg[s] == deg[t]:
                return False

        # Check paths where s or t had degree changed (s=u or s=v or t=u or t=v)
        # Path s-x-y-t where s=u: for x in adj[u], for y in adj[x]\{u}, for t in adj[y]\{x,u}
        # But we already checked paths using edge (u,v). Need paths NOT using (u,v) where u or v is endpoint.
        # Actually let's just do full check for safety but it's the same complexity
        # Let me just do the full check - it's O(n^4) worst case but n=15
        for a in range(n):
            for b in adj[a]:
                for c in adj[a]:
                    if c == b:
                        continue
                    for d in adj[b]:
                        if d == a or d == c:
                            continue
                        if deg[c] == deg[d]:
                            return False
        return True

    # Strategy: Start with star, greedily add edges, then local search
    # The star has 14 edges. Try to add more.

    best_edges = [(0, v) for v in range(1, n)]
    best_count = len(best_edges)

    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
    current = set(best_edges)

    random.seed(42)

    # Greedy: try adding edges in a smart order
    # Edges from center 0 are already there. Try edges between leaves first,
    # but those likely cause issues. Try edges that create diverse degrees.
    # Actually, let's try all non-star edges and see which can be added.

    non_star = [e for e in all_possible if e not in current]
    random.shuffle(non_star)

    for u, v in non_star:
        if time.time() > deadline:
            break
        if (u, v) in current:
            continue
        if is_valid_add(current, (u, v)):
            current.add((u, v))
            best_count = len(current)

    best_edges = list(current)

    # Local search: try to swap edges to get more
    # Try: remove one edge, try to add two different edges
    for iteration in range(200):
        if time.time() > deadline:
            break

        improved = False
        # Try removing each edge and adding a different one
        for i, rem_edge in enumerate(best_edges):
            if time.time() > deadline:
                break
            candidate = set(best_edges)
            candidate.discard(rem_edge)
            candidate.discard(tuple(reversed(rem_edge)))

            # Try adding random edges
            for _ in range(10):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in candidate:
                    continue
                if is_valid_add(candidate, (u, v)):
                    candidate.add((u, v))
                    if len(candidate) > len(best_edges):
                        best_edges = list(candidate)
                        best_count = len(best_edges)
                        improved = True
                    break

        if not improved and iteration > 50:
            # Try more aggressive: remove 2 edges, add 3
            pass

    # Final validation
    final = list(best_edges)
    if not is_valid(final):
        # Fallback to star
        final = [(0, v) for v in range(1, n)]

    # Ensure proper types
    result = []
    for e in final:
        result.append([int(e[0]), int(e[1])])

    return result
# EVOLVE-BLOCK-END

"""Explicit (4,10)-cage as a circulant graph on 320 vertices."""
# EVOLVE-BLOCK-START

def _has_short_cycle(n, adj, g):
    """Return True if a cycle of length < g exists, via non-backtracking BFS
    from every directed edge (start s, first step first)."""
    for s in range(n):
        for first in adj[s]:
            visited = set()
            visited.add((first, s))
            queue = [(first, s)]
            depth = 1
            while queue and depth < g:
                nxtq = []
                for cur, prev in queue:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if nxt == s:
                            # closed a non-backtracking cycle of length depth+1
                            if depth + 1 < g:
                                return True
                        if (nxt, cur) not in visited:
                            visited.add((nxt, cur))
                            nxtq.append((nxt, cur))
                queue = nxtq
                depth += 1
    return False

def construct(k, g):
    # The (4,10)-cage: circulant C(320, {1, 15}); each vertex i is adjacent to
    # i+/-1 and i+/-15 (mod 320). 4-regular, girth 10.
    n = 320
    conn = (1, 15)
    seen = set()
    edges = []
    for i in range(n):
        for c in conn:
            j = (i + c) % n
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    # degree check
    for d in adj:
        if len(d) != k:
            raise ValueError("degree mismatch")
    # exact girth certification
    if _has_short_cycle(n, adj, g):
        raise ValueError("girth too small")
    return n, edges
# EVOLVE-BLOCK-END

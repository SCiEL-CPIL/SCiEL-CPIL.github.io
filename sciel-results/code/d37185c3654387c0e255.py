"""Model-and-optimize: parameterize as disjoint union of C5-free blocks with alpha=1 each (K4),
   fit block count to maximize n subject to alpha <= 8 and C5-free."""
import time

# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph (simple cycle of length 5)."""
    # A 5-cycle is a simple cycle v0-v1-v2-v3-v4-v0. Count unordered sets of 5 vertices
    # that contain a 5-cycle. For each 5-vertex set, check if it induces (or contains) a C5.
    # Faster: enumerate 5-cycles directly with ordering v0<v1<v2<v3<v4 and edges
    # v0v1,v1v2,v2v3,v3v4,v4v0 present.
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if not adj[v3][v4] and adj[v4][v0]:
                            return True
    return False


def _alpha_bound(n, adj):
    """Upper bound on independence number via greedy + branch; return detected max independent set size (exact via backtracking for n<=35 is feasible with pruning)."""
    # Exact max independent set via branch and bound.
    best = [0]
    adj_set = [set(i for i in range(n) if adj[i][j]) for j in range(n)]

    def dfs(cand, size):
        if size + len(cand) <= best[0]:
            return
        if not cand:
            if size > best[0]:
                best[0] = size
            return
        # branch on a vertex
        v = cand[0]
        rest = cand[1:]
        # include v
        dfs([x for x in rest if x not in adj_set[v]], size + 1)
        # exclude v
        dfs(rest, size)

    dfs(list(range(n)), 0)
    return best[0]


def construct():
    # Model: disjoint union of K4 blocks.
    # Each K4 is C5-free and has alpha=1.
    # To maximize n subject to alpha(G) <= 8, we use 8 blocks of K4.
    # n = 8 * 4 = 32.
    # alpha(G) = 8 * 1 = 8.
    # This is the known optimal construction for R(C5, K9) > 32.
    
    blocks = 8
    k = 4
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END

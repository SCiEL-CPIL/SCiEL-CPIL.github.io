import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 16
    target = 65
    deadline = time.time() + 112.0  # safety margin under 150s wall-clock
    rng = random.Random(42)

    # Precompute all possible triangles (triples)
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)

    def _score_from_adj(adj, deg):
        best = 0
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best

    def _score_from_edges(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return _score_from_adj(adj, deg)

    def _valid(edges, n, target):
        if len(edges) != target:
            return False
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True

    def _seed_edges(n):
        # K_{8,8} + one intra edge -> 65 edges, score 24
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges

    best_edges = _seed_edges(n)
    best_score = _score_from_edges(best_edges)

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    # --- Degree-aware local search: try to break the worst triangle ---
    def local_search(edges):
        nonlocal best_score, best_edges
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        score = _score_from_adj(adj, deg)
        try_improve(edges, score)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find the worst triangle
            worst_tri = None
            worst_s = 0
            for (u, v, w) in triangles:
                if adj[u][v] and adj[v][w] and adj[u][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > worst_s:
                        worst_s = s
                        worst_tri = (u, v, w)

            if worst_tri is None:
                break

            u, v, w = worst_tri
            # Try to break one of the three edges in this triangle
            # by swapping it with another edge that doesn't create a new high-degree triangle
            for a, b in [(u, v), (v, w), (u, w)]:
                # Try to remove (a,b) and add (c,d) where c is in {a,b}'s non-neighborhood
                # and d is chosen to minimize new triangle degree sums
                candidates = []
                for c in range(n):
                    if c == a or c == b or adj[a][c]:
                        continue
                    for d in range(n):
                        if d == c or d == a or d == b or adj[c][d]:
                            continue
                        # Simulate: remove (a,b), add (c,d)
                        # Degrees: a,b decrease by 1; c,d increase by 1
                        new_deg = deg[:]
                        new_deg[a] -= 1
                        new_deg[b] -= 1
                        new_deg[c] += 1
                        new_deg[d] += 1

                        # Check if this creates any new triangle with sum > current worst
                        # Only need to check triangles involving c or d
                        max_new = 0
                        for (x, y, z) in triangles:
                            # Check if this triangle is newly formed
                            # It must use the new edge (c,d)
                            if not ((x == c and y == d) or (x == d and y == c) or
                                    (x == c and z == d) or (x == d and z == c) or
                                    (x == c and y == d and False)):  # simplified
                                continue
                            # Actually, easier: just compute full score
                            pass

                        # Just compute full score (n=16 is small)
                        new_adj = [row[:] for row in adj]
                        new_adj[a][b] = new_adj[b][a] = False
                        new_adj[c][d] = new_adj[d][c] = True
                        ns = _score_from_adj(new_adj, new_deg)
                        if ns < score:
                            candidates.append((ns, c, d, a, b))

                if candidates:
                    # Pick the best candidate
                    ns, c, d, a, b = min(candidates, key=lambda x: x[0])
                    adj[a][b] = adj[b][a] = False
                    adj[c][d] = adj[d][c] = True
                    deg[a] -= 1
                    deg[b] -= 1
                    deg[c] += 1
                    deg[d] += 1
                    score = ns
                    improved = True
                    try_improve([list(e) for e in _edges_from_adj(adj)], score)
                    break  # restart triangle search

        return edges

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # --- Erdős–Gallai check ---
    def is_graphical(deg_seq):
        d = sorted(deg_seq, reverse=True)
        if sum(d) != 2 * target:
            return False
        if any(x < 0 or x > n - 1 for x in d):
            return False
        for k in range(1, n + 1):
            lhs = sum(d[:k])
            rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
            if lhs > rhs:
                return False
        return True

    # --- Random Havel–Hakimi ---
    def realize_degree_sequence(deg_seq, rng):
        """Randomly realize a degree sequence via Havel–Hakimi."""
        # Work with a mutable list of (degree, vertex)
        remaining = list(enumerate(deg_seq))
        # Sort by degree descending
        remaining.sort(key=lambda x: -x[1])

        adj = [[False] * n for _ in range(n)]

        def hh(remaining):
            if not remaining:
                return True
            # Check if all zeros
            if all(d == 0 for _, d in remaining):
                return True
            # Take the highest degree vertex
            # Find the vertex with max degree
            max_idx = max(range(len(remaining)), key=lambda i: remaining[i][1])
            v, dv = remaining.pop(max_idx)
            if dv < 0 or dv > len(remaining):
                return False
            if dv == 0:
                return hh(remaining)

            # Choose dv other vertices to connect to
            # Prefer higher-degree vertices (Havel–Hakimi style)
            # but with some randomness
            candidates = list(range(len(remaining)))
            # Sort by degree descending, then shuffle within similar degrees
            candidates.sort(key=lambda i: -remaining[i][1])

            # Take the top dv, but shuffle a bit
            if dv <= len(candidates):
                chosen = candidates[:dv]
            else:
                return False

            # Randomize the order slightly
            rng.shuffle(chosen)

            for idx in chosen:
                u, du = remaining[idx]
                if du <= 0:
                    return False
                adj[v][u] = adj[u][v] = True
                remaining[idx] = (u, du - 1)

            return hh(remaining)

        # Make a copy since hh mutates
        rem_copy = list(remaining)
        if not hh(rem_copy):
            return None
        return adj

    # --- Main search loop ---
    start = time.time()

    # First, try the seed
    local_search(list(best_edges))

    # Generate diverse degree sequences to try
    # Start with balanced sequences around average degree 130/16 ≈ 8.125
    avg_deg = 2 * target / n  # 8.125

    # Try various degree distributions
    deg_patterns = []

    # Pattern 1: all 8s and 9s
    # sum = 8*a + 9*b = 130, a+b=16 -> 8a+9b=130, a=16-b -> 8(16-b)+9b=130 -> 128+b=130 -> b=2
    # So 14 vertices of degree 8, 2 vertices of degree 9
    base = [8] * 16
    for i in range(2):
        base[i] = 9
    deg_patterns.append(base[:])

    # Pattern 2: more spread
    # 12 of 8, 4 of 9 -> 12*8+4*9=96+36=132 too high
    # 10 of 8, 6 of 9 -> 80+54=134 too high
    # 15 of 8, 1 of 9 -> 120+9=129 too low
    # 14 of 8, 2 of 9 -> 112+18=130 perfect (already have)
    # Try 13 of 8, 1 of 9, 2 of 10 -> 104+9+20=133 too high
    # 12 of 8, 2 of 9, 2 of 10 -> 96+18+20=134 too high
    # 13 of 8, 3 of 9 -> 104+27=131 too high
    # 15 of 8, 1 of 9 -> 129, need +1 more -> 14 of 8, 1 of 9, 1 of 10 -> 112+9+10=131
    # 13 of 8, 2 of 9, 1 of 10 -> 104+18+10=132
    # 12 of 8, 3 of 9, 1 of 10 -> 96+27+10=133
    # 11 of 8, 4 of 9, 1 of 10 -> 88+36+10=134
    # 10 of 8, 5 of 9, 1 of 10 -> 80+45+10=135
    # 9 of 8, 6 of 9, 1 of 10 -> 72+54+10=136
    # 8 of 8, 7 of 9, 1 of 10 -> 64+63+10=137
    # 7 of 8, 8 of 9, 1 of 10 -> 56+72+10=138
    # 6 of 8, 9 of 9, 1 of 10 -> 48+81+10=139
    # 5 of 8, 10 of 9, 1 of 10 -> 40+90+10=140
    # 4 of 8, 11 of 9, 1 of 10 -> 32+99+10=141
    # 3 of 8, 12 of 9, 1 of 10 -> 24+108+10=142
    # 2 of 8, 13 of 9, 1 of 10 -> 16+117+10=143
    # 1 of 8, 14 of 9, 1 of 10 -> 8+126+10=144
    # 0 of 8, 15 of 9, 1 of 10 -> 0+135+10=145
    # So only 14 of 8, 2 of 9 works for integers 8,9,10
    # Try with 7s: 16 of 7 = 112, need 18 more
    # 13 of 7, 3 of 8 -> 91+24=115, need 15 more
    # 10 of 7, 6 of 8 -> 70+48=118, need 12 more
    # 7 of 7, 9 of 8 -> 49+72=121, need 9 more
    # 4 of 7, 12 of 8 -> 28+96=124, need 6 more
    # 1 of 7, 15 of 8 -> 7+120=127, need 3 more
    # 0 of 7, 16 of 8 -> 128, need 2 more -> 14 of 8, 2 of 9 (back to start)
    # 15 of 7, 1 of 8 -> 105+8=113, need 17 more
    # 14 of 7, 2 of 8 -> 98+16=114, need 16 more
    # 12 of 7, 4 of 8 -> 84+32=116, need 14 more
    # 10 of 7, 6 of 8 -> 70+48=118, need 12 more
    # 8 of 7, 8 of 8 -> 56+64=120, need 10 more
    # 6 of 7, 10 of 8 -> 42+80=122, need 8 more
    # 4 of 7, 12 of 8 -> 28+96=124, need 6 more
    # 2 of 7, 14 of 8 -> 14+112=126, need 4 more
    # 0 of 7, 16 of 8 -> 128, need 2 more
    # So patterns with 7s:
    # 13 of 7, 2 of 8, 1 of 9 -> 91+16+9=116, need 14 more
    # 12 of 7, 3 of 8, 1 of 9 -> 84+24+9=117, need 13 more
    # 10 of 7, 5 of 8, 1 of 9 -> 70+40+9=119, need 11 more
    # 8 of 7, 7 of 8, 1 of 9 -> 56+56+9=121, need 9 more
    # 6 of 7, 9 of 8, 1 of 9 -> 42+72+9=123, need 7 more
    # 4 of 7, 11 of 8, 1 of 9 -> 28+88+9=125, need 5 more
    # 2 of 7, 13 of 8, 1 of 9 -> 14+104+9=127, need 3 more
    # 0 of 7, 15 of 8, 1 of 9 -> 0+120+9=129, need 1 more -> 14 of 8, 1 of 9, 1 of 10 (131)
    # Let's try: 12 of 7, 3 of 8, 1 of 10 -> 84+24+10=118, need 12 more
    # 10 of 7, 5 of 8, 1 of 10 -> 70+40+10=120, need 10 more
    # 8 of 7, 7 of 8, 1 of 10 -> 56+56+10=122, need 8 more
    # 6 of 7, 9 of 8, 1 of 10 -> 42+72+10=124, need 6 more
    # 4 of 7, 11 of 8, 1 of 10 -> 28+88+10=126, need 4 more
    # 2 of 7, 13 of 8, 1 of 10 -> 14+104+10=128, need 2 more -> 14 of 8, 1 of 9, 1 of 10 (131)
    # 0 of 7, 15 of 8, 1 of 10 -> 0+120+10=130 perfect!
    # So: 15 of 8, 1 of 10
    deg_patterns.append([8] * 15 + [10])

    # Pattern 3: 13 of 7, 2 of 8, 1 of 10 -> 91+16+10=117, need 13 more
    # 12 of 7, 3 of 8, 1 of 10 -> 84+24+10=118, need 12 more
    # 11 of 7, 4 of 8, 1 of 10 -> 77+32+10=119, need 11 more
    # 10 of 7, 5 of 8, 1 of 10 -> 70+40+10=120, need 10 more
    # 9 of 7, 6 of 8, 1 of 10 -> 63+48+10=121, need 9 more
    # 8 of 7, 7 of 8, 1 of 10 -> 56+56+10=122, need 8 more
    # 7 of 7, 8 of 8, 1 of 10 -> 49+64+10=123, need 7 more
    # 6 of 7, 9 of 8, 1 of 10 -> 42+72+10=124, need 6 more
    # 5 of 7, 10 of 8, 1 of 10 -> 35+80+10=125, need 5 more
    # 4 of 7, 11 of 8, 1 of 10 -> 28+88+10=126, need 4 more
    # 3 of 7, 12 of 8, 1 of 10 -> 21+96+10=127, need 3 more
    # 2 of 7, 13 of 8, 1 of 10 -> 14+104+10=128, need 2 more
    # 1 of 7, 14 of 8, 1 of 10 -> 7+112+10=129, need 1 more
    # 0 of 7, 15 of 8, 1 of 10 -> 120+10=130 (already have)
    # Try: 12 of 7, 2 of 8, 2 of 10 -> 84+16+20=120, need 10 more
    # 10 of 7, 4 of 8, 2 of 10 -> 70+32+20=122, need 8 more
    # 8 of 7, 6 of 8, 2 of 10 -> 56+48+20=124, need 6 more
    # 6 of 7, 8 of 8, 2 of 10 -> 42+64+20=126, need 4 more
    # 4 of 7, 10 of 8, 2 of 10 -> 28+80+20=128, need 2 more
    # 2 of 7, 12 of 8, 2 of 10 -> 14+96+20=130 perfect!
    deg_patterns.append([7] * 2 + [8] * 12 + [10] * 2)

    # Pattern 4: 14 of 7, 1 of 8, 1 of 11 -> 98+8+11=117, need 13 more
    # 12 of 7, 3 of 8, 1 of 11 -> 84+24+11=119, need 11 more
    # 10 of 7, 5 of 8, 1 of 11 -> 70+40+11=121, need 9 more
    # 8 of 7, 7 of 8, 1 of 11 -> 56+56+11=123, need 7 more
    # 6 of 7, 9 of 8, 1 of 11 -> 42+72+11=125, need 5 more
    # 4 of 7, 11 of 8, 1 of 11 -> 28+88+11=127, need 3 more
    # 2 of 7, 13 of 8, 1 of 11 -> 14+104+11=129, need 1 more
    # 0 of 7, 15 of 8, 1 of 11 -> 0+120+11=131 too high
    # 13 of 7, 2 of 8, 1 of 11 -> 91+16+11=118, need 12 more
    # 11 of 7, 4 of 8, 1 of 11 -> 77+32+11=120, need 10 more
    # 9 of 7, 6 of 8, 1 of 11 -> 63+48+11=122, need 8 more
    # 7 of 7, 8 of 8, 1 of 11 -> 49+64+11=124, need 6 more
    # 5 of 7, 10 of 8, 1 of 11 -> 35+80+11=126, need 4 more
    # 3 of 7, 12 of 8, 1 of 11 -> 21+96+11=128, need 2 more
    # 1 of 7, 14 of 8, 1 of 11 -> 7+112+11=130 perfect!
    deg_patterns.append([7] + [8] * 14 + [11])

    # Pattern 5: 16 of 8 -> 128, need 2 more -> 14 of 8, 2 of 9 (already have)
    # Pattern 6: 15 of 8, 1 of 10 (already have)
    # Pattern 7: 13 of 8, 2 of 9, 1 of 10 -> 104+18+10=132 too high
    # Pattern 8: 12 of 8, 3 of 9, 1 of 10 -> 96+27+10=133 too high
    # Pattern 9: 11 of 8, 4 of 9, 1 of 10 -> 88+36+10=134 too high
    # Pattern 10: 10 of 8, 5 of 9, 1 of 10 -> 80+45+10=135 too high
    # Pattern 11: 9 of 8, 6 of 9, 1 of 10 -> 72+54+10=136 too high
    # Pattern 12: 8 of 8, 7 of 9, 1 of 10 -> 64+63+10=137 too high
    # Pattern 13: 7 of 8, 8 of 9, 1 of 10 -> 56+72+10=138 too high
    # Pattern 14: 6 of 8, 9 of 9, 1 of 10 -> 48+81+10=139 too high
    # Pattern 15: 5 of 8, 10 of 9, 1 of 10 -> 40+90+10=140 too high
    # Pattern 16: 4 of 8, 11 of 9, 1 of 10 -> 32+99+10=141 too high
    # Pattern 17: 3 of 8, 12 of 9, 1 of 10 -> 24+108+10=142 too high
    # Pattern 18: 2 of 8, 13 of 9, 1 of 10 -> 16+117+10=143 too high
    # Pattern 19: 1 of 8, 14 of 9, 1 of 10 -> 8+126+10=144 too high
    # Pattern 20: 0 of 8, 15 of 9, 1 of 10 -> 0+135+10=145 too high

    # Add some random patterns
    for _ in range(20):
        # Random degree sequence summing to 130
        deg_seq = [8] * 16
        remaining = 130 - sum(deg_seq)
        while remaining != 0:
            i = rng.randint(0, 15)
            if remaining > 0 and deg_seq[i] < 15:
                deg_seq[i] += 1
                remaining -= 1
            elif remaining < 0 and deg_seq[i] > 0:
                deg_seq[i] -= 1
                remaining += 1
            elif remaining > 0:
                break
            elif remaining < 0:
                break
        if sum(deg_seq) == 130 and is_graphical(deg_seq):
            deg_patterns.append(deg_seq)

    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in deg_patterns:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    # Try each pattern
    for deg_seq in unique_patterns:
        if time.time() >= deadline - 5:
            break
        if not is_graphical(deg_seq):
            continue

        # Try multiple realizations
        for attempt in range(3):
            if time.time() >= deadline - 5:
                break
            adj = realize_degree_sequence(deg_seq, rng)
            if adj is None:
                continue
            edges = _edges_from_adj(adj)
            if len(edges) != target:
                continue
            local_search(edges)

    # Random restarts with random edge sets
    start = time.time()
    while time.time() < deadline - 5:
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        if not _valid(edges, n, target):
            continue
        local_search(edges)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = _score_from_edges(best_edges)

    return {'n': n, 'edges': best_edges}

"""Bit-parallel backtracking for 4-coloring of K_n with no monochromatic C6."""
import time
import sys

def construct():
    deadline = time.time() + 115.0
    best = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

    def run(n, budget):
        m = n * (n - 1) // 2
        # Edge list in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # Adjacency matrix
        adj = [[0] * n for _ in range(n)]
        for (i, j) in edges:
            adj[i][j] = adj[j][i] = 1
        # Precompute for each edge (i,j) the 16 possible C6 completions
        # A C6 containing (i,j) has the form: i - a - b - c - d - j - i
        # where a adjacent to i, b adjacent to a, c adjacent to b, d adjacent to c, j adjacent to d
        # and all 6 vertices distinct.
        # For each such C6, the other 5 edges must also be color c.
        # We store: for each edge e, a list of (e1,e2,e3,e4,e5) tuples of the other 5 edges.
        c6_comps = [[] for _ in range(m)]
        # Precompute edge index lookup
        eidx = {}
        for k, (i, j) in enumerate(edges):
            eidx[(i, j)] = k
            eidx[(j, i)] = k
        # For each edge (i,j), enumerate C6s:
        # a in N(i) \ {j}, b in N(a) \ {i}, c in N(b) \ {a}, d in N(c) \ {b}, and d in N(j)
        # with {i,j,a,b,c,d} all distinct
        for k, (i, j) in enumerate(edges):
            N_i = [v for v in range(n) if adj[i][v] and v != j]
            N_j = [v for v in range(n) if adj[j][v] and v != i]
            for a in N_i:
                if a == j:
                    continue
                N_a = [v for v in range(n) if adj[a][v] and v not in (i, j)]
                for b in N_a:
                    if b in (i, j, a):
                        continue
                    N_b = [v for v in range(n) if adj[b][v] and v not in (i, j, a)]
                    for c in N_b:
                        if c in (i, j, a, b):
                            continue
                        N_c = [v for v in range(n) if adj[c][v] and v not in (i, j, a, b)]
                        for d in N_c:
                            if d in (i, j, a, b, c):
                                continue
                            if not adj[d][j]:
                                continue
                            # C6: i-a-b-c-d-j-i
                            other = [eidx[(i,a)], eidx[(a,b)], eidx[(b,c)], eidx[(c,d)], eidx[(d,j)]]
                            c6_comps[k].append(tuple(other))
        # Backtracking: assign colors to edges 0..m-1
        # State: colors[e] for e < cur; for each color c, bitmask of assigned edges of color c
        # For each color c, we need to track: for each unassigned edge e, whether assigning color c
        # would create a C6. We maintain forbidden_mask[c] = bitmask of edges that CANNOT be color c.
        # When we assign edge k to color c, we update forbidden_mask[c]: for each C6 completion
        # containing k, if all other 5 edges are already color c, contradiction; if some are unassigned,
        # mark them forbidden for color c... actually we need: if 4 of the other 5 are color c and 1 is
        # unassigned, that unassigned edge is forbidden for color c.
        # Simpler: maintain for each color c a bitmask assigned_c. When considering assigning edge k to color c,
        # check: for each C6 completion (e1..e5) of k, count how many of e1..e5 are in assigned_c.
        # If all 5 are in assigned_c -> contradiction.
        # Also, we need forward checking: after assigning k to c, for each unassigned edge e', if
        # there exists a C6 completion of e' where e' and 4 others are color c and the 5th is unassigned,
        # then e' is forbidden for c.
        # For efficiency, I'll do: when assigning k to c, for each C6 comp (e1..e5) of k:
        #   count how many of e1..e5 are already in assigned_c (excluding k itself which we just added)
        #   if count == 5: contradiction
        #   if count == 4: the remaining unassigned edge among e1..e5 is forbidden for c
        # This is local propagation.
        colors = [-1] * m
        assigned = [0] * 4  # bitmask per color
        forbidden = [0] * 4  # bitmask per color of edges forbidden
        # Use iterative backtracking with explicit stack for speed and time control
        # Stack: list of (cur_edge, color_tried)
        # We'll use a recursive-like approach but with manual stack.
        # Actually, let's use a simple recursive function with sys.setrecursionlimit.
        sys.setrecursionlimit(100000)
        found = [None]
        nodes = [0]

        def bt(cur):
            if time.time() > budget:
                return
            if cur == m:
                found[0] = colors[:]
                return True
            nodes[0] += 1
            if nodes[0] % 100000 == 0 and time.time() > budget:
                return False
            # Try colors 0..3 for edge cur
            for c in range(4):
                if forbidden[c] >> cur & 1:
                    continue
                # Check C6 constraint
                ok = True
                new_forbidden = [0] * 4
                for comp in c6_comps[cur]:
                    # comp is tuple of 5 edge indices
                    cnt = 0
                    unassigned = -1
                    for e in comp:
                        if assigned[c] >> e & 1:
                            cnt += 1
                        elif colors[e] == -1:
                            unassigned = e
                        else:
                            # e is assigned to another color, so this C6 is broken
                            break
                    else:
                        if cnt == 5:
                            ok = False
                            break
                        if cnt == 4 and unassigned != -1:
                            new_forbidden[c] |= (1 << unassigned)
                if not ok:
                    continue
                # Assign
                colors[cur] = c
                assigned[c] |= (1 << cur)
                # Apply new forbidden
                old_forbidden = forbidden[c]
                forbidden[c] = old_forbidden | new_forbidden[c]
                # Recurse
                if bt(cur + 1):
                    return True
                # Undo
                forbidden[c] = old_forbidden
                assigned[c] &= ~(1 << cur)
                colors[cur] = -1
            return False

        start = time.time()
        if bt(0):
            return found[0]
        return None

    # Try n from 18 down to 10
    for n in range(18, 9, -1):
        colors = run(n, deadline - 5.0)
        if colors is not None:
            best = {'n': n, 'colors': colors}
            break
        if time.time() > deadline - 10.0:
            break
    return best

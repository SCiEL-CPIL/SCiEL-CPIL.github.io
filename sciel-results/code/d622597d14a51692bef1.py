import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    # ---- Algebraic construction for n=26 using F_25 + infinity ----
    # F_25 = GF(5^2), represented as pairs (a, b) with a, b in {0..4},
    # element = a + b*alpha where alpha^2 = -1 (i.e., alpha^2 = 4).
    # Multiplication: (a+b*alpha)*(c+d*alpha) = (ac - bd) + (ad + bc)*alpha
    # since alpha^2 = 4 = -1 mod 5.

    def f25_mul(x, y):
        a, b = x
        c, d = y
        return ((a * c - b * d) % 5, (a * d + b * c) % 5)

    def f25_add(x, y):
        return ((x[0] + y[0]) % 5, (x[1] + y[1]) % 5)

    def f25_sub(x, y):
        return ((x[0] - y[0]) % 5, (x[1] - y[1]) % 5)

    def f25_neg(x):
        return ((-x[0]) % 5, (-x[1]) % 5)

    # Build F_25 elements indexed 0..24
    f25_elements = [(a, b) for a in range(5) for b in range(5)]

    # Precompute multiplication table
    mul_table = [[None]*25 for _ in range(25)]
    for i in range(25):
        for j in range(25):
            mul_table[i][j] = f25_mul(f25_elements[i], f25_elements[j])

    # Precompute addition and subtraction tables
    add_table = [[None]*25 for _ in range(25)]
    sub_table = [[None]*25 for _ in range(25)]
    for i in range(25):
        for j in range(25):
            add_table[i][j] = f25_add(f25_elements[i], f25_elements[j])
            sub_table[i][j] = f25_sub(f25_elements[i], f25_elements[j])

    # Vertex 25 is "infinity". Vertices 0..24 correspond to f25_elements[0..24].

    # Legendre symbol in F_25: an element x != 0 is a square iff x^12 = 1 (since |F_25^*| = 24, squares are subgroup of index 2).
    # Actually, x is a square iff x^12 == 1. Let's precompute which elements are squares.
    # F_25^* has order 24. The squares are the 12 elements with x^12 = 1.
    # Equivalently, x is a square iff x = y^2 for some y in F_25.

    def f25_pow(x, p):
        result = (1, 0)  # multiplicative identity
        base = x
        while p > 0:
            if p & 1:
                result = f25_mul(result, base)
            base = f25_mul(base, base)
            p >>= 1
        return result

    # Precompute square status for each element (index 0..24)
    is_square = [False] * 25
    for i in range(25):
        el = f25_elements[i]
        if el == (0, 0):
            is_square[i] = False  # 0 is not a square in the multiplicative sense
        else:
            # Check if el^12 == 1
            el12 = f25_pow(el, 12)
            is_square[i] = (el12 == (1, 0))

    # Precompute the index lookup: element (a,b) -> index
    elem_to_idx = {}
    for i in range(25):
        elem_to_idx[f25_elements[i]] = i

    def get_idx(el):
        return elem_to_idx[el]

    # Coloring rule for edge (u, v) where u, v in 0..24 (finite vertices):
    # Compute d = u - v in F_25. Color = Legendre symbol of d (0 if non-square, 1 if square) mapped to {0,1}.
    # But we need 5 colors. We need a more refined coloring.
    #
    # Better approach: Use the 5 "directions" in F_25. The 24 nonzero elements of F_25 can be partitioned
    # into 5 sets of 4 based on the cosets of the subgroup of order 4 (the 4th roots of unity).
    # Actually, F_25^* / (F_25^*)^4 gives 5 cosets? No, |F_25^*| = 24, so F_25^*/(F_25^*)^k has size k if k|24.
    # Let's use the 5 cosets of the subgroup of order 4 in F_25^*.
    # H = {x in F_25^* : x^4 = 1} has 4 elements.
    # The 5 cosets of H in F_25^* give 5 colors for finite-finite edges.
    # For edges involving infinity, we need a separate rule.
    #
    # Let's find H: elements x with x^4 = 1 in F_25.
    H_elements = []
    for i in range(25):
        el = f25_elements[i]
        if el == (0, 0):
            continue
        el4 = f25_pow(el, 4)
        if el4 == (1, 0):
            H_elements.append(i)

    # H should have 4 elements
    # Cosets: for each h in H, the coset is {h * x : x in F_25^*}
    # We assign color based on which coset d = u - v falls into.
    # There are 5 cosets (since 24/4 = 6? No, 24/4 = 6 cosets. That gives 6 colors, too many.)
    #
    # Let me reconsider. We need exactly 5 colors.
    # Alternative: Use the norm map N: F_25 -> F_5, N(a+b*alpha) = a^2 + b^2 (since alpha^2 = -1).
    # N maps F_25^* onto F_5^* (which has 4 elements). So N gives 4 values for nonzero elements.
    # Plus 0 for the zero element. That's 5 values: {0, 1, 2, 3, 4} in F_5.
    # This gives exactly 5 colors!
    #
    # Color of edge (u,v) for u,v in 0..24: N(u - v) mod 5, where N(a+b*alpha) = a^2 + b^2 mod 5.
    # For edges involving infinity (vertex 25): we need a rule.
    # By symmetry, we can define color(25, u) = N(u) mod 5 (treating infinity as "0" in some sense).
    # Let's verify this works.

    def norm(el):
        a, b = el
        return (a * a + b * b) % 5

    n = 26
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges

    # Build edge list in lexicographic order
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Both finite: color = N(i - j) mod 5
                d = sub_table[i][j]
                d_idx = get_idx(d)
                colors[edge_idx] = norm(f25_elements[d_idx])
            elif i == 25 or j == 25:
                # One is infinity
                finite = j if i == 25 else i
                colors[edge_idx] = norm(f25_elements[finite])
            edge_idx += 1

    # Verify the coloring
    def verify(colors, n):
        if len(colors) != n * (n - 1) // 2:
            return False
        color_adj = [[set() for _ in range(n)] for _ in range(5)]
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx]
                edge_idx += 1
                color_adj[c][i].add(j)
                color_adj[c][j].add(i)

        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = len(color_adj[c][i] & color_adj[c][j])
                    if common >= 2:
                        return False
        return True

    if verify(colors, n):
        return {'n': n, 'colors': colors}

    # If the algebraic construction fails verification (shouldn't happen), fall back to greedy
    def greedy_packing(n, seed_val):
        random.seed(seed_val)
        num_edges = n * (n - 1) // 2
        endpoints = []
        for i in range(n):
            for j in range(i + 1, n):
                endpoints.append((i, j))

        nbrs = [[set() for _ in range(5)] for _ in range(n)]
        CN = [[[0] * n for _ in range(n)] for _ in range(5)]
        colors = [0] * num_edges
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)

        for e_idx in edge_order:
            u, v = endpoints[e_idx]
            best_c = -1
            best_cost = float('inf')
            for c in range(5):
                cost = 0
                for x in nbrs[v][c]:
                    if x != u:
                        cost += CN[c][u][x]
                for x in nbrs[u][c]:
                    if x != v:
                        cost += CN[c][v][x]
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
            colors[e_idx] = best_c
            c = best_c
            for x in nbrs[v][c]:
                if x != u:
                    CN[c][u][x] += 1
                    CN[c][x][u] += 1
            for x in nbrs[u][c]:
                if x != v:
                    CN[c][v][x] += 1
                    CN[c][x][v] += 1
            nbrs[u][c].add(v)
            nbrs[v][c].add(u)

        if verify(colors, n):
            return colors
        return None

    for n_try in range(25, 10, -1):
        if time.time() > deadline - 5:
            break
        for r in range(10):
            if time.time() > deadline - 3:
                break
            res = greedy_packing(n_try, seed_val=2000 + r)
            if res is not None:
                return {'n': n_try, 'colors': res}

    # Fallback
    n = 11
    num_edges = n * (n - 1) // 2
    colors = [i % 5 for i in range(num_edges)]
    return {'n': n, 'colors': colors}

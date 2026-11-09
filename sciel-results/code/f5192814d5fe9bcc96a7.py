import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill

    def edge_index(n, i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index(n, i, j)]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def has_c6_color(n, adj_c):
        """Return True if color class adj_c (list of n bitmasks) contains a C6."""
        for a in range(n):
            na = adj_c[a]
            # iterate neighbors b of a
            b = na & -na
            while b:
                bb = b.bit_length() - 1
                na = na ^ b
                b = na & -na
                # path a - b - c - d - e - f - a, all distinct
                nb = adj_c[bb] & ~(1 << a)
                c = nb & -nb
                while c:
                    cc = c.bit_length() - 1
                    nb = nb ^ c
                    c = nb & -c
                    # middle vertex d
                    nd = adj_c[cc] & ~(1 << a) & ~(1 << bb) & ~(1 << cc)
                    d = nd & -nd
                    while d:
                        dd = d.bit_length() - 1
                        nd = nd ^ d
                        d = nd & -d
                        # f must be neighbor of both e and a, distinct from a,b,c,d
                        ne = adj_c[dd] & ~(1 << a) & ~(1 << bb) & ~(1 << cc) & ~(1 << dd)
                        if ne & adj_c[a]:
                            return True
        return False

    def count_c6_color(n, adj_c):
        count = 0
        for a in range(n):
            na = adj_c[a]
            b = na & -na
            while b:
                bb = b.bit_length() - 1
                na = na ^ b
                b = na & -na
                nb = adj_c[bb] & ~(1 << a)
                c = nb & -nb
                while c:
                    cc = c.bit_length() - 1
                    nb = nb ^ c
                    c = nb & -c
                    nd = adj_c[cc] & ~(1 << a) & ~(1 << bb) & ~(1 << cc)
                    d = nd & -nd
                    while d:
                        dd = d.bit_length() - 1
                        nd = nd ^ d
                        d = nd & -d
                        ne = adj_c[dd] & ~(1 << a) & ~(1 << bb) & ~(1 << cc) & ~(1 << dd)
                        count += bin(ne & adj_c[a]).count('1')
        return count

    def check_c6(n, colors):
        adj = build_adj(n, colors)
        for c in range(4):
            if has_c6_color(n, adj[c]):
                return True
        return False

    def count_c6(n, colors):
        adj = build_adj(n, colors)
        total = 0
        for c in range(4):
            total += count_c6_color(n, adj[c])
        return total

    def precompute_path4(v, adj_list):
        """Path4[c][i][d] for i,d < v: simple path of 4 edges i-a-b-e-d in color c."""
        Path4 = [[[False] * v for _ in range(v)] for _ in range(4)]
        for c in range(4):
            for i in range(v):
                if not adj_list[c][i]:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    for a in adj_list[c][i]:
                        if a == d:
                            continue
                        for b in adj_list[c][a]:
                            if b == i or b == d or b == a:
                                continue
                            for e in adj_list[c][b]:
                                if e == i or e == d or e == a or e == b:
                                    continue
                                if d in adj_list[c][e]:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    Path4[c][i][d] = found
        return Path4

    def backtrack_solve(n_target):
        """Incremental backtracking: add vertices 0..n-1, keep best partial coloring."""
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        best = {'v': 1, 'adj': [row[:] for row in adj]}

        def search(v):
            nonlocal best
            if time.time() > deadline:
                return False
            if v == n_target:
                return True
            if v > best['v']:
                best['v'] = v
                best['adj'] = [row[:] for row in adj]
            if v == 0:
                return search(v + 1)
            Path4 = precompute_path4(v, adj_list)

            def assign(i):
                if time.time() > deadline:
                    return False
                if i == v:
                    return search(v + 1)
                for c in range(4):
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c and Path4[c][i][d]:
                            valid = False
                            break
                    if not valid:
                        continue
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    if assign(i + 1):
                        return True
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                return False

            return assign(0)

        search(0)
        return best

    def anneal_repair(n, colors):
        """Hill-climb / simulated annealing to eliminate C6s."""
        colors = colors[:]
        num_edges = n * (n - 1) // 2
        cur_pen = count_c6(n, colors)
        best = colors[:]
        best_pen = cur_pen
        T = 2.0
        it = 0
        while time.time() < deadline - 2:
            it += 1
            if it % 400 == 0:
                T = max(0.05, T * 0.95)
            idx = random.randrange(num_edges)
            old_c = colors[idx]
            new_c = random.randrange(4)
            if new_c == old_c:
                continue
            colors[idx] = new_c
            new_pen = count_c6(n, colors)
            if new_pen < cur_pen or (new_pen > cur_pen and random.random() < (2.718281828 ** ((cur_pen - new_pen) / max(T, 1e-9)))):
                cur_pen = new_pen
                if new_pen < best_pen:
                    best_pen = new_pen
                    best = colors[:]
                    if best_pen == 0:
                        break
            else:
                colors[idx] = old_c
        return best, best_pen

    # --- main ---
    # 1) Try exact incremental backtracking starting from a large target.
    for target in [18, 17, 16]:
        if time.time() > deadline - 4:
            break
        best = backtrack_solve(target)
        if best['v'] >= target:
            colors = []
            for i in range(target):
                for j in range(i + 1, target):
                    colors.append(best['adj'][i][j])
            if not check_c6(target, colors):
                return {'n': target, 'colors': colors}

    # 2) Fallback: repair the best partial coloring found.
    if best is not None and best['v'] >= 6:
        v = best['v']
        colors = []
        for i in range(v):
            for j in range(i + 1, v):
                colors.append(best['adj'][i][j])
        # If partial already valid, return it.
        if not check_c6(v, colors):
            return {'n': v, 'colors': colors}
        # Try to repair the partial (remove C6s) then optionally extend.
        repaired, pen = anneal_repair(v, colors)
        if pen == 0:
            return {'n': v, 'colors': repaired}

    # 3) Last resort: anneal from a valid base to try to reach n=16+.
    base_n = 9
    base = [min(j - i, base_n - j + i) - 1 for i in range(base_n) for j in range(i + 1, base_n)]
    # Pad to a larger n with random colors, then anneal.
    for target in [16, 15, 14, 13]:
        if time.time() > deadline - 3:
            break
        colors = base[:]
        # add edges for new vertices
        for i in range(base_n, target):
            for j in range(i + 1, target):
                colors.append(random.randrange(4))
        # also edges between old and new vertices
        # rebuild properly
        colors = [random.randrange(4) for _ in range(target * (target - 1) // 2)]
        repaired, pen = anneal_repair(target, colors)
        if pen == 0:
            return {'n': target, 'colors': repaired}

    # 4) Return the known valid seed.
    return {'n': base_n, 'colors': base}

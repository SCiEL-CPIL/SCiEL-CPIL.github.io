"""Stronger algorithm: greedy seed + local search + branch-and-bound with vertex-cover bound."""
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[i][j] = lcm(i, j)

    # For each pair (a,b), list of c that form conflict triple
    # conflict_with_pair[(a,b)] = frozenset of c
    conflict_with_pair = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = lcm_table[a][b]
            conflicts = set()
            for c in range(1, N + 1):
                if c != a and c != b and lcm_table[a][c] == L and lcm_table[b][c] == L:
                    conflicts.add(c)
            if conflicts:
                conflict_with_pair[(a, b)] = frozenset(conflicts)

    # 2-shadow adjacency: adj[v] = set of u such that (v,u) is in some conflict triple
    adj = [set() for _ in range(N + 1)]
    for (a, b), conflicts in conflict_with_pair.items():
        for c in conflicts:
            adj[a].add(b)
            adj[a].add(c)
            adj[b].add(a)
            adj[b].add(c)
            adj[c].add(a)
            adj[c].add(b)

    def can_add(S, x):
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                key = (a, b) if a < b else (b, a)
                if key in conflict_with_pair and x in conflict_with_pair[key]:
                    return False
        return True

    def greedy_seed():
        S = []
        for x in range(1, N + 1):
            if can_add(S, x):
                S.append(x)
        return set(S)

    def local_search(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
            for x in list(S):
                S.remove(x)
                added = False
                for y in range(1, N + 1):
                    if y not in S and can_add(S, y):
                        S.add(y)
                        added = True
                        improved = True
                        break
                if not added:
                    S.add(x)
        return S

    def greedy_vertex_cover(vertices):
        remaining = set(vertices)
        cover = 0
        while remaining:
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_deg == 0:
                break
            cover += 1
            remaining.discard(best_v)
            remaining -= adj[best_v]
        return cover

    def branch_and_bound(S, remaining, best_size):
        nonlocal best_size
        if time.time() > deadline:
            return best_size
        if not remaining:
            if len(S) > best_size:
                best_size = len(S)
            return best_size
        vc = greedy_vertex_cover(set(remaining))
        upper_bound = len(S) + len(remaining) - vc
        if upper_bound <= best_size:
            return best_size
        remaining_list = list(remaining)
        best_v = remaining_list[0]
        best_deg = -1
        for v in remaining_list:
            deg = len(adj[v] & remaining)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if can_add(S, best_v):
            new_S = S | {best_v}
            new_remaining = frozenset(remaining - {best_v})
            best_size = branch_and_bound(new_S, new_remaining, best_size)
            if time.time() > deadline:
                return best_size
        new_remaining = frozenset(remaining - {best_v})
        best_size = branch_and_bound(S, new_remaining, best_size)
        return best_size

    S = greedy_seed()
    best_size = len(S)
    S = local_search(S)
    best_size = len(S)
    best_set = set(S)

    if time.time() > deadline:
        return sorted(best_set)

    remaining = frozenset(set(range(1, N + 1)) - best_set)
    best_size = branch_and_bound(frozenset(best_set), remaining, best_size)

    if time.time() < deadline - 5:
        for perm in [range(N, 0, -1), range(1, N + 1, 2), range(2, N + 1, 2)]:
            S2 = set()
            for x in perm:
                if x not in S2 and can_add(S2, x):
                    S2.add(x)
            S2 = local_search(S2)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = S2
            if time.time() > deadline - 10:
                break

    return sorted(best_set)

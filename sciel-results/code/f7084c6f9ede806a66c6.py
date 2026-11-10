import time
import math
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 256
    max_sum = 2 * N

    # Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)

    # Filter vertices: 2*v must not be a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)

    if not valid_vertices:
        return [1]

    M = len(valid_vertices)

    # Build conflict graph
    conflict = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                conflict[i].add(j)
                conflict[j].add(i)

    # Convert to sorted lists for speed
    conflict_lists = [sorted(cs) for cs in conflict]

    # Precompute conflict sets as frozensets for fast membership
    conflict_sets = [frozenset(cs) for cs in conflict]

    def greedy_init():
        """Greedy: pick vertices with fewest conflicts first."""
        order = sorted(range(M), key=lambda i: len(conflict_lists[i]))
        S = set()
        for i in order:
            if not S & conflict_sets[i]:
                S.add(i)
        return S

    def local_search(S):
        """1-move and 2-move local search."""
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try adding
            for i in range(M):
                if i not in S and not (S & conflict_sets[i]):
                    S.add(i)
                    improved = True
            # Try swapping: remove one, add another
            if not improved:
                best_gain = 0
                best_move = None
                for u in list(S):
                    S_without_u = S - {u}
                    # Find vertices that can be added
                    addable = [v for v in range(M) if v not in S_without_u and not (S_without_u & conflict_sets[v])]
                    if len(addable) > 1:
                        # Try adding multiple
                        for a in addable:
                            S_new = S_without_u | {a}
                            # Try to add more
                            changed = True
                            while changed:
                                changed = False
                                for v in range(M):
                                    if v not in S_new and not (S_new & conflict_sets[v]):
                                        S_new.add(v)
                                        changed = True
                            gain = len(S_new) - len(S)
                            if gain > best_gain:
                                best_gain = gain
                                best_move = S_new
                if best_move is not None:
                    S = best_move
                    improved = True
        return S

    # Initial solution
    best_S = greedy_init()
    best_S = local_search(best_S)
    best_size = len(best_S)

    # Branch and bound
    # State: (active_set, forced_in, forced_out)
    # active_set: vertices not yet decided
    # forced_in: vertices forced to be in solution
    # forced_out: vertices forced to be out

    def upper_bound(active, forced_in, forced_out):
        """Compute upper bound on solution size."""
        # Upper bound = |forced_in| + |active| (trivial)
        # Better: LP relaxation
        if not active:
            return len(forced_in)

        # Solve LP: max sum(x_i) s.t. x_i + x_j <= 1 for conflicts, 0 <= x_i <= 1
        # For vertices in forced_out, x_i = 0
        # For vertices in forced_in, x_i = 1 (and their neighbors in active must be 0)

        # First, remove vertices that conflict with forced_in
        active_clean = set(active)
        for fi in forced_in:
            for nb in conflict_lists[fi]:
                if nb in active_clean:
                    active_clean.discard(nb)

        if not active_clean:
            return len(forced_in)

        # LP on active_clean
        n = len(active_clean)
        if n <= 1:
            return len(forced_in) + n

        # Build LP
        c = -np.ones(n)
        A_rows = []
        b_vals = []

        active_list = sorted(active_clean)
        idx_map = {v: i for i, v in enumerate(active_list)}

        # Conflict constraints among active_clean
        for i in active_list:
            for j in conflict_lists[i]:
                if j > i and j in idx_map:
                    row = np.zeros(n)
                    row[idx_map[i]] = 1
                    row[idx_map[j]] = 1
                    A_rows.append(row)
                    b_vals.append(1)

        if not A_rows:
            return len(forced_in) + n

        A = np.array(A_rows)
        b = np.array(b_vals)

        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, method='highs')

        if res.success:
            lp_val = -res.fun
            return len(forced_in) + lp_val
        else:
            return len(forced_in) + n  # Fallback: trivial bound

    def branch_and_bound():
        """Branch and bound for maximum independent set."""
        nonlocal best_S, best_size

        # Start with all vertices active
        active = set(range(M))
        forced_in = set()
        forced_out = set()

        # Stack for DFS: (active, forced_in, forced_out, bound)
        stack = []

        # Initial bound
        ub = upper_bound(active, forced_in, forced_out)
        if ub > best_size:
            stack.append((active, forced_in, forced_out, ub))

        nodes_explored = 0

        while stack:
            if time.time() > deadline - 3.0:
                break

            active, forced_in, forced_out, ub = stack.pop()
            nodes_explored += 1

            if ub <= best_size:
                continue

            if not active:
                candidate = forced_in
                if len(candidate) > best_size:
                    best_S = candidate
                    best_size = len(candidate)
                continue

            # Try to find a good branching variable
            # Heuristic: pick variable with highest degree in active subgraph
            # or one that appears in many conflicts

            # Compute degrees within active
            deg = {}
            for i in active:
                d = 0
                for nb in conflict_lists[i]:
                    if nb in active:
                        d += 1
                deg[i] = d

            # Pick variable with max degree (or random among top)
            max_deg = max(deg.values()) if deg else 0
            candidates = [i for i in active if deg[i] == max_deg]
            if len(candidates) > 1:
                # Randomize to break ties
                v = random.choice(candidates)
            else:
                v = candidates[0] if candidates else None

            if v is None:
                continue

            # Branch 1: v in solution
            new_forced_in_1 = forced_in | {v}
            new_forced_out_1 = forced_out
            new_active_1 = active - {v}
            # Remove neighbors of v from active
            for nb in conflict_lists[v]:
                if nb in new_active_1:
                    new_active_1.discard(nb)
                    new_forced_out_1.add(nb)

            ub1 = upper_bound(new_active_1, new_forced_in_1, new_forced_out_1)
            if ub1 > best_size:
                stack.append((new_active_1, new_forced_in_1, new_forced_out_1, ub1))

            # Branch 2: v not in solution
            new_forced_in_2 = forced_in
            new_forced_out_2 = forced_out | {v}
            new_active_2 = active - {v}

            ub2 = upper_bound(new_active_2, new_forced_in_2, new_forced_out_2)
            if ub2 > best_size:
                stack.append((new_active_2, new_forced_in_2, new_forced_out_2, ub2))

        return best_S, best_size

    # Run branch and bound
    best_S, best_size = branch_and_bound()

    # Final local search polish
    best_S = local_search(best_S)

    # Convert back to vertex values
    result = [valid_vertices[i] for i in sorted(best_S)]

    if not result:
        return [1]

    return result

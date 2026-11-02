"""Improved program: branch-and-bound MIS on conflict graph with Tomita coloring + greedy fallback for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit

    D = 11

    # Generate all candidate vectors
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Focus on weight-4 vectors (norm^2 = 4) as primary candidates
    wt4_indices = np.where(norms_sq == 4)[0]
    n4 = len(wt4_indices)  # 5280

    # Build adjacency list for conflict graph on weight-4 vectors
    # For efficiency, use bitsets or sorted lists
    adj = [[] for _ in range(n4)]
    for i in range(n4):
        gi = wt4_indices[i]
        # Find all j != i that conflict with i
        conflicts = np.where(conflict[gi] == 1)[0]
        # Map to local indices
        for cj in conflicts:
            if cj in wt4_indices:
                # Find local index
                # This is slow; let's precompute a mapping
                pass

    # Better approach: precompute local adjacency
    # Create a mapping from global index to local index for weight-4 vectors
    global_to_local = {}
    for i, gi in enumerate(wt4_indices):
        global_to_local[gi] = i

    adj = [set() for _ in range(n4)]
    for i in range(n4):
        gi = wt4_indices[i]
        conflicts = np.where(conflict[gi] == 1)[0]
        for cj in conflicts:
            if cj in global_to_local:
                j = global_to_local[cj]
                if j != i:
                    adj[i].add(j)
                    adj[j].add(i)

    # Convert to sorted lists for faster iteration
    adj = [sorted(s) for s in adj]

    # Tomita coloring for upper bound
    # Color the graph such that adjacent vertices have different colors
    # The number of colors is an upper bound on the size of any independent set

    def color_graph(adj, n):
        """Assign colors to vertices using greedy coloring (first-fit)."""
        colors = [-1] * n
        for v in range(n):
            # Find the smallest color not used by neighbors
            used_colors = set()
            for u in adj[v]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
        return colors

    # Branch and bound for Maximum Independent Set
    best_size = 0
    best_set = []

    def expand(R, C, P):
        """
        R: set of selected vertices (current independent set)
        C: candidate vertices that can be added
        P: vertices that cannot be added (conflict with R or already considered)
        """
        global best_size, best_set

        if time.time() > deadline:
            return

        if len(C) == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            return

        # Upper bound: |R| + |C|
        if len(R) + len(C) <= best_size:
            return

        # Color C to get a tighter upper bound
        # We need to color only the subgraph induced by C
        c_list = list(C)
        n_c = len(c_list)
        if n_c == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            return

        # Build local adjacency for C
        c_to_local = {v: i for i, v in enumerate(c_list)}
        local_adj = [[] for _ in range(n_c)]
        for i, v in enumerate(c_list):
            for u in adj[v]:
                if u in c_to_local:
                    local_adj[i].append(c_to_local[u])

        # Greedy coloring
        colors = [-1] * n_c
        for i in range(n_c):
            used = set()
            for j in local_adj[i]:
                if colors[j] != -1:
                    used.add(colors[j])
            c = 0
            while c in used:
                c += 1
            colors[i] = c

        max_color = max(colors) + 1 if n_c > 0 else 0

        # Upper bound using coloring
        if len(R) + max_color <= best_size:
            return

        # Select vertex with fewest candidates (most constrained)
        # Pick vertex with highest color (or fewest neighbors in C)
        # For simplicity, pick the last vertex in C
        v = c_list[-1]
        C.remove(v)

        # Branch 1: include v
        R.add(v)
        # Remove v and its neighbors from C
        new_C = set(C)
        for u in adj[v]:
            if u in new_C:
                new_C.discard(u)
        expand(R, new_C, P)

        # Branch 2: exclude v
        R.discard(v)
        P.add(v)
        expand(R, C, P)
        P.discard(v)

    # Initial call
    C = set(range(n4))
    P = set()
    R = set()
    expand(R, C, P)

    # If BB didn't find a good solution, use greedy as fallback
    if best_size == 0 or time.time() > deadline:
        # Min-degree greedy
        remaining = set(range(n4))
        selected = []
        while time.time() < deadline and remaining:
            # Find vertex with fewest neighbors in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len([u for u in adj[v] if u in remaining])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                if time.time() > deadline:
                    break
            if best_v is None:
                break
            selected.append(best_v)
            remaining.discard(best_v)
            for u in adj[best_v]:
                if u in remaining:
                    remaining.discard(u)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected

    # Map back to global indices
    selected_global = [wt4_indices[i] for i in best_set]

    # Try to add weight-2 and weight-1 vectors
    if time.time() < deadline - 5:
        sel_set = set(selected_global)
        # Find candidates compatible with all selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected_global:
            compat_with_all = compat_with_all & compat[s]
        compat_with_all[list(sel_set)] = 0

        addable = np.where(compat_with_all > 0)[0]
        if len(addable) > 0:
            # Greedy add
            new_selected = list(selected_global)
            for a in addable:
                if time.time() > deadline - 2:
                    break
                # Check if a is compatible with all in new_selected
                ok = True
                for s in new_selected:
                    if not compat[a, s]:
                        ok = False
                        break
                if ok:
                    new_selected.append(a)

            if len(new_selected) > len(selected_global):
                selected_global = new_selected

    if len(selected_global) == 0:
        # Fallback: return a single vector
        selected_global = [0]

    result = candidates[selected_global]
    return result.astype(np.int64)

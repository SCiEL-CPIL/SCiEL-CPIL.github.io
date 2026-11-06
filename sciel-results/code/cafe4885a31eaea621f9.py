"""Stronger approach: model as CSP with backtracking + value ordering, and use a deterministic
greedy edge-insertion that maintains a valid coloring whenever possible."""
import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0   # 10s safety margin under the 150s wall-clock
    n = 20

    # Precompute all 4-cycles as tuples (u, v, w, x) representing the cycle u-v-w-x-u.
    # We will enumerate once (static set of all 4-cycles in K20).
    cycles = []
    for u, w in itertools.combinations(range(n), 2):
        for v in range(n):
            if v == u or v == w:
                continue
            for x in range(n):
                if x == u or x == w or x == v:
                    continue
                # We will deduplicate later by sorting the unordered cycle edges
                # For now, record cycle as (u, v, w, x) meaning edges (u,v),(v,w),(w,x),(x,u)
                cycles.append((u, v, w, x))

    # Deduplicate cycles by the frozenset of its 4 edges.
    seen = set()
    unique_cycles = []
    for (u, v, w, x) in cycles:
        edges = frozenset(frozenset((a, b)) for a, b in [(u, v), (v, w), (w, x), (x, u)])
        if edges not in seen:
            seen.add(edges)
            unique_cycles.append((u, v, w, x))

    # Map each edge to the list of cycle indices in which it appears.
    # This enables incremental counting of "bad" cycles per edge color.
    edge_to_cycles = {}
    for idx, (u, v, w, x) in enumerate(unique_cycles):
        for a, b in [(u, v), (v, w), (w, x), (x, u)]:
            key = (min(a, b), max(a, b))
            edge_to_cycles.setdefault(key, []).append(idx)

    # Count how many cycles each edge belongs to (useful for heuristics).
    edge_cycle_count = {key: len(v) for key, v in edge_to_cycles.items()}

    # We'll use a coloring over the full edge set E_all of K20.
    E_all = [(min(a, b), max(a, b)) for a in range(n) for b in range(a + 1, n)]

    # Precompute, for each edge, the set of colors that would immediately cause a violation
    # in any of its cycles if we fix the other three edges' colors.
    # We'll compute this on-the-fly during search.

    # Strategy:
    # 1. Try to find a rainbow 4-cycle coloring of K20 via backtracking CSP.
    #    If successful, return all 190 edges.
    # 2. If not, greedily insert edges in an order, maintaining validity, and return that.
    #    To bias toward density, we insert edges in an order that minimizes incremental conflicts,
    #    e.g., prefer edges with fewer cycles, or a fixed order.

    # --- Phase 1: CSP backtracking to color K20 ---
    # We'll assign colors to edges one by one.
    # Variable ordering: choose the uncolored edge that is in the most currently violated cycles.
    # Since we start with all cycles violated (no colors assigned), we can order by edge_cycle_count descending.
    # Value ordering: for a chosen edge, try colors in order of how many cycles they would satisfy.

    # We maintain:
    #   color_of_edge: dict edge -> color or None
    #   cycle_satisfied: for each cycle, whether all 4 edges are colored and all distinct.
    #   cycle_partial: for each cycle, how many of its 4 edges are colored, and the set of colors used so far.
    #
    # To be efficient, we maintain for each cycle:
    #   colored_edges: set of edges in the cycle that are colored
    #   colors_used: set of colors among those colored edges
    # A cycle is "satisfied" if all 4 edges are colored and colors_used has size 4.
    # A cycle is "violated" if all 4 edges are colored and colors_used size < 4.
    # A cycle is "pending" if fewer than 4 edges are colored.

    # We'll do a depth-first search with backtracking.
    # To keep it tractable, we limit the number of nodes explored.
    max_nodes = 200000

    # Order edges: those in more cycles first.
    edge_order = sorted(E_all, key=lambda e: -edge_cycle_count.get(e, 0))

    # Precompute for each edge the list of cycles it is in.
    edge_to_cycles_list = {key: v for key, v in edge_to_cycles.items()}

    # We'll represent state as:
    #   colors: list of length |E_all| with -1 for uncolored, else color 0..19.
    #   We'll map edge to index.
    edge_to_idx = {e: i for i, e in enumerate(E_all)}

    # For quick access, for each cycle, the indices of its 4 edges.
    cycle_edge_idxs = []
    for (u, v, w, x) in unique_cycles:
        edges = [(u, v), (v, w), (w, x), (x, u)]
        idxs = tuple(edge_to_idx[(min(a, b), max(a, b))] for a, b in edges)
        cycle_edge_idxs.append(idxs)

    # Number of cycles.
    C = len(unique_cycles)

    # We'll use a recursive DFS with backtracking.
    # To speed up, we maintain:
    #   cycle_colored_count: for each cycle, how many of its 4 edges are colored.
    #   cycle_colors: for each cycle, a bitmask or set of colors used (only for colored edges).
    # We'll use a list of 4-element arrays per cycle for the colors, with -1 for uncolored.

    # Let's implement a non-recursive backtracking with an explicit stack to avoid recursion limits.
    # But for simplicity and given the limits, recursion with a node limit is fine.

    # We'll use a global counter for nodes.
    node_count = 0

    # State:
    colors = [-1] * len(E_all)
    # For each cycle, we track the colors of its 4 edges (as a tuple of 4, -1 if uncolored).
    # We'll maintain this as a list of lists for mutability.
    cycle_colors = [[-1, -1, -1, -1] for _ in range(C)]

    # For each cycle, we track how many edges are colored.
    cycle_colored = [0] * C

    # For each cycle, we track the set of colors used among colored edges (as a bitmask).
    # Colors are 0..19, so 20 bits.
    cycle_color_mask = [0] * C

    # For each edge, we track which cycles it's in (list of cycle indices).
    edge_cycles = [[] for _ in range(len(E_all))]
    for ci, idxs in enumerate(cycle_edge_idxs):
        for ei in idxs:
            edge_cycles[ei].append(ci)

    def assign_color(ei, c):
        """Assign color c to edge ei, updating cycle state."""
        colors[ei] = c
        bit = 1 << c
        for ci in edge_cycles[ei]:
            idxs = cycle_edge_idxs[ci]
            # Find position of ei in idxs
            pos = idxs.index(ei)
            cycle_colors[ci][pos] = c
            cycle_colored[ci] += 1
            cycle_color_mask[ci] |= bit

    def unassign_color(ei):
        """Unassign color from edge ei, updating cycle state."""
        c = colors[ei]
        colors[ei] = -1
        bit = 1 << c
        for ci in edge_cycles[ei]:
            idxs = cycle_edge_idxs[ci]
            pos = idxs.index(ei)
            cycle_colors[ci][pos] = -1
            cycle_colored[ci] -= 1
            # Recompute mask for this cycle
            mask = 0
            for cc in cycle_colors[ci]:
                if cc != -1:
                    mask |= (1 << cc)
            cycle_color_mask[ci] = mask

    # Popcount function.
    def popcount(x):
        return bin(x).count('1')

    def is_cycle_satisfied(ci):
        return cycle_colored[ci] == 4 and popcount(cycle_color_mask[ci]) == 4

    def is_cycle_violated(ci):
        return cycle_colored[ci] == 4 and popcount(cycle_color_mask[ci]) < 4

    # We'll do DFS.
    # Variable ordering: choose the uncolored edge that is in the most "active" cycles.
    # For simplicity, we'll use a static order (edge_order) but try to pick the one with the most
    # currently pending cycles that involve it. To keep it simple, we'll use the static order.

    # To make value ordering effective, for a chosen edge, we'll try colors in order of how many
    # of its cycles would become satisfied (or remain non-violated) if we assign that color.

    def dfs(depth):
        nonlocal node_count
        if node_count > max_nodes:
            return False
        if time.time() > deadline - 5.0:
            return False

        if depth == len(edge_order):
            # All edges colored. Check if all cycles are satisfied.
            # Actually, if we maintain that we never create a violation, then at the end all are satisfied.
            # But we might have pending cycles that are not yet checked? No, all edges are colored, so all cycles are either satisfied or violated.
            # We'll check if any cycle is violated.
            for ci in range(C):
                if is_cycle_violated(ci):
                    return False
            return True

        node_count += 1

        ei = edge_order[depth]

        # Value ordering: try colors in order of a heuristic.
        # Heuristic: for each color c, count how many of ei's cycles would be "good" if we assign c.
        # "Good" means: after assigning c, the cycle is either not yet fully colored (pending) or is satisfied.
        # We want to avoid colors that would cause a violation in any of ei's cycles.
        # Let's compute for each color c the number of cycles that would become violated if we assign c.
        # We want to try colors with 0 violations first.

        # To compute this efficiently, for each cycle ci in edge_cycles[ei]:
        #   If cycle_colored[ci] < 3: assigning any color to ei will leave the cycle with 3 or 4 colored edges.
        #     If cycle_colored[ci] == 3: after assigning, it will have 4. It is satisfied if the new color is not in the mask.
        #     If cycle_colored[ci] == 2: after assigning, it will have 3. No violation possible (violation only at 4).
        #   So only cycles with cycle_colored[ci] == 3 can become violated by our assignment.
        #   For such a cycle, the color c is bad if c is in cycle_color_mask[ci].

        # So for a given edge ei, the set of colors that would cause a violation is the union, over all ci in edge_cycles[ei] with cycle_colored[ci] == 3, of the colors in cycle_color_mask[ci].
        # We want to avoid those colors.

        bad_colors = [0] * 20
        for ci in edge_cycles[ei]:
            if cycle_colored[ci] == 3:
                mask = cycle_color_mask[ci]
                for c in range(20):
                    if mask & (1 << c):
                        bad_colors[c] += 1

        # Sort colors by bad_colors ascending, then by c.
        color_order = sorted(range(20), key=lambda c: (bad_colors[c], c))

        for c in color_order:
            # Check if assigning c to ei would cause any violation.
            # This is the case if there exists a cycle ci in edge_cycles[ei] with cycle_colored[ci] == 3 and c in cycle_color_mask[ci].
            # If bad_colors[c] == 0, then no such cycle.
            if bad_colors[c] > 0:
                continue

            # Assign.
            assign_color(ei, c)

            if dfs(depth + 1):
                return True

            unassign_color(ei)

        return False

    # Start DFS.
    found = dfs(0)

    if found:
        # We have a valid coloring of K20.
        result = []
        for i, (u, v) in enumerate(E_all):
            c = colors[i]
            if c != -1:
                result.append([u, v, c])
        return result

    # --- Phase 2: Greedy edge insertion maintaining validity ---
    # We'll insert edges one by one in an order. For each edge, we try to find a color that
    # does not create any new violation in any cycle that includes this edge.
    # A cycle that includes this edge and has the other 3 edges already colored is at risk.
    # We maintain the same state structure.

    # Reset state.
    colors = [-1] * len(E_all)
    cycle_colors = [[-1, -1, -1, -1] for _ in range(C)]
    cycle_colored = [0] * C
    cycle_color_mask = [0] * C

    # We'll insert edges in an order that tries to keep the graph dense.
    # A good order might be to insert edges in a way that the resulting graph is as dense as possible.
    # For simplicity, we'll use a random order, but try each edge and if it can't be colored, skip it.
    # To bias toward density, we'll try multiple random orders and keep the best.

    # But to save time, we'll do a few random orders and pick the best.

    def try_greedy_order(order, time_limit):
        """Try to insert edges in the given order, maintaining validity.
        Returns the list of edges that were successfully inserted."""
        # Reset state.
        colors[:] = [-1] * len(E_all)
        for ci in range(C):
            cycle_colors[ci] = [-1, -1, -1, -1]
            cycle_colored[ci] = 0
            cycle_color_mask[ci] = 0

        inserted = []
        for ei in order:
            if time.time() > time_limit:
                break

            # Find a color c for edge ei such that no cycle in edge_cycles[ei] with cycle_colored[ci] == 3
            # has c in its mask.
            bad_colors = [0] * 20
            for ci in edge_cycles[ei]:
                if cycle_colored[ci] == 3:
                    mask = cycle_color_mask[ci]
                    for c in range(20):
                        if mask & (1 << c):
                            bad_colors[c] += 1

            # Find a color with bad_colors[c] == 0.
            chosen_c = None
            for c in range(20):
                if bad_colors[c] == 0:
                    chosen_c = c
                    break

            if chosen_c is None:
                # Cannot insert this edge.
                continue

            # Assign.
            colors[ei] = chosen_c
            bit = 1 << chosen_c
            for ci in edge_cycles[ei]:
                idxs = cycle_edge_idxs[ci]
                pos = idxs.index(ei)
                cycle_colors[ci][pos] = chosen_c
                cycle_colored[ci] += 1
                cycle_color_mask[ci] |= bit

            inserted.append(ei)

        return inserted

    # We'll try several random orders and keep the best.
    # To save time, we'll do up to 20 trials, each with a time limit.
    best_inserted = []
    num_trials = 20
    per_trial_time = (deadline - time.time()) / num_trials

    for trial in range(num_trials):
        order = E_all[:]
        random.shuffle(order)
        inserted = try_greedy_order(order, time.time() + per_trial_time)
        if len(inserted) > len(best_inserted):
            best_inserted = inserted

    # Convert best_inserted to result.
    result = []
    for ei in best_inserted:
        u, v = E_all[ei]
        c = colors[ei]
        if c != -1:
            result.append([u, v, c])

    return result

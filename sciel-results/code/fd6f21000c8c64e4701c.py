import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150 s limit

    n = 20
    # Precompute all 4-cycles as frozensets of 4 edge-keys.
    # An edge-key is (u, v) with u < v.
    # 4-cycle: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
    all_4cycles = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # cycle a-b-c-d-a
                    e1 = (a, b)
                    e2 = (b, c)
                    e3 = (c, d)
                    e4 = (d, a)
                    all_4cycles.append(frozenset([e1, e2, e3, e4]))
                    # cycle a-b-d-c-a
                    e1b = (a, b)
                    e2b = (b, d)
                    e3b = (c, d)
                    e4b = (a, c)
                    all_4cycles.append(frozenset([e1b, e2b, e3b, e4b]))
                    # cycle a-c-b-d-a
                    e1c = (a, c)
                    e2c = (b, c)
                    e3c = (b, d)
                    e4c = (a, d)
                    all_4cycles.append(frozenset([e1c, e2c, e3c, e4c]))

    # Map each edge to the list of 4-cycle indices that contain it.
    edge_to_cycles = {}
    for ci, cyc in enumerate(all_4cycles):
        for e in cyc:
            edge_to_cycles.setdefault(e, []).append(ci)

    # Precompute all edge pairs (u<v).
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)

    # A cycle is "bad" if all 4 of its edges are present AND it has a colour conflict.
    # But we colour greedily: the real constraint is that when we have a set of edges,
    # we must be able to assign colours so every 4-cycle is rainbow.
    # Simpler model: we decide colours as we go. We'll use a greedy colouring after
    # the edge set is fixed, and only keep the edge set if a valid colouring exists.

    def try_colour(edges_present):
        """Given a set of present edges, greedily assign colours 0..19 so every 4-cycle
        that is fully present is rainbow. Return list of [u,v,c] or None on failure."""
        # Build adjacency of present edges.
        adj = [[-1] * n for _ in range(n)]
        present = set(edges_present)
        # Greedy: order edges, assign a colour that doesn't conflict with any
        # already-assigned edge sharing a 4-cycle (where the 4-cycle is fully present).
        # We need: for each 4-cycle fully present, its 4 edges get distinct colours.
        # This is a graph-colouring-like constraint. Use a simple greedy with
        # conflict checking.
        # Precompute: for each pair of edges, do they share a 4-cycle that is fully present?
        # That's expensive; instead, when assigning colour to edge e, check all 4-cycles
        # containing e that are fully present, and ensure no other edge in that cycle
        # already has the same colour.
        edge_list = sorted(present)
        # For speed, map edge -> index in edge_list
        # We'll track colour per edge.
        colour = {}
        # For each edge, the set of 4-cycles it is in (that are fully present).
        for e in edge_list:
            colour[e] = -1
        # Order: try by edge index; assign smallest feasible colour.
        for e in edge_list:
            u, v = e
            feasible = True
            for c in range(n):
                ok = True
                for ci in edge_to_cycles.get(e, []):
                    cyc = all_4cycles[ci]
                    # check cycle fully present
                    if not all(ed in present for ed in cyc):
                        continue
                    # check no other edge in cyc has colour c
                    for ed in cyc:
                        if ed == e:
                            continue
                        if colour.get(ed, -1) == c:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    colour[e] = c
                    feasible = True
                    break
            if not feasible:
                return None
        # Verify all 4-cycles fully present are rainbow.
        for ci, cyc in enumerate(all_4cycles):
            if all(ed in present for ed in cyc):
                cols = [colour[ed] for ed in cyc]
                if len(set(cols)) != 4:
                    return None
        result = []
        for (u, v) in edge_list:
            result.append([u, v, colour[(u, v)]])
        return result

    best_edges = [[0, v, 0] for v in range(1, n)]  # star fallback
    best_count = len(best_edges)

    def count_conflict_cycles(present):
        """Count 4-cycles fully present that have a colour conflict, assuming a
        reference colouring. We instead count fully-present 4-cycles (potential
        conflicts) as a proxy for how 'hard' the instance is."""
        cnt = 0
        for cyc in all_4cycles:
            if all(ed in present for ed in cyc):
                cnt += 1
        return cnt

    for attempt in range(400):
        if time.time() > deadline:
            break

        # Start from complete graph.
        present = set(all_edges)

        # Iteratively remove edges to reduce fully-present 4-cycles, prioritising
        # edges in the most 4-cycles, with randomization.
        # Strategy: repeatedly pick an edge to remove. Score = number of fully-present
        # 4-cycles containing it (i.e., removing it breaks those cycles).
        # Stop when no fully-present 4-cycle remains, OR we hit time, OR we try to
        # colour and it works.
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Check if we can already colour successfully (only check periodically to save time).
            if it % 25 == 0:
                res = try_colour(present)
                if res is not None:
                    if len(res) > best_count:
                        best_count = len(res)
                        best_edges = res[:]
                    break
            # Find edges that are in at least one fully-present 4-cycle.
            # Compute for each edge the count of fully-present 4-cycles it's in.
            # To be efficient, only consider edges in cycles that are fully present.
            # Build list of fully-present cycles.
            # This is O(#cycles * 4) which is ~ (190 choose-ish) but #cycles for n=20:
            # number of 4-cycles in K20 = C(20,4)*3 = 4845*3 = 14535. Manageable.
            # Count per edge.
            edge_score = {}
            for cyc in all_4cycles:
                if all(ed in present for ed in cyc):
                    for ed in cyc:
                        edge_score[ed] = edge_score.get(ed, 0) + 1
            if not edge_score:
                # No fully-present 4-cycles: try to colour.
                res = try_colour(present)
                if res is not None:
                    if len(res) > best_count:
                        best_count = len(res)
                        best_edges = res[:]
                break
            # Pick edge to remove: weighted by score, with randomness.
            # Use a softmax-ish selection: pick among top scorers randomly.
            max_s = max(edge_score.values())
            top = [e for e, s in edge_score.items() if s >= max_s * 0.5]
            if not top:
                top = list(edge_score.keys())
            e = random.choice(top)
            present.discard(e)

        # Final colouring attempt for this attempt's edge set.
        if time.time() < deadline:
            res = try_colour(present)
            if res is not None and len(res) > best_count:
                best_count = len(res)
                best_edges = res[:]

    return best_edges

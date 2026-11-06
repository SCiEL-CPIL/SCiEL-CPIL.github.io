"""Strong edge colouring: induced-matching extraction + clique-cover set-cover with local refinement."""
import time
import math
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges):
        m = len(edges)
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        conflict = [0] * m
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]
        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a] |= 1 << b
                    conflict[b] |= 1 << a
        return conflict

    def max_induced_matching(edges, conflict, rng, time_budget=1.0):
        """Greedy + local improvement to find a large induced matching."""
        m = len(edges)
        start = time.time()
        # Greedy: pick edges in random order, keep if compatible with all picked
        order = list(range(m))
        rng.shuffle(order)
        selected = []
        selected_mask = 0
        for e in order:
            if conflict[e] & selected_mask == 0:
                selected.append(e)
                selected_mask |= 1 << e
        # Local improvement: try to swap in/out edges
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            sel_set = set(selected)
            for e in list(selected):
                # Try removing e and adding two others
                without_e = [x for x in selected if x != e]
                mask_wo = 0
                for x in without_e:
                    mask_wo |= 1 << x
                candidates = [x for x in range(m) if x not in sel_set and conflict[x] & mask_wo == 0]
                for i in range(len(candidates)):
                    if time.time() - start > time_budget:
                        break
                    for j in range(i + 1, len(candidates)):
                        if conflict[candidates[i]] & (1 << candidates[j]) == 0:
                            # Swap e for candidates[i], candidates[j]
                            new_sel = without_e + [candidates[i], candidates[j]]
                            new_mask = mask_wo | (1 << candidates[i]) | (1 << candidates[j])
                            # Verify all pairwise non-conflict
                            ok = True
                            for k in range(len(new_sel)):
                                for l in range(k + 1, len(new_sel)):
                                    if conflict[new_sel[k]] & (1 << new_sel[l]):
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                selected = new_sel
                                sel_set = set(selected)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return selected

    def greedy_set_cover(edges, conflict, rng, time_budget=5.0):
        """Iteratively extract induced matchings to cover all edges (set-cover style)."""
        m = len(edges)
        start = time.time()
        uncovered_mask = (1 << m) - 1
        cover = []  # list of induced matchings (each is a list of edge indices)
        while uncovered_mask and time.time() - start < time_budget:
            # Find a large induced matching among uncovered edges
            uncovered_list = [i for i in range(m) if (uncovered_mask >> i) & 1]
            if not uncovered_list:
                break
            # Greedy induced matching on uncovered
            order = uncovered_list[:]
            rng.shuffle(order)
            selected = []
            sel_mask = 0
            for e in order:
                if conflict[e] & sel_mask == 0:
                    selected.append(e)
                    sel_mask |= 1 << e
            if not selected:
                # Shouldn't happen, but safety
                selected = [uncovered_list[0]]
            cover.append(selected)
            for e in selected:
                uncovered_mask &= ~(1 << e)
        # Check completeness
        if uncovered_mask:
            # Force add remaining
            for i in range(m):
                if (uncovered_mask >> i) & 1:
                    cover.append([i])
        return cover

    def local_refinement_cover(edges, conflict, cover, rng, time_budget=10.0):
        """Local refinement: try to merge cover classes, move edges between classes."""
        m = len(edges)
        start = time.time()
        # Build colour array from cover
        colour = [-1] * m
        for c, cls in enumerate(cover):
            for e in cls:
                colour[e] = c
        # Build class masks
        def build_masks(col):
            masks = {}
            for e in range(m):
                c = col[e]
                if c not in masks:
                    masks[c] = 0
                masks[c] |= 1 << e
            return masks

        masks = build_masks(colour)

        for iteration in range(10000):
            if time.time() - start > time_budget:
                break
            improved = False
            maxc = max(colour) if any(c >= 0 for c in colour) else -1
            if maxc < 1:
                break

            # Try merging: for each pair (a, b) with a > b, check if class a can merge into b
            merged = False
            for a in range(maxc, -1, -1):
                if a not in masks or not masks[a]:
                    continue
                for b in range(a):
                    if b not in masks or not masks[b]:
                        continue
                    # Check if all edges in class a are compatible with class b
                    ok = True
                    for e in range(m):
                        if colour[e] == a:
                            if conflict[e] & masks[b]:
                                ok = False
                                break
                    if ok:
                        for e in range(m):
                            if colour[e] == a:
                                colour[e] = b
                        masks[b] |= masks[a]
                        del masks[a]
                        merged = True
                        improved = True
                        break
                if merged:
                    break

            if improved:
                continue

            # Try moving a single edge from a higher colour to a lower one
            moved = False
            for _ in range(50):
                e = rng.randrange(m)
                ce = colour[e]
                if ce <= 0:
                    continue
                for d in range(ce):
                    if d not in masks or not masks[d]:
                        continue
                    if conflict[e] & masks[d] == 0:
                        colour[e] = d
                        masks[d] |= 1 << e
                        masks[ce] &= ~(1 << e)
                        if not masks[ce]:
                            del masks[ce]
                        moved = True
                        improved = True
                        break
                if moved:
                    break

            if improved:
                continue

            # Try 1-swap
            if rng.random() < 0.4:
                swapped = False
                for _ in range(30):
                    e = rng.randrange(m)
                    ce = colour[e]
                    if ce <= 0:
                        continue
                    lower = [d for d in range(ce) if d in masks and masks[d]]
                    if not lower:
                        continue
                    d = rng.choice(lower)
                    # Pick f from class d
                    f_candidates = [f for f in range(m) if colour[f] == d]
                    if not f_candidates:
                        continue
                    f = rng.choice(f_candidates)
                    mask_d_wo_f = masks[d] & ~(1 << f)
                    mask_ce_wo_e = masks[ce] & ~(1 << e)
                    if conflict[e] & mask_d_wo_f == 0 and conflict[f] & mask_ce_wo_e == 0:
                        colour[e] = d
                        colour[f] = ce
                        masks[d] = (masks[d] & ~(1 << f)) | (1 << e)
                        masks[ce] = (masks[ce] & ~(1 << e)) | (1 << f)
                        swapped = True
                        improved = True
                        break
                if swapped:
                    continue

            if not improved:
                # Try 2-swap
                two_swapped = False
                for _ in range(20):
                    a = rng.randrange(maxc + 1)
                    if a not in masks or not masks[a]:
                        continue
                    b = rng.randrange(a) if a > 0 else 0
                    if b == a or b not in masks or not masks[b]:
                        continue
                    e_candidates = [e for e in range(m) if colour[e] == a]
                    f_candidates = [f for f in range(m) if colour[f] == b]
                    if not e_candidates or not f_candidates:
                        continue
                    e = rng.choice(e_candidates)
                    f = rng.choice(f_candidates)
                    mask_a_wo_e = masks[a] & ~(1 << e)
                    mask_b_wo_f = masks[b] & ~(1 << f)
                    if conflict[e] & mask_b_wo_f == 0 and conflict[f] & mask_a_wo_e == 0:
                        colour[e] = b
                        colour[f] = a
                        masks[b] = (masks[b] & ~(1 << f)) | (1 << e)
                        masks[a] = (masks[a] & ~(1 << e)) | (1 << f)
                        two_swapped = True
                        improved = True
                        break
                if not improved:
                    break

        # Renumber compactly
        present = sorted(set(c for c in colour if c >= 0))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            if colour[e] >= 0:
                colour[e] = remap[colour[e]]
        return colour, len(present)

    def simulated_annealing_cover(edges, conflict, colour, used, rng, time_budget=5.0):
        """Simulated annealing on the cover representation."""
        m = len(edges)
        start = time.time()
        best_colour = colour[:]
        best_used = used
        current_colour = colour[:]
        current_used = used

        def build_masks(col):
            masks = {}
            for e in range(m):
                c = col[e]
                if c not in masks:
                    masks[c] = 0
                masks[c] |= 1 << e
            return masks

        masks = build_masks(current_colour)
        T = 10.0
        T_min = 0.1

        for step in range(50000):
            if time.time() - start > time_budget:
                break
            T = max(T_min, T * 0.9995)

            # Random move
            move_type = rng.random()
            if move_type < 0.3:
                # Move edge to random colour
                e = rng.randrange(m)
                new_c = rng.randrange(current_used)
                if new_c == current_colour[e]:
                    continue
                old_c = current_colour[e]
                mask_new = masks.get(new_c, 0)
                if conflict[e] & mask_new:
                    continue
                # Check new colour class remains valid (it does since we only add)
                # But we need to check that removing e from old_c doesn't break anything (it doesn't)
                current_colour[e] = new_c
                masks[new_c] = mask_new | (1 << e)
                masks[old_c] &= ~(1 << e)
                if not masks[old_c]:
                    del masks[old_c]
                new_used = max(masks.keys()) + 1 if masks else 0
                delta = new_used - current_used
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_used = new_used
                else:
                    # Revert
                    current_colour[e] = old_c
                    masks[old_c] |= (1 << e)
                    masks[new_c] = mask_new
                    if not masks[old_c]:
                        del masks[old_c]
            elif move_type < 0.6:
                # Swap two edges
                e = rng.randrange(m)
                f = rng.randrange(m)
                if e == f:
                    continue
                ce = current_colour[e]
                cf = current_colour[f]
                if ce == cf:
                    continue
                mask_ce_wo_e = masks[ce] & ~(1 << e)
                mask_cf_wo_f = masks[cf] & ~(1 << f)
                if conflict[e] & mask_cf_wo_f or conflict[f] & mask_ce_wo_e:
                    continue
                current_colour[e] = cf
                current_colour[f] = ce
                masks[cf] = (masks[cf] & ~(1 << f)) | (1 << e)
                masks[ce] = (masks[ce] & ~(1 << e)) | (1 << f)
                if not masks[cf]:
                    del masks[cf]
                if not masks[ce]:
                    del masks[ce]
                new_used = max(masks.keys()) + 1 if masks else 0
                delta = new_used - current_used
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current_used = new_used
                else:
                    current_colour[e] = ce
                    current_colour[f] = cf
                    masks[ce] = (masks[ce] & ~(1 << f)) | (1 << e)
                    masks[cf] = (masks[cf] & ~(1 << e)) | (1 << f)
                    if not masks[ce]:
                        del masks[ce]
                    if not masks[cf]:
                        del masks[cf]
            else:
                # Try to merge two classes
                if current_used < 2:
                    continue
                a = rng.randrange(current_used)
                b = rng.randrange(current_used)
                if a == b or a not in masks or b not in masks:
                    continue
                # Check if a can merge into b
                ok = True
                for e in range(m):
                    if current_colour[e] == a:
                        if conflict[e] & masks[b]:
                            ok = False
                            break
                if not ok:
                    continue
                for e in range(m):
                    if current_colour[e] == a:
                        current_colour[e] = b
                masks[b] |= masks[a]
                del masks[a]
                new_used = max(masks.keys()) + 1 if masks else 0
                if new_used < current_used:
                    current_used = new_used

            # Update best
            if current_used < best_used:
                best_used = current_used
                best_colour = current_colour[:]

        return best_colour, best_used

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)

        # Phase 1: Multiple set-cover constructions
        n_restarts = 50
        for r in range(n_restarts):
            if time.time() > DEADLINE - 20.0:
                break
            cover = greedy_set_cover(edges, conflict, rng, time_budget=2.0)
            # Convert to colour array
            colour = [-1] * m
            for c, cls in enumerate(cover):
                for e in cls:
                    colour[e] = c
            used = len(cover)
            # Local refinement
            colour, used = local_refinement_cover(edges, conflict, cover, rng, time_budget=5.0)
            if used < best_used:
                best_colour = colour[:]
                best_used = used

        # Phase 2: Simulated annealing on best
        if best_colour is not None and time.time() < DEADLINE - 15.0:
            best_colour, best_used = simulated_annealing_cover(
                edges, conflict, best_colour, best_used, rng, time_budget=15.0
            )

        # Phase 3: More targeted restarts
        if time.time() < DEADLINE - 10.0:
            for r in range(30):
                if time.time() > DEADLINE - 5.0:
                    break
                cover = greedy_set_cover(edges, conflict, rng, time_budget=1.0)
                colour = [-1] * m
                for c, cls in enumerate(cover):
                    for e in cls:
                        colour[e] = c
                used = len(cover)
                colour, used = local_refinement_cover(edges, conflict, cover, rng, time_budget=2.0)
                if used <= best_used:
                    colour, used = simulated_annealing_cover(
                        edges, conflict, colour, used, rng, time_budget=3.0
                    )
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used

        if best_colour is None:
            # Fallback: simple greedy
            order = list(range(m))
            colour = [-1] * m
            used = 0
            col_masks = []
            for e in order:
                ce = 1 << e
                placed = False
                for c in range(used):
                    if col_masks[c] & conflict[e] == 0:
                        colour[e] = c
                        col_masks[c] |= ce
                        placed = True
                        break
                if not placed:
                    colour[e] = used
                    col_masks.append(ce)
                    used += 1
            best_colour = colour
            best_used = used

        out.append(best_colour)

    return out

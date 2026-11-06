"""Reciprocal weight of three-progression-free sets on [1,256].
Optimized recursive B&B with efficient propagation and bound computation.
"""
import time
from fractions import Fraction

_N = 256
_DEADLINE_MARGIN = 2.0

def construct():
    t_start = time.time()
    deadline = t_start + 90.0 - _DEADLINE_MARGIN

    # Precompute APs
    ap_elems = []
    for a in range(1, _N + 1):
        for c in range(a + 2, _N + 1, 2):
            b = (a + c) // 2
            if a < b < c:
                ap_elems.append((a - 1, b - 1, c - 1))

    n_aps = len(ap_elems)
    weights = [Fraction(0)] + [Fraction(1, x) for x in range(1, _N + 1)]

    # AP masks for each AP
    ap_masks = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in ap_elems]

    # For each element, list of APs it belongs to
    elem_aps = [[] for _ in range(_N)]
    for idx, (a, b, c) in enumerate(ap_elems):
        elem_aps[a].append(idx)
        elem_aps[b].append(idx)
        elem_aps[c].append(idx)

    all_mask = (1 << _N) - 1

    # Sort elements by weight descending for branching
    elem_order = list(range(_N))
    elem_order.sort(key=lambda i: weights[i + 1], reverse=True)

    best_weight = Fraction(0)
    best_set = []

    def propagate(in_mask, out_mask):
        """If 2 of 3 in an AP are in, force the 3rd out."""
        changed = True
        while changed:
            changed = False
            for idx in range(n_aps):
                a, b, c = ap_elems[idx]
                am, bm, cm = 1 << a, 1 << b, 1 << c
                a_in = (in_mask & am) != 0
                b_in = (in_mask & bm) != 0
                c_in = (in_mask & cm) != 0
                if a_in and b_in and (out_mask & cm) == 0:
                    out_mask |= cm
                    changed = True
                elif a_in and c_in and (out_mask & bm) == 0:
                    out_mask |= bm
                    changed = True
                elif b_in and c_in and (out_mask & am) == 0:
                    out_mask |= am
                    changed = True
        return in_mask, out_mask

    def get_bound(in_mask, out_mask, current_weight):
        """Upper bound: current + sum of free weights, minus penalties for tight APs."""
        free_mask = all_mask & ~in_mask & ~out_mask
        if free_mask == 0:
            return current_weight

        bound = current_weight
        # Add all free weights
        for i in range(_N):
            if (free_mask >> i) & 1:
                bound += weights[i + 1]

        # Subtract penalties: for each AP with exactly 1 in and 2 free,
        # at most 1 of the 2 free can be added, so subtract min of their weights.
        # For each AP with 0 in and 3 free, at most 2 of 3 can be added,
        # so subtract min of the 3 weights.
        for idx in range(n_aps):
            a, b, c = ap_elems[idx]
            am, bm, cm = 1 << a, 1 << b, 1 << c
            in_count = ((in_mask & am) != 0) + ((in_mask & bm) != 0) + ((in_mask & cm) != 0)
            free_count = ((free_mask & am) != 0) + ((free_mask & bm) != 0) + ((free_mask & cm) != 0)
            if in_count == 1 and free_count == 2:
                # Find the two free elements
                free_elems = []
                if (free_mask & am) != 0:
                    free_elems.append(a)
                if (free_mask & bm) != 0:
                    free_elems.append(b)
                if (free_mask & cm) != 0:
                    free_elems.append(c)
                if len(free_elems) == 2:
                    bound -= min(weights[free_elems[0] + 1], weights[free_elems[1] + 1])
            elif in_count == 0 and free_count == 3:
                bound -= min(weights[a + 1], weights[b + 1], weights[c + 1])

        return bound

    def branch_and_bound(in_mask, out_mask, current_weight):
        nonlocal best_weight, best_set

        if time.time() > deadline:
            return

        in_mask, out_mask = propagate(in_mask, out_mask)
        free_mask = all_mask & ~in_mask & ~out_mask
        if free_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = [i + 1 for i in range(_N) if (in_mask >> i) & 1]
            return

        bound = get_bound(in_mask, out_mask, current_weight)
        if bound <= best_weight:
            return

        # Pick highest-weight free element
        best_i = -1
        for i in elem_order:
            if (free_mask >> i) & 1:
                best_i = i
                break
        if best_i < 0:
            return

        # Branch: include
        branch_and_bound(in_mask | (1 << best_i), out_mask, current_weight + weights[best_i + 1])
        if time.time() > deadline:
            return
        # Branch: exclude
        branch_and_bound(in_mask, out_mask | (1 << best_i), current_weight)

    branch_and_bound(0, 0, Fraction(0))

    if not best_set:
        best_set = [1]

    return best_set

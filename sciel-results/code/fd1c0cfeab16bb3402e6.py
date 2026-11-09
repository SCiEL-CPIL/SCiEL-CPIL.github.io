import time
import numpy as np

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# Precompute AP indices as lists for faster access
_ALL_APS_LIST = [[tuple(ap)] for ap in _ALL_APS]

def construct():
    start = time.time()
    DEADLINE = 75.0  # safety margin before 90s

    n = 128
    # Precompute for each position, which progressions it participates in (as a member)
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)

    # Precompute degree (number of APs each position is in)
    degree = [len(pos_in_aps[i]) for i in range(n)]

    best = None
    best_ncolors = n  # start with worst

    # Try increasing number of colors
    for k_colors in range(3, 7):
        if time.time() > DEADLINE:
            break
        # Backtracking search with MRV and least-constraining value heuristic
        colors = [-1] * n
        unassigned_set = set(range(n))

        def get_allowed_colors(pos, k):
            """Get allowed colors for pos given current assignment."""
            al = set(range(k))
            for ap in pos_in_aps[pos]:
                others = [i for i in ap if i != pos]
                oc = set()
                unassigned_others = 0
                for i in others:
                    c = colors[i]
                    if c == -1:
                        unassigned_others += 1
                    else:
                        oc.add(c)
                # For each candidate color c, check if AP can still reach >=3 distinct colors
                for c in list(al):
                    current_colors = oc | {c}
                    max_possible = len(current_colors) + unassigned_others
                    if max_possible < 3:
                        al.discard(c)
            return al

        def count_conflicts(pos, c, k):
            """Count how many APs containing pos would be in a tight state if pos=c."""
            count = 0
            for ap in pos_in_aps[pos]:
                others = [i for i in ap if i != pos]
                oc = set()
                unassigned_others = 0
                for i in others:
                    cc = colors[i]
                    if cc == -1:
                        unassigned_others += 1
                    else:
                        oc.add(cc)
                current_colors = oc | {c}
                max_possible = len(current_colors) + unassigned_others
                if max_possible == 3:
                    count += 1
            return count

        def backtrack():
            nonlocal best, best_ncolors
            if time.time() > DEADLINE:
                return False

            if not unassigned_set:
                # All assigned; verify all APs
                for ap in _ALL_APS:
                    if len({colors[i] for i in ap}) < 3:
                        return False
                used = set(c for c in colors if c != -1)
                if len(used) < best_ncolors:
                    best = colors[:]
                    best_ncolors = len(used)
                return True

            # MRV: pick unassigned position with fewest allowed colors
            min_allowed = k_colors + 1
            chosen_pos = -1
            chosen_allowed = None
            for pos in unassigned_set:
                al = get_allowed_colors(pos, k_colors)
                if len(al) < min_allowed:
                    min_allowed = len(al)
                    chosen_pos = pos
                    chosen_allowed = al
                    if len(al) == 0:
                        break
                    if len(al) == 1:
                        break

            if chosen_pos == -1 or chosen_allowed is None:
                return False
            if len(chosen_allowed) == 0:
                return False

            # LCV: order colors by least constraining (fewest conflicts)
            color_list = list(chosen_allowed)
            if len(color_list) > 1:
                color_list.sort(key=lambda c: count_conflicts(chosen_pos, c, k_colors))

            for c in color_list:
                colors[chosen_pos] = c
                unassigned_set.discard(chosen_pos)

                # Forward checking: ensure no unassigned position has empty domain
                ok = True
                for pos in unassigned_set:
                    if len(get_allowed_colors(pos, k_colors)) == 0:
                        ok = False
                        break

                if ok:
                    if backtrack():
                        return True

                colors[chosen_pos] = -1
                unassigned_set.add(chosen_pos)

            return False

        if backtrack():
            break
        else:
            # Didn't find with k_colors within time; try more colors
            pass

    if best is not None:
        # Ensure all are valid non-negative and within 0..127
        return best
    else:
        # Fallback: use a known valid coloring pattern
        # Pattern: color[i] = i % 4 gives 4 colors
        # Verify this works for 4-term progressions
        colors = [i % 4 for i in range(128)]
        # Check if it's valid
        valid = True
        for ap in _ALL_APS:
            if len({colors[i] for i in ap}) < 3:
                valid = False
                break
        if valid:
            return colors
        else:
            # Use unique colors as ultimate fallback (always valid)
            return list(range(128))

# Quick self-check during module load is not allowed (must be fast); skip.

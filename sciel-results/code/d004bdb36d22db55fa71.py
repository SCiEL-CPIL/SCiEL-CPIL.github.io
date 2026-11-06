import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    n = 16
    num_edges = 120

    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1

    for a, b, c, d in combinations(range(n), 4):
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)

    num_k4s = len(k4s)

    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)

    # For each K4, track color counts to quickly determine distinct colors
    # k4_color_count[k4_idx][color] = number of edges in K4 with that color
    # k4_distinct[k4_idx] = number of distinct colors in K4

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        """Check if every K4 has at least 5 distinct colors."""
        for edge_idxs in k4s:
            if len(set(colors[ei] for ei in edge_idxs)) < 5:
                return False
        return True

    # Try structured initial colorings
    candidates = []

    # Additive: (i + j) % p
    for p in [17, 19, 23, 29, 31]:
        colors = [(i + j) % p for i in range(n) for j in range(i + 1, n)]
        candidates.append(colors)

    # Multiplicative: (i * j) % p
    for p in [17, 19, 23, 29, 31]:
        colors = [(i * j) % p for i in range(n) for j in range(i + 1, n)]
        candidates.append(colors)

    # Mixed linear: (i*a + j*b) % p
    for p in [17, 19, 23]:
        for a in range(1, p):
            for b in range(1, p):
                colors = [(i * a + j * b) % p for i in range(n) for j in range(i + 1, n)]
                candidates.append(colors)

    # Find best valid initial coloring
    best_init_colors = None
    best_init_distinct = float('inf')
    for colors in candidates:
        if is_valid(colors):
            d = count_distinct(colors)
            if d < best_init_distinct:
                best_init_distinct = d
                best_init_colors = colors[:]

    if best_init_colors is None:
        # Fallback
        colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    else:
        colors = best_init_colors

    # Build incremental tracking structures
    # For each K4, track: dict color -> count, and distinct count
    k4_color_counts = [{} for _ in range(num_k4s)]
    k4_distinct = [0] * num_k4s

    for k4_idx, edge_idxs in enumerate(k4s):
        cc = k4_color_counts[k4_idx]
        for ei in edge_idxs:
            c = colors[ei]
            cc[c] = cc.get(c, 0) + 1
        k4_distinct[k4_idx] = len(cc)

    def get_violations():
        """Return number of K4s with < 5 distinct colors."""
        return sum(1 for d in k4_distinct if d < 5)

    current_violations = get_violations()
    current_distinct = count_distinct(colors)

    best_distinct = current_distinct
    best_colors = colors[:]

    random.seed(42)

    # Local search: try to reduce distinct colors by remapping edges
    # Strategy: pick a color with fewest edges, try to remap all its edges to other colors
    # while maintaining validity

    while time.time() < deadline:
        used_colors = sorted(set(colors))
        if len(used_colors) <= best_distinct:
            break

        # Color frequency
        color_counts = {}
        for c in used_colors:
            color_counts[c] = 0
        for c in colors:
            color_counts[c] += 1

        # Target: color with fewest edges (easiest to eliminate)
        target_color = min(used_colors, key=lambda c: color_counts[c])
        edges_with_target = [i for i in range(num_edges) if colors[i] == target_color]

        if not edges_with_target:
            continue

        # Try to remap each edge with target_color to another color
        improved = False
        for edge_idx in edges_with_target:
            if time.time() > deadline:
                break

            # Try each other color
            for new_color in used_colors:
                if new_color == target_color:
                    continue

                old_color = colors[edge_idx]
                colors[edge_idx] = new_color

                # Update K4 tracking incrementally
                for k4_idx in edge_to_k4s[edge_idx]:
                    cc = k4_color_counts[k4_idx]
                    # Remove old color
                    cc[old_color] -= 1
                    if cc[old_color] == 0:
                        del cc[old_color]
                        k4_distinct[k4_idx] -= 1
                    # Add new color
                    cc[new_color] = cc.get(new_color, 0) + 1
                    if cc[new_color] == 1:
                        k4_distinct[k4_idx] += 1

                # Check if any K4 now has < 5 colors
                valid = all(k4_distinct[k] >= 5 for k in edge_to_k4s[edge_idx])

                if valid:
                    # Check if target_color is fully eliminated
                    if target_color not in colors:
                        new_distinct = current_distinct - 1
                        if new_distinct < best_distinct:
                            best_distinct = new_distinct
                            best_colors = colors[:]
                        current_distinct = new_distinct
                        improved = True
                        break
                    else:
                        # Still valid but didn't eliminate; keep the change
                        # Actually, let's keep it since it doesn't hurt
                        improved = True
                        break
                else:
                    # Revert
                    colors[edge_idx] = old_color
                    for k4_idx in edge_to_k4s[edge_idx]:
                        cc = k4_color_counts[k4_idx]
                        cc[new_color] -= 1
                        if cc[new_color] == 0:
                            del cc[new_color]
                            k4_distinct[k4_idx] -= 1
                        cc[old_color] = cc.get(old_color, 0) + 1
                        if cc[old_color] == 1:
                            k4_distinct[k4_idx] += 1

            if improved:
                break

        if not improved:
            # Try random perturbation to escape local optima
            for _ in range(5):
                if time.time() > deadline:
                    break
                edge_idx = random.randint(0, num_edges - 1)
                old_color = colors[edge_idx]
                new_color = random.choice(used_colors)
                if new_color == old_color:
                    continue

                colors[edge_idx] = new_color

                # Update tracking
                for k4_idx in edge_to_k4s[edge_idx]:
                    cc = k4_color_counts[k4_idx]
                    cc[old_color] -= 1
                    if cc[old_color] == 0:
                        del cc[old_color]
                        k4_distinct[k4_idx] -= 1
                    cc[new_color] = cc.get(new_color, 0) + 1
                    if cc[new_color] == 1:
                        k4_distinct[k4_idx] += 1

                # Check validity
                valid = all(k4_distinct[k] >= 5 for k in edge_to_k4s[edge_idx])
                if not valid:
                    # Revert
                    colors[edge_idx] = old_color
                    for k4_idx in edge_to_k4s[edge_idx]:
                        cc = k4_color_counts[k4_idx]
                        cc[new_color] -= 1
                        if cc[new_color] == 0:
                            del cc[new_color]
                            k4_distinct[k4_idx] -= 1
                        cc[old_color] = cc.get(old_color, 0) + 1
                        if cc[old_color] == 1:
                            k4_distinct[k4_idx] += 1

    # Final validation
    if not is_valid(best_colors):
        # Fallback to known valid
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    return best_colors

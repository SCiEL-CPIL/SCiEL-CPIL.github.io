import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles and their edge indices as numpy arrays for speed
    cycles_list = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles_list.append([edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]])
        # Cycle 2: a-b-d-c-a
        cycles_list.append([edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]])
        # Cycle 3: a-c-b-d-a
        cycles_list.append([edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]])

    # Convert to numpy array for vectorized operations
    cycles_arr = np.array(cycles_list, dtype=np.int32)
    num_cycles = len(cycles_arr)

    # Map edge index to list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles_list):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # Convert edge_to_cycles to numpy arrays for speed
    edge_to_cycles_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_cycles]

    def is_valid_change(colors_np, edge, new_color):
        """Check if changing edge to new_color maintains validity."""
        # Get all cycles containing this edge
        cyc_indices = edge_to_cycles_arr[edge]
        if len(cyc_indices) == 0:
            return True

        # Get the cycle edges
        cyc_edges = cycles_arr[cyc_indices]  # shape (k, 4)

        # Get current colors of those edges
        cur_colors = colors_np[cyc_edges]  # shape (k, 4)

        # Replace the edge being changed with new_color
        mask = (cyc_edges == edge)
        cur_colors[mask] = new_color

        # Check for 4 distinct colors in any cycle
        # A cycle has 4 distinct colors iff all 6 pairwise comparisons are unequal
        # col0!=col1, col0!=col2, col0!=col3, col1!=col2, col1!=col3, col2!=col3
        c0 = cur_colors[:, 0]
        c1 = cur_colors[:, 1]
        c2 = cur_colors[:, 2]
        c3 = cur_colors[:, 3]

        rainbow = (c0 != c1) & (c0 != c2) & (c0 != c3) & \
                  (c1 != c2) & (c1 != c3) & \
                  (c2 != c3)

        return not np.any(rainbow)

    def count_colors(colors_np):
        return len(set(colors_np.tolist()))

    def get_modular_seed():
        """Generate a highly symmetric modular coloring."""
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                idx = edge_idx[i][j]
                # Use a combination that maximizes distinctness
                colors[idx] = (i * j + 3*i + 7*j) % 19
        return colors

    def get_seed_colors():
        """Generate recursive partition seed."""
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    def local_search(initial_colors, time_limit, seed):
        """Run local search from initial_colors, return best found."""
        rng = random.Random(seed)
        colors_np = np.array(initial_colors, dtype=np.int32)
        freq = [0] * MAX_COLORS
        for c in initial_colors:
            freq[c] += 1

        best_score = count_colors(colors_np)
        best_colors = colors_np.copy()

        steps = 0
        while time.time() < time_limit:
            steps += 1
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = int(colors_np[e])

            # Strategy: Try to introduce new colors or rare colors
            # With 50% chance, try a completely new color
            if rng.random() < 0.5:
                # Pick a color not currently used, if possible
                unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                if unused:
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, MAX_COLORS - 1)
            else:
                # Pick a random color
                new_color = rng.randint(0, MAX_COLORS - 1)

            if new_color == old_color:
                continue

            if is_valid_change(colors_np, e, new_color):
                # Update freq
                freq[old_color] -= 1
                freq[new_color] += 1
                colors_np[e] = new_color

                score = count_colors(colors_np)

                if score > best_score:
                    best_score = score
                    best_colors = colors_np.copy()
            # Don't revert on invalid or non-improving moves to allow exploration
            # But do revert if score decreased significantly to avoid drift
            # Actually, keep the move even if score drops to escape local optima

        return best_colors.tolist(), best_score

    # Try multiple strategies
    random.seed(42)

    # Strategy 1: Modular seed + local search
    mod_seed = get_modular_seed()
    mod_colors, mod_score = local_search(mod_seed, start_time + 30.0, 42)

    # Strategy 2: Recursive partition seed + local search
    part_seed = get_seed_colors()
    part_colors, part_score = local_search(part_seed, start_time + 60.0, 123)

    # Strategy 3: Combined - start from better of the two, continue search
    if mod_score >= part_score:
        current_colors = mod_colors
        current_score = mod_score
    else:
        current_colors = part_colors
        current_score = part_score

    # Final polish: continue local search from best
    best_colors, final_score = local_search(current_colors, deadline, 999)

    return best_colors

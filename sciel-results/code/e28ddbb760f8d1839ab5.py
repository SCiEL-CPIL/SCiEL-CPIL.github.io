import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 57.0

    # Precompute all 4-cycles as arrays of edge indices
    # Each cycle is 4 edges that form a 4-cycle in K16
    all_cycles = []
    vertices = list(range(N))
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on 4 vertices:
        # 1: a-b, b-c, c-d, d-a
        all_cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        # 2: a-b, b-d, d-c, c-a
        all_cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
        # 3: a-c, c-b, b-d, d-a
        all_cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))

    all_cycles = np.array(all_cycles, dtype=np.int32)
    num_cycles = len(all_cycles)  # C(16,4)*3 = 1820*3 = 5460

    # For each edge, precompute which cycles contain it and at what position
    # This allows fast local validity checking
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for pos in range(4):
            e = all_cycles[ci, pos]
            edge_to_cycles[e].append((ci, pos))

    # Convert to numpy arrays for speed
    edge_cycle_indices = []
    edge_cycle_positions = []
    for e in range(NUM_EDGES):
        if edge_to_cycles[e]:
            cys = np.array([x[0] for x in edge_to_cycles[e]], dtype=np.int32)
            poss = np.array([x[1] for x in edge_to_cycles[e]], dtype=np.int32)
        else:
            cys = np.array([], dtype=np.int32)
            poss = np.array([], dtype=np.int32)
        edge_cycle_indices.append(cys)
        edge_cycle_positions.append(poss)

    def count_violations(colors):
        """Count number of 4-cycles with all 4 edges having distinct colors."""
        # colors: shape (120,)
        # all_cycles: shape (5460, 4)
        # Get colors for each edge in each cycle: shape (5460, 4)
        cycle_colors = colors[all_cycles]  # (5460, 4)
        c0 = cycle_colors[:, 0]
        c1 = cycle_colors[:, 1]
        c2 = cycle_colors[:, 2]
        c3 = cycle_colors[:, 3]
        # A cycle is a violation if all 4 are distinct
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return int(np.sum(distinct))

    def is_valid_move(edge_idx, new_color, colors):
        """Check if setting colors[edge_idx] = new_color keeps the coloring valid."""
        old_color = colors[edge_idx]
        if new_color == old_color:
            return True

        cys = edge_cycle_indices[edge_idx]
        poss = edge_cycle_positions[edge_idx]
        if len(cys) == 0:
            return True

        # Get the cycles this edge participates in
        cycle_edges = all_cycles[cys]  # (n_cycles, 4)
        # Get current colors of all edges in these cycles
        cycle_colors = colors[cycle_edges]  # (n_cycles, 4)

        # For each cycle, check if the other 3 edges already have 3 distinct colors
        # AND new_color differs from all 3
        # The edge is at position poss[i] in cycle i
        # We need to check: are the 3 other edges all distinct from each other AND from new_color?

        # Build a mask of the other 3 positions
        # For each cycle, the other 3 positions are [0,1,2,3] minus poss[i]
        # Create a 4-element mask per cycle
        n_cyc = len(cys)
        # positions matrix: (n_cyc, 4) where 1 means "this is the edge we're changing"
        pos_mask = np.zeros((n_cyc, 4), dtype=np.int32)
        for i in range(n_cyc):
            pos_mask[i, poss[i]] = 1

        # Get the 3 other colors for each cycle
        # other_colors[i, j] = color of the j-th other edge in cycle i
        # Simpler: just check directly
        violations = 0
        for i in range(n_cyc):
            p = poss[i]
            # Get the 3 other edge colors
            other = cycle_colors[i]
            other[0] = -1  # placeholder, we'll use indices
            # Actually, let's just extract the 3 other positions
            oth_pos = [x for x in range(4) if x != p]
            c_a = cycle_colors[i, oth_pos[0]]
            c_b = cycle_colors[i, oth_pos[1]]
            c_c = cycle_colors[i, oth_pos[2]]

            # Check if new_color, c_a, c_b, c_c are all distinct
            if new_color == c_a or new_color == c_b or new_color == c_c:
                continue
            if c_a == c_b or c_a == c_c or c_b == c_c:
                continue
            return False
        return True

    def is_move_valid_fast(edge_idx, new_color, colors):
        """Faster vectorized check for a single move."""
        old_color = colors[edge_idx]
        if new_color == old_color:
            return True

        cys = edge_cycle_indices[edge_idx]
        poss = edge_cycle_positions[edge_idx]
        n_cyc = len(cys)
        if n_cyc == 0:
            return True

        # Get all edges in these cycles
        cycle_edges = all_cycles[cys]  # (n_cyc, 4)
        # Get their colors
        cyc_cols = colors[cycle_edges]  # (n_cyc, 4)

        # For each cycle, the edge at position poss[i] is the one being changed
        # We need to check the other 3 edges
        # Create indices for the other 3 positions
        # other_pos[i] = the 3 positions in cycle i that are not poss[i]
        # We can use: for each cycle, mask out poss[i]
        # Build a (n_cyc, 3) array of the other positions
        other_pos = np.empty((n_cyc, 3), dtype=np.int32)
        for i in range(n_cyc):
            p = poss[i]
            others = [x for x in range(4) if x != p]
            other_pos[i] = others

        # Get colors of other edges: (n_cyc, 3)
        # Use advanced indexing
        row_idx = np.arange(n_cyc)[:, None]  # (n_cyc, 1)
        col_idx = other_pos  # (n_cyc, 3)
        other_colors = cyc_cols[row_idx, col_idx]  # (n_cyc, 3)

        # Check: new_color not in other_colors, and other_colors all distinct
        # new_color != other_colors for all
        new_differs = (other_colors != new_color).all(axis=1)  # (n_cyc,)
        # other 3 colors all distinct
        oc0 = other_colors[:, 0]
        oc1 = other_colors[:, 1]
        oc2 = other_colors[:, 2]
        others_distinct = (oc0 != oc1) & (oc0 != oc2) & (oc1 != oc2)  # (n_cyc,)

        # Violation if new_differs AND others_distinct
        if np.any(new_differs & others_distinct):
            return False
        return True

    def num_distinct_colors(colors):
        return len(set(colors.tolist()))

    # --- Simulated Annealing ---
    rng = np.random.default_rng(42)

    def run_sa(initial_colors, max_time):
        """Run simulated annealing starting from initial_colors."""
        colors = initial_colors.copy()
        current_violations = count_violations(colors)
        current_distinct = num_distinct_colors(colors)
        best_colors = colors.copy()
        best_distinct = current_distinct
        best_violations = current_violations

        # SA parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.99997  # Slow cooling for many iterations

        steps = 0
        while time.time() < max_time:
            # Pick a random edge
            e = rng.integers(0, NUM_EDGES)
            old_c = colors[e]

            # Pick a new color
            # Bias towards colors not currently used to encourage new colors
            if rng.random() < 0.3:
                # Try a random color (possibly new)
                new_c = int(rng.integers(0, MAX_COLORS))
            else:
                # Try a color from a small set of existing + some new
                existing = list(set(colors.tolist()))
                if len(existing) < MAX_COLORS:
                    # Mix: 50% existing, 50% random new
                    if rng.random() < 0.5:
                        new_c = int(rng.choice(existing))
                    else:
                        new_c = int(rng.integers(0, MAX_COLORS))
                else:
                    new_c = int(rng.choice(existing))

            if new_c == old_c:
                continue

            # Check validity of the move
            if not is_move_valid_fast(e, new_c, colors):
                continue

            # Move is valid. Apply it.
            colors[e] = new_c
            new_distinct = num_distinct_colors(colors)
            # Since move is valid, violations = 0
            # Actually, we need to verify: the move is valid means it doesn't CREATE a violation.
            # But it might fix one? No, we're going from a valid state, so it stays valid.
            # Wait, the initial state might not be valid. Let me handle that.

            # Actually, let's just track violations properly.
            # For speed, since we check validity of each move, if we start valid, we stay valid.
            # Let's ensure we start from a valid state.

            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = colors.copy()

            # For SA, we need an energy function. Since we're constrained to valid moves,
            # the "energy" is just -num_distinct. We always accept improvements.
            # For exploration, we can occasionally accept moves that don't improve.
            # But since all valid moves maintain validity, and we want to maximize distinct,
            # let's just do a greedy hill-climb with random restarts.
            # Actually, let's do proper SA where we can temporarily reduce distinct count.
            # Energy = -distinct_count. We accept worse moves with probability exp(-delta/T).

            # The move always keeps us valid (we checked), so violations stay at 0.
            # The only change is in distinct count.
            delta = new_distinct - current_distinct
            if delta < 0:
                # Worse move: accept with probability exp(delta/T)
                if rng.random() < np.exp(delta / T):
                    current_distinct = new_distinct
                else:
                    # Reject: revert
                    colors[e] = old_c
            else:
                current_distinct = new_distinct

            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min

            steps += 1

        return best_colors, best_distinct

    # --- Initial valid coloring: all zeros ---
    # This is trivially valid (no 4 distinct colors possible)
    base_colors = np.zeros(NUM_EDGES, dtype=np.int32)

    best_overall = base_colors.copy()
    best_overall_distinct = 1

    # Run multiple SA restarts with different initial perturbations
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Create a perturbed initial state
        # Start from base, then randomly assign some edges to random colors
        # while maintaining validity
        init = base_colors.copy()
        num_perturb = rng.integers(5, 30)
        for _ in range(num_perturb):
            e = int(rng.integers(0, NUM_EDGES))
            c = int(rng.integers(0, MAX_COLORS))
            if is_move_valid_fast(e, c, init):
                init[e] = c

        # Run SA
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        sa_time = min(remaining, 2.0)  # 2 seconds per restart
        result, distinct = run_sa(init, start_time + time.time() - start_time + sa_time)
        # Fix the time calculation
        # Actually, let's just pass the deadline
        # Let me restructure

    # Let me restructure the time management
    # Actually, the above has a bug. Let me redo it cleanly.

    # Restart: clean implementation
    best_overall = base_colors.copy()
    best_overall_distinct = 1

    for r in range(100):
        if time.time() > deadline:
            break

        # Create perturbed initial state
        init = base_colors.copy()
        num_perturb = int(rng.integers(5, 40))
        for _ in range(num_perturb):
            e = int(rng.integers(0, NUM_EDGES))
            c = int(rng.integers(0, MAX_COLORS))
            if is_move_valid_fast(e, c, init):
                init[e] = c

        # Run SA for a fixed number of steps or time
        sa_deadline = min(deadline, time.time() + 1.5)
        colors = init.copy()
        current_distinct = num_distinct_colors(colors)

        T = 5.0
        T_min = 0.05
        cooling = 0.9999

        while time.time() < sa_deadline:
            e = int(rng.integers(0, NUM_EDGES))
            old_c = colors[e]

            # Choose new color
            if rng.random() < 0.4:
                new_c = int(rng.integers(0, MAX_COLORS))
            else:
                existing = list(set(colors.tolist()))
                if len(existing) > 0:
                    new_c = int(rng.choice(existing))
                else:
                    new_c = 0

            if new_c == old_c:
                continue

            if not is_move_valid_fast(e, new_c, colors):
                continue

            colors[e] = new_c
            new_distinct = num_distinct_colors(colors)

            if new_distinct > best_overall_distinct:
                best_overall_distinct = new_distinct
                best_overall = colors.copy()

            delta = new_distinct - current_distinct
            if delta < 0:
                if rng.random() < np.exp(delta / T):
                    current_distinct = new_distinct
                else:
                    colors[e] = old_c
            else:
                current_distinct = new_distinct

            T *= cooling
            if T < T_min:
                T = T_min

    # Final verification
    if count_violations(best_overall) != 0:
        # Fallback to all zeros
        best_overall = np.zeros(NUM_EDGES, dtype=np.int32)

    return best_overall.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
    print(colors)

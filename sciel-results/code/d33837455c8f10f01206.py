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
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))

    # Convert to edge indices
    cycle_indices = np.array([tuple(edge_idx[u][v] for u, v in c) for c in cycles], dtype=np.int32)

    # For each edge, which cycles does it belong to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    def get_seed_colors(seed=0):
        """Generate a valid coloring via recursive partitioning with randomization."""
        colors = [0] * NUM_EDGES
        color_id = 0
        rng = random.Random(seed)
        
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            # Try different split points for variety
            splits = [len(vertices) // 2]
            if len(vertices) >= 4:
                splits.append(max(1, min(len(vertices)-1, len(vertices)//2 + rng.randint(-1, 1))))
            mid = splits[0]
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

    def check_validity(colors):
        """Check if coloring is valid (no rainbow 4-cycle)."""
        c = np.array(colors, dtype=np.int32)
        # For each cycle, check if all 4 colors are distinct
        for ci in cycle_indices:
            c0, c1, c2, c3 = c[ci[0]], c[ci[1]], c[ci[2]], c[ci[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Try multiple initial seeds and pick the best
    best_colors = None
    best_score = 0
    for s in range(10):
        if time.time() > deadline - 5.0:
            break
        colors = get_seed_colors(s)
        if check_validity(colors):
            sc = count_colors(colors)
            if sc > best_score:
                best_score = sc
                best_colors = colors.copy()

    if best_colors is None:
        best_colors = get_seed_colors(0)
        best_score = count_colors(best_colors)

    current_colors = best_colors.copy()
    current_distinct = best_score

    # Track frequency
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1

    # Local search with simulated annealing
    rng = random.Random(123)
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 50000

    while time.time() < deadline:
        # Pick edge: bias toward high-frequency colors
        # With probability 0.7, pick from high-freq edges; else random
        if rng.random() < 0.7:
            # Weighted pick: probability proportional to freq
            total_freq = sum(freq)
            if total_freq == 0:
                e = rng.randint(0, NUM_EDGES - 1)
            else:
                # Build cumulative weights for edges
                # For speed, just pick a random edge and check its freq
                e = rng.randint(0, NUM_EDGES - 1)
        else:
            e = rng.randint(0, NUM_EDGES - 1)

        old_color = current_colors[e]

        # Choose new color: bias toward unused or rare colors
        if rng.random() < 0.6:
            # Try an unused color
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, MAX_COLORS - 1)
        else:
            new_color = rng.randint(0, MAX_COLORS - 1)

        if new_color == old_color:
            continue

        # Check validity: only check cycles involving edge e
        valid = True
        for c_idx in edge_to_cycles[e]:
            ci = cycle_indices[c_idx]
            c0 = current_colors[ci[0]] if ci[0] != e else new_color
            c1 = current_colors[ci[1]] if ci[1] != e else new_color
            c2 = current_colors[ci[2]] if ci[2] != e else new_color
            c3 = current_colors[ci[3]] if ci[3] != e else new_color
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                valid = False
                break

        if valid:
            old_freq = freq[old_color]
            new_freq = freq[new_color]

            freq[old_color] -= 1
            freq[new_color] += 1

            new_distinct = current_distinct
            if old_freq == 1:
                new_distinct -= 1
            if new_freq == 0:
                new_distinct += 1

            if new_distinct >= current_distinct:
                accept = True
            else:
                delta = new_distinct - current_distinct
                accept = rng.random() < np.exp(delta / T)

            if accept:
                current_colors[e] = new_color
                current_distinct = new_distinct
                if new_distinct > best_score:
                    best_score = new_distinct
                    best_colors = current_colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_colors[e] = old_color
                freq[old_color] += 1
                freq[new_color] -= 1
                no_improve += 1
        else:
            no_improve += 1

        T *= cooling
        if T < T_min:
            T = T_min

        # Restart if stuck for too long
        if no_improve > max_no_improve:
            no_improve = 0
            T = 5.0
            # Perturb: randomly change a few edges
            for _ in range(5):
                pe = rng.randint(0, NUM_EDGES - 1)
                pc = rng.randint(0, MAX_COLORS - 1)
                if pc != current_colors[pe]:
                    # Quick validity check
                    ok = True
                    for c_idx in edge_to_cycles[pe]:
                        ci = cycle_indices[c_idx]
                        c0 = current_colors[ci[0]] if ci[0] != pe else pc
                        c1 = current_colors[ci[1]] if ci[1] != pe else pc
                        c2 = current_colors[ci[2]] if ci[2] != pe else pc
                        c3 = current_colors[ci[3]] if ci[3] != pe else pc
                        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                            ok = False
                            break
                    if ok:
                        old_c = current_colors[pe]
                        freq[old_c] -= 1
                        freq[pc] += 1
                        current_colors[pe] = pc
                        if old_c == 1:
                            current_distinct -= 1
                        if freq[pc] == 1:
                            current_distinct += 1

    return best_colors

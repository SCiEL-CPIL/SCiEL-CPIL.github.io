"""
Combined approach: Penalty-based simulated annealing with structural color
elimination. Uses a unified energy landscape (violations + color count) to
search for valid colorings with minimal colors, then applies targeted
elimination moves to reduce the palette further.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = 128

    # Precompute all 4-APs (0-indexed): (a, a+d, a+2d, a+3d) with a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def validate(coloring):
        """Check every nonconstant 4-term AP has >= 3 distinct colors."""
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def count_colors(colors):
        return len(set(colors))

    def compute_energy(coloring, k, w_color=50.0):
        """
        Compute energy: violations * 1000 + distinct_colors * w_color.
        The large violation penalty ensures validity is prioritized,
        but the color term drives minimization within valid solutions.
        """
        violations = count_violations(coloring)
        distinct = count_colors(coloring)
        return violations * 1000 + distinct * w_color

    def delta_energy(coloring, idx, new_color, k, w_color=50.0):
        """
        Compute change in energy when recoloring position idx to new_color.
        Uses incremental violation counting and tracks color count change.
        """
        old_color = coloring[idx]
        if old_color == new_color:
            return 0

        # Compute delta violations
        delta_violations = 0
        for ap_id in ap_indices[idx]:
            ap = aps[ap_id]
            c0 = coloring[ap[0]]
            c1 = coloring[ap[1]]
            c2 = coloring[ap[2]]
            c3 = coloring[ap[3]]
            
            old_set_size = len({c0, c1, c2, c3})
            
            new_c0 = new_color if ap[0] == idx else c0
            new_c1 = new_color if ap[1] == idx else c1
            new_c2 = new_color if ap[2] == idx else c2
            new_c3 = new_color if ap[3] == idx else c3
            
            new_set_size = len({new_c0, new_c1, new_c2, c3 if ap[3] != idx else new_color})
            
            if old_set_size < 3 and new_set_size >= 3:
                delta_violations -= 1
            elif old_set_size >= 3 and new_set_size < 3:
                delta_violations += 1

        # Compute delta distinct colors
        # Check if old_color disappears or new_color appears
        color_counts = {}
        for c in coloring:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        old_count = color_counts.get(old_color, 0)
        new_count = color_counts.get(new_color, 0)
        
        delta_distinct = 0
        if old_count == 1:
            # old_color disappears
            delta_distinct -= 1
        if new_count == 0:
            # new_color appears
            delta_distinct += 1

        return delta_violations * 1000 + delta_distinct * w_color

    def penalty_sa(k, time_budget=10.0, rng=None, w_color=50.0):
        """
        Penalty-based simulated annealing.
        Energy = violations * 1000 + distinct_colors * w_color.
        This allows the search to temporarily accept invalid colorings
        (higher violations) if it reduces the color count, helping
        escape local minima.
        """
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        
        best_coloring = None
        best_energy = float('inf')
        
        # Multiple restarts with different initializations
        init_strategies = [
            lambda: [rng.randint(0, k - 1) for _ in range(N)],
            lambda: [i % k for i in range(N)],
            lambda: [(i * 7) % k for i in range(N)],
            lambda: [bin(i).count('1') % k for i in range(N)],
            lambda: [(i * 13 + 5) % k for i in range(N)],
            lambda: [rng.choice([0, 1]) for _ in range(N)] if k >= 2 else [0]*N,
        ]
        
        num_restarts = 4
        time_per_restart = time_budget / num_restarts
        
        for restart in range(num_restarts):
            if time.time() - start_t > time_budget - 1.0:
                break
            
            init_fn = init_strategies[restart % len(init_strategies)]
            coloring = init_fn()
            
            current_energy = compute_energy(coloring, k, w_color)
            if current_energy < best_energy:
                best_energy = current_energy
                best_coloring = coloring[:]
            
            # Check if already valid
            if count_violations(coloring) == 0:
                return coloring
            
            T = 15.0
            cooling = 0.9995
            steps = 0
            max_steps = 80000
            
            while time.time() - start_t < time_budget - 1.0 and steps < max_steps:
                steps += 1
                idx = rng.randint(0, N - 1)
                new_color = rng.randint(0, k - 1)
                
                if coloring[idx] == new_color:
                    continue
                
                de = delta_energy(coloring, idx, new_color, k, w_color)
                
                if de < 0:
                    # Always accept improvements
                    coloring[idx] = new_color
                    current_energy += de
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_coloring = coloring[:]
                else:
                    # Accept worse solutions with probability
                    if T > 0.01:
                        accept = rng.random() < np.exp(-de / T)
                    else:
                        accept = False
                    
                    if accept:
                        coloring[idx] = new_color
                        current_energy += de
                
                T *= cooling
            
            # Check if we found a valid coloring
            if count_violations(coloring) == 0:
                return coloring
        
        # Return best found
        if best_coloring is not None:
            return best_coloring
        return None

    def eliminate_color(coloring, target_color, rng=None, max_time=3.0):
        """
        Try to eliminate a specific color by recoloring all positions
        that use it. Returns the modified coloring if successful,
        otherwise the original.
        """
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        colors = coloring[:]
        n_colors = count_colors(colors)
        unique_colors = list(set(colors))
        
        positions = [i for i in range(N) if colors[i] == target_color]
        if not positions:
            return colors
        
        # Try to recolor each position to another color
        success = True
        for p in positions:
            if time.time() - start_t > max_time:
                return coloring
            old = colors[p]
            found = False
            # Shuffle for variety
            shuffled = unique_colors[:]
            rng.shuffle(shuffled)
            for nc in shuffled:
                if nc == old:
                    continue
                colors[p] = nc
                if validate(colors):
                    found = True
                    break
                colors[p] = old
            if not found:
                colors[p] = old
                success = False
                break
        
        if success:
            return colors
        return coloring

    def greedy_construct(k):
        """
        Greedy construction: assign colors one by one, choosing the
        color that minimizes violations while trying to keep color count low.
        """
        colors = [0] * N
        for i in range(N):
            best_c = 0
            best_v = float('inf')
            for c in range(k):
                colors[i] = c
                v = count_violations(colors)
                if v < best_v:
                    best_v = v
                    best_c = c
            colors[i] = best_c
        return colors

    # Strategy: Use penalty-based SA to find valid colorings with minimal colors
    # Try k=3, 4, 5, 6 with increasing time budgets
    
    best_colors = None
    best_count = N + 1
    
    rng = random.Random(42)
    
    # Allocate time for each k
    # k=3 is most valuable, give it more time
    time_allocations = [20.0, 15.0, 12.0, 10.0]
    
    for idx, k in enumerate([3, 4, 5, 6]):
        if time.time() > deadline - 15:
            break
        
        print(f"Trying k={k} colors with penalty SA...", file=sys.stderr)
        time_budget = min(time_allocations[idx], deadline - time.time() - 10)
        if time_budget < 3.0:
            break
        
        result = penalty_sa(k, time_budget=time_budget, rng=rng, w_color=50.0)
        
        if result is not None and validate(result):
            best_colors = result
            best_count = k
            print(f"Found valid coloring with k={k} colors", file=sys.stderr)
            break
        else:
            print(f"k={k} failed", file=sys.stderr)
    
    # If no valid coloring found via SA, use greedy fallback
    if best_colors is None:
        print("SA failed, using greedy fallback...", file=sys.stderr)
        for k in [4, 5, 6, 8]:
            if time.time() > deadline - 5:
                break
            best_colors = greedy_construct(k)
            if validate(best_colors):
                best_count = count_colors(best_colors)
                print(f"Greedy found valid coloring with {best_count} colors", file=sys.stderr)
                break
        
        if best_colors is None or not validate(best_colors):
            # Emergency fallback: all distinct colors
            best_colors = list(range(N))
            best_count = N
    
    # Try to eliminate colors from the valid coloring
    if best_count > 3 and time.time() < deadline - 10:
        print(f"Attempting color elimination from {best_count} colors...", file=sys.stderr)
        max_elim_time = min(20.0, deadline - time.time() - 5)
        if max_elim_time > 2.0:
            current = best_colors[:]
            current_count = best_count
            unique_colors = list(set(current))
            rng.shuffle(unique_colors)
            
            elim_start = time.time()
            for target in unique_colors:
                if time.time() - elim_start > max_elim_time - 2.0:
                    break
                if current_count <= 3:
                    break
                
                new_current = eliminate_color(current, target, rng=rng, max_time=3.0)
                if count_colors(new_current) < current_count:
                    current = new_current
                    current_count = count_colors(current)
                    print(f"Eliminated color, now {current_count} colors", file=sys.stderr)
            
            if current_count < best_count:
                best_colors = current
                best_count = current_count
    
    # Final relabel
    best_colors, final_count = relabel(best_colors)
    
    # Final validation
    if not validate(best_colors):
        print("WARNING: final validation failed, using periodic fallback", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END

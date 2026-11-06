"""
Combined approach: Backtracking with constraint propagation for exact color minimization,
combined with simulated annealing for robustness and speed.
Uses a structural shift from pure SA to constraint-satisfaction-first,
with SA as a fallback/refinement mechanism.
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

    def get_forbidden(colors, pos, k):
        """Get set of colors forbidden at position pos given current partial coloring."""
        forbidden = set()
        
        # pos as 4th term: a, a+d, a+2d, pos where pos = a+3d => a = pos-3d
        for d in range(1, pos // 3 + 1):
            a = pos - 3 * d
            if a < 0:
                continue
            b, c = a + d, a + 2 * d
            if b < N and c < N:
                s = {colors[a], colors[b], colors[c]}
                if len(s) <= 2:
                    forbidden |= s
        
        # pos as 3rd term: a, a+d, pos, a+3d where pos = a+2d => a = pos-2d
        for d in range(1, min(pos // 2, (N - 1 - pos)) + 1):
            a = pos - 2 * d
            if a < 0:
                continue
            b = a + d  # = pos - d
            e = a + 3 * d  # = pos + d
            if e < N:
                s = {colors[a], colors[b], colors[e]}
                if len(s) <= 2:
                    forbidden |= s
        
        # pos as 2nd term: a, pos, a+2d, a+3d where pos = a+d => a = pos-d
        for d in range(1, min(pos, (N - 1 - pos) // 2) + 1):
            a = pos - d
            if a < 0:
                continue
            c = pos + d  # a+2d
            e = pos + 2 * d  # a+3d
            if e < N:
                s = {colors[a], colors[c], colors[e]}
                if len(s) <= 2:
                    forbidden |= s
        
        # pos as 1st term: pos, pos+d, pos+2d, pos+3d
        for d in range(1, (N - 1 - pos) // 3 + 1):
            b = pos + d
            c = pos + 2 * d
            e = pos + 3 * d
            if b < N and c < N and e < N:
                s = {colors[b], colors[c], colors[e]}
                if len(s) <= 2:
                    forbidden |= s
        
        return forbidden

    def backtrack_coloring(k, time_budget=10.0, rng=None):
        """
        Backtracking with constraint propagation to find a valid k-coloring.
        Tries to assign colors 0..k-1 to positions 0..N-1.
        Uses MRV (minimum remaining values) heuristic.
        """
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        
        # colors[i] = -1 means unassigned
        colors = [-1] * N
        
        # Precompute which positions are "critical" (involved in many APs)
        # to guide ordering
        ap_counts = [len(ap_indices[i]) for i in range(N)]
        
        def is_consistent(pos, c):
            """Check if assigning color c to pos is consistent with existing assignments."""
            # Check all APs involving pos
            for ap_id in ap_indices[pos]:
                ap = aps[ap_id]
                # Collect colors of assigned positions in this AP
                assigned_colors = []
                all_assigned = True
                for idx in ap:
                    if idx == pos:
                        continue
                    if colors[idx] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[idx])
                
                if not all_assigned:
                    # Not all other positions are assigned, so we can't fully check yet
                    # But we can check if the assigned ones already force a violation
                    # If pos gets color c, and the other 3 assigned colors have <= 2 distinct,
                    # and c is not in that set, then we'd have exactly the set size
                    # Actually, we need: len({c} ∪ assigned_colors) >= 3
                    # If len(assigned_colors distinct) >= 3, always ok
                    # If len(assigned_colors distinct) == 2, need c not in that set
                    # If len(assigned_colors distinct) == 1, need c != that color
                    # If len(assigned_colors distinct) == 0, always ok
                    distinct = set(assigned_colors)
                    if len(distinct) >= 3:
                        continue
                    elif len(distinct) == 2:
                        if c in distinct:
                            # Then total distinct = 2, violation
                            return False
                    elif len(distinct) == 1:
                        if c in distinct:
                            # Then total distinct = 1, violation
                            return False
                    # else len 0, always ok
            
            return True
        
        def backtrack(pos, remaining):
            """Recursively assign color to position pos."""
            if time.time() - start_t > time_budget:
                return False
            
            if pos == N:
                return True
            
            # MRV: try colors in order, but skip forbidden ones
            forbidden = get_forbidden(colors, pos, k)
            
            # Try each color
            for c in range(k):
                if c in forbidden:
                    continue
                
                colors[pos] = c
                
                if backtrack(pos + 1, remaining - 1):
                    return True
                
                colors[pos] = -1
            
            return False
        
        # Start backtracking from position 0
        if backtrack(0, N):
            return colors
        
        return None

    def try_k_colors_sa(k, time_budget=8.0, rng=None):
        """Use simulated annealing to find a valid coloring with at most k colors."""
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        
        # Initialize with random colors
        coloring = [rng.randint(0, k - 1) for _ in range(N)]
        
        def count_violations(col):
            violations = 0
            for ap in aps:
                c0, c1, c2, c3 = col[ap[0]], col[ap[1]], col[ap[2]], col[ap[3]]
                if len({c0, c1, c2, c3}) < 3:
                    violations += 1
            return violations
        
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        max_steps = 200000
        
        while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget:
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations incrementally
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
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta_violations == 0:
                    accept = rng.random() < 0.5
                else:
                    if T > 0.001:
                        accept = rng.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        if best_violations == 0:
            return best_coloring
        if current_violations == 0:
            return coloring
        return None

    def local_search_reduce(colors, max_time=5.0, rng=None):
        """Try to reduce the number of distinct colors via local search."""
        if rng is None:
            rng = random.Random()
        colors = colors[:]
        n_colors = count_colors(colors)
        start_t = time.time()
        
        unique_colors = list(set(colors))
        
        for _ in range(50000):
            if time.time() - start_t > max_time:
                break
            
            improved = False
            pos = rng.randint(0, N - 1)
            old_c = colors[pos]
            
            for nc in unique_colors:
                if nc == old_c:
                    continue
                colors[pos] = nc
                if validate(colors):
                    new_count = count_colors(colors)
                    if new_count < n_colors:
                        n_colors = new_count
                        unique_colors = list(set(colors))
                        improved = True
                        break
                colors[pos] = old_c
            
            if not improved:
                for target_color in unique_colors:
                    positions = [i for i in range(N) if colors[i] == target_color]
                    if len(positions) == 0:
                        continue
                    
                    success = True
                    for p in positions:
                        old = colors[p]
                        found = False
                        for nc in unique_colors:
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
                        n_colors = count_colors(colors)
                        unique_colors = list(set(colors))
                        improved = True
                        break
            
            if not improved:
                pos = rng.randint(0, N - 1)
                old_c = colors[pos]
                nc = rng.choice(unique_colors)
                if nc != old_c:
                    colors[pos] = nc
                    if validate(colors):
                        new_count = count_colors(colors)
                        if new_count < n_colors:
                            n_colors = new_count
                            unique_colors = list(set(colors))
                    else:
                        colors[pos] = old_c
        
        return colors

    # Strategy: First try backtracking with constraint propagation for k=3,4,5
    # This is the structural change: exact constraint satisfaction first
    best_colors = None
    best_count = N + 1
    
    rng = random.Random(42)
    
    # Try backtracking first for small k
    for k in [3, 4, 5]:
        if time.time() > deadline - 20:
            break
        
        print(f"Trying backtracking with k={k} colors...", file=sys.stderr)
        time_budget = min(15.0, deadline - time.time() - 10)
        result = backtrack_coloring(k, time_budget=time_budget, rng=rng)
        
        if result is not None and validate(result):
            best_colors = result
            best_count = k
            print(f"Backtracking found valid coloring with k={k} colors", file=sys.stderr)
            break
        else:
            print(f"Backtracking k={k} failed or timed out", file=sys.stderr)
    
    # If backtracking failed, try SA
    if best_colors is None:
        print("Backtracking failed, trying SA...", file=sys.stderr)
        for k in [3, 4, 5, 6]:
            if time.time() > deadline - 15:
                break
            
            time_budget = min(8.0, deadline - time.time() - 10)
            result = try_k_colors_sa(k, time_budget=time_budget, rng=rng)
            
            if result is not None and validate(result):
                best_colors = result
                best_count = k
                print(f"SA found valid coloring with k={k} colors", file=sys.stderr)
                break
            else:
                print(f"SA k={k} failed", file=sys.stderr)
    
    # If no valid coloring found, use greedy fallback
    if best_colors is None:
        print("All methods failed, using greedy fallback...", file=sys.stderr)
        best_colors = [0] * N
        for i in range(N):
            forbidden = get_forbidden(best_colors, i, N)
            found = False
            for c in range(N):
                if c not in forbidden:
                    best_colors[i] = c
                    found = True
                    break
            if not found:
                best_colors[i] = 0
        
        if not validate(best_colors):
            for init in range(1, 10):
                if time.time() > deadline - 5:
                    break
                best_colors = [init % 4] * N
                valid = True
                for i in range(1, N):
                    forbidden = get_forbidden(best_colors, i, 4)
                    found = False
                    for c in range(4):
                        if c not in forbidden:
                            best_colors[i] = c
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid and validate(best_colors):
                    break
        
        best_count = count_colors(best_colors)
    
    # Local search to try to reduce colors further
    if best_count > 3 and time.time() < deadline - 10:
        print(f"Local search to reduce from {best_count} colors...", file=sys.stderr)
        reduced = local_search_reduce(best_colors, max_time=min(10.0, deadline - time.time() - 5), rng=rng)
        if count_colors(reduced) < best_count:
            best_colors = reduced
            best_count = count_colors(best_colors)
            print(f"Reduced to {best_count} colors", file=sys.stderr)
    
    # Final relabel
    best_colors, final_count = relabel(best_colors)
    
    # Final validation
    if not validate(best_colors):
        print("WARNING: final validation failed, using periodic fallback", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END

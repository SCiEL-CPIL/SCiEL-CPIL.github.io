"""
Spectral + structured approach for 4-AP colorings on [0,128].
Uses algebraic structure (modular patterns) combined with targeted local search.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
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

    def incremental_delta(coloring, idx, old_color, new_color):
        """Compute change in violation count when changing position idx from old_color to new_color."""
        if old_color == new_color:
            return 0
        delta = 0
        for ap_id in ap_indices[idx]:
            ap = aps[ap_id]
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            old_set = {c0, c1, c2, c3}
            old_size = len(old_set)
            
            new_c0 = new_color if ap[0] == idx else c0
            new_c1 = new_color if ap[1] == idx else c1
            new_c2 = new_color if ap[2] == idx else c2
            new_c3 = new_color if ap[3] == idx else c3
            new_set = {new_c0, new_c1, new_c2, new_c3}
            new_size = len(new_set)
            
            if old_size < 3 and new_size >= 3:
                delta -= 1
            elif old_size >= 3 and new_size < 3:
                delta += 1
        return delta

    def try_sa_k_colors(k, time_budget, rng, init=None):
        """Simulated annealing to find valid k-coloring."""
        start_t = time.time()
        if init is not None:
            coloring = init[:]
        else:
            coloring = [rng.randint(0, k - 1) for _ in range(N)]
        
        current_violations = count_violations(coloring)
        if current_violations == 0:
            return coloring
        
        best_coloring = coloring[:]
        best_violations = current_violations
        T = 3.0
        cooling = 0.9998
        steps = 0
        max_steps = 200000
        
        while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget - 0.5:
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            delta = incremental_delta(coloring, idx, old_color, new_color)
            
            if delta < 0:
                coloring[idx] = new_color
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta == 0:
                    accept = rng.random() < 0.4
                else:
                    if T > 0.01:
                        accept = rng.random() < np.exp(-delta / T)
                    else:
                        accept = False
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta
            
            T *= cooling
        
        if best_violations == 0:
            return best_coloring
        return None

    def reduce_colors(coloring, max_time, rng):
        """Try to reduce the number of colors in a valid coloring."""
        coloring = coloring[:]
        n_colors = count_colors(coloring)
        start_t = time.time()
        
        # Repeatedly try to eliminate one color
        while time.time() - start_t < max_time and n_colors > 3:
            unique = sorted(set(coloring))
            eliminated = False
            
            for target in unique:
                if time.time() - start_t > max_time:
                    break
                if n_colors <= 3:
                    break
                
                positions = [i for i in range(N) if coloring[i] == target]
                if not positions:
                    continue
                
                # Try to recolor all positions of target to other colors
                saved = coloring[:]
                success = True
                for p in positions:
                    found = False
                    for nc in unique:
                        if nc == target:
                            continue
                        coloring[p] = nc
                        if validate(coloring):
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                if success:
                    n_colors = count_colors(coloring)
                    eliminated = True
                    break
                else:
                    coloring = saved[:]
            
            if not eliminated:
                # Try random perturbation
                if time.time() - start_t > max_time:
                    break
                for _ in range(100):
                    if time.time() - start_t > max_time:
                        break
                    pos = rng.randint(0, N - 1)
                    old_c = coloring[pos]
                    nc = rng.randint(0, n_colors - 1)
                    if nc != old_c:
                        coloring[pos] = nc
                        if validate(coloring):
                            new_n = count_colors(coloring)
                            if new_n < n_colors:
                                n_colors = new_n
                                break
                        else:
                            coloring[pos] = old_c
        
        return coloring

    # Strategy 1: Try structured initializations
    rng = random.Random(12345)
    
    best_colors = None
    best_count = N + 1
    
    # Generate diverse structured initializations
    structured_inits = []
    
    # Modulo-based patterns
    for mod in [4, 5, 6, 7, 8]:
        structured_inits.append([i % mod for i in range(N)])
    
    # Bit-pattern based
    for shift in range(4):
        structured_inits.append([(i >> shift) & 3 for i in range(N)])
    
    # Mixed patterns
    structured_inits.append([bin(i).count('1') % 4 for i in range(N)])
    structured_inits.append([(i * 7 + 3) % 5 for i in range(N)])
    structured_inits.append([(i * 11 + 7) % 6 for i in range(N)])
    
    # Try each structured initialization with SA
    for init in structured_inits:
        if time.time() > deadline - 10:
            break
        k = count_colors(init)
        if k >= best_count:
            continue
        
        result = try_sa_k_colors(k, time_budget=3.0, rng=rng, init=init)
        if result is not None:
            if best_colors is None or count_colors(result) < best_count:
                best_colors = result
                best_count = count_colors(result)
                if best_count <= 3:
                    break
    
    # Strategy 2: Pure SA with random inits for k=3,4,5
    for k in [3, 4, 5]:
        if time.time() > deadline - 8:
            break
        if k >= best_count:
            continue
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            result = try_sa_k_colors(k, time_budget=2.0, rng=rng)
            if result is not None:
                if best_colors is None or k < best_count:
                    best_colors = result
                    best_count = k
                    break
    
    # Fallback: greedy construction
    if best_colors is None:
        best_colors = [0] * N
        for i in range(N):
            forbidden = set()
            # Check all APs containing position i
            for ap_id in ap_indices[i]:
                ap = aps[ap_id]
                others = [ap[j] for j in range(4) if ap[j] != i]
                if len(others) == 3:
                    s = {best_colors[others[0]], best_colors[others[1]], best_colors[others[2]]}
                    if len(s) <= 2:
                        forbidden |= s
            
            found = False
            for c in range(N):
                if c not in forbidden:
                    best_colors[i] = c
                    found = True
                    break
            if not found:
                best_colors[i] = 0
        
        if not validate(best_colors):
            # Emergency: use a known-good periodic pattern
            best_colors = [i % 4 for i in range(N)]
        
        best_count = count_colors(best_colors)
    
    # Try to reduce colors
    if best_count > 3 and time.time() < deadline - 10:
        max_reduce_time = min(15.0, deadline - time.time() - 5)
        if max_reduce_time > 2.0:
            reduced = reduce_colors(best_colors, max_reduce_time, rng)
            if count_colors(reduced) < best_count:
                best_colors = reduced
                best_count = count_colors(best_colors)
    
    # Final relabel and validation
    best_colors, final_count = relabel(best_colors)
    
    if not validate(best_colors):
        # Last resort
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END

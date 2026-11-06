"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 85.0  # safety margin
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
    
    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations
    
    def greedy_construct(k, seed=None):
        """Build coloring greedily, position by position, choosing color that minimizes future violations."""
        if seed is not None:
            random.seed(seed)
        coloring = [None] * N
        for pos in range(N):
            best_color = 0
            best_score = float('inf')
            # Try all k colors, score by how many violations this creates with already-colored positions
            for c in range(k):
                coloring[pos] = c
                # Count violations involving this position with already-colored positions
                violations = 0
                for ap_id in ap_indices[pos]:
                    ap = aps[ap_id]
                    # Check if all other positions are colored
                    # Optimization: precompute the indices of the AP
                    i0, i1, i2, i3 = ap
                    # Since we are filling sequentially, 'others' might not be colored yet.
                    # We only count violations if ALL 4 are colored.
                    if coloring[i0] is not None and coloring[i1] is not None and coloring[i2] is not None and coloring[i3] is not None:
                        if len({coloring[i0], coloring[i1], coloring[i2], coloring[i3]}) < 3:
                            violations += 1
                coloring[pos] = None
                if violations < best_score:
                    best_score = violations
                    best_color = c
            coloring[pos] = best_color
        return coloring
    
    def simulated_annealing(coloring, k, max_steps=5000, deadline_t=None):
        """Refine coloring using SA with penalty-based objective: violations + weight * distinct_colors."""
        current_violations = count_violations(coloring)
        
        # Maintain color counts for O(1) distinct count updates
        color_counts = [0] * k
        for c in coloring:
            color_counts[c] += 1
        current_distinct = sum(1 for cnt in color_counts if cnt > 0)
        
        best_coloring = coloring[:]
        best_violations = current_violations
        best_distinct = current_distinct
        
        # Penalty weight: large enough to strongly prefer fewer colors when violations are equal
        color_penalty = 100.0
        T = 10.0
        cooling = 0.999
        steps = 0
        rng = random.Random() # Local RNG for thread safety/consistency if needed, though global is fine
        
        while steps < max_steps:
            if deadline_t is not None and time.time() > deadline_t:
                break
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                i0, i1, i2, i3 = ap
                c0 = coloring[i0]
                c1 = coloring[i1]
                c2 = coloring[i2]
                c3 = coloring[i3]
                
                old_set_size = len({c0, c1, c2, c3})
                
                # Construct new set efficiently
                if i0 == idx:
                    new_c0 = new_color
                else:
                    new_c0 = c0
                if i1 == idx:
                    new_c1 = new_color
                else:
                    new_c1 = c1
                if i2 == idx:
                    new_c2 = new_color
                else:
                    new_c2 = c2
                if i3 == idx:
                    new_c3 = new_color
                else:
                    new_c3 = c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            # Track distinct colors using counts
            delta_distinct = 0
            if color_counts[old_color] == 1:
                delta_distinct -= 1
            if color_counts[new_color] == 0:
                delta_distinct += 1
            
            # Energy calculation
            # E = V + P * max(0, D - 3)
            old_penalty = color_penalty * max(0, current_distinct - 3)
            new_penalty = color_penalty * max(0, current_distinct + delta_distinct - 3)
            delta_energy = delta_violations + new_penalty - old_penalty
            
            if delta_energy < 0 or (T > 0.01 and rng.random() < np.exp(-delta_energy / T)):
                # Accept move
                coloring[idx] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                current_violations += delta_violations
                current_distinct += delta_distinct
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_distinct = current_distinct
                    best_coloring = coloring[:]
            
            T *= cooling
        
        return best_coloring, best_violations
    
    # Try k=3 first, then k=4, etc.
    # We limit k to small numbers because the problem likely has a small solution.
    # If k=3 fails, try k=4.
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 10.0:
            break
        
        found_valid = False
        best_coloring = None
        best_violations = float('inf')
        
        # Multiple restarts
        # We can afford more restarts because each SA run is faster (O(1) distinct count)
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Greedy construction
            coloring = greedy_construct(k, seed=42 + restart * 137)
            
            # Refine with SA
            refined, violations = simulated_annealing(coloring, k, max_steps=2000, deadline_t=deadline - 1.0)
            
            if violations < best_violations:
                best_violations = violations
                best_coloring = refined
            
            if violations == 0:
                found_valid = True
                break
        
        if found_valid or best_violations == 0:
            return best_coloring
    
    # Fallback: return a valid coloring with more colors
    # Use distinct colors to guarantee validity (always valid, but not minimal)
    return list(range(N))
# EVOLVE-BLOCK-END

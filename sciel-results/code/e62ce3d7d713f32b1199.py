import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    positions = list(range(1, n + 1))
    
    # Precompute all nonconstant 4-APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d <= n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
            else:
                break
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = {i: [] for i in positions}
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Precompute AP color patterns for fast violation checking
    # For each AP, store the 4 positions as indices 0..127
    ap_positions = [[p - 1 for p in ap] for ap in aps]
    
    def count_violations(colors):
        """Count APs with fewer than 3 distinct colors."""
        violations = 0
        for ap_idx in range(num_aps):
            c0 = colors[ap_positions[ap_idx][0]]
            c1 = colors[ap_positions[ap_idx][1]]
            c2 = colors[ap_positions[ap_idx][2]]
            c3 = colors[ap_positions[ap_idx][3]]
            # Count distinct colors
            distinct = 1
            if c1 != c0:
                distinct += 1
            if c2 != c0 and c2 != c1:
                distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2:
                distinct += 1
            if distinct < 3:
                violations += 1
        return violations
    
    def get_violated_aps(colors):
        """Return list of AP indices that are violated."""
        violated = []
        for ap_idx in range(num_aps):
            c0 = colors[ap_positions[ap_idx][0]]
            c1 = colors[ap_positions[ap_idx][1]]
            c2 = colors[ap_positions[ap_idx][2]]
            c3 = colors[ap_positions[ap_idx][3]]
            distinct = 1
            if c1 != c0:
                distinct += 1
            if c2 != c0 and c2 != c1:
                distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2:
                distinct += 1
            if distinct < 3:
                violated.append(ap_idx)
        return violated
    
    def simulated_annealing(k, max_iters=50000, seed=0):
        rng = random.Random(seed)
        
        # Initialize with random coloring
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        current_energy = count_violations(colors)
        
        best_colors = colors[:]
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random position to recolor
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            new_color = rng.randint(0, k - 1)
            
            if new_color == old_color:
                continue
            
            colors[pos] = new_color
            
            # Compute delta energy efficiently
            # Only APs containing pos can change violation status
            delta_energy = 0
            for ap_idx in pos_to_aps[pos + 1]:
                # Check old state
                ap_pos = ap_positions[ap_idx]
                # We need to check if this AP was violated before and after
                # Since we already changed colors[pos], let's compute both
                # Actually, let's just recompute for affected APs
                pass
            
            # Simpler: just compute new energy for affected APs
            # This is O(degree) per move
            new_energy = current_energy
            
            # For each AP containing pos, check if violation status changed
            # First, undo the change to compute old status
            colors[pos] = old_color
            old_violated_count = 0
            for ap_idx in pos_to_aps[pos + 1]:
                ap_pos = ap_positions[ap_idx]
                c0 = colors[ap_pos[0]]
                c1 = colors[ap_pos[1]]
                c2 = colors[ap_pos[2]]
                c3 = colors[ap_pos[3]]
                distinct = 1
                if c1 != c0:
                    distinct += 1
                if c2 != c0 and c2 != c1:
                    distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2:
                    distinct += 1
                if distinct < 3:
                    old_violated_count += 1
            
            # Now apply new color
            colors[pos] = new_color
            new_violated_count = 0
            for ap_idx in pos_to_aps[pos + 1]:
                ap_pos = ap_positions[ap_idx]
                c0 = colors[ap_pos[0]]
                c1 = colors[ap_pos[1]]
                c2 = colors[ap_pos[2]]
                c3 = colors[ap_pos[3]]
                distinct = 1
                if c1 != c0:
                    distinct += 1
                if c2 != c0 and c2 != c1:
                    distinct += 1
                if c3 != c0 and c3 != c1 and c3 != c2:
                    distinct += 1
                if distinct < 3:
                    new_violated_count += 1
            
            # Energy is total violations, so delta = new - old for affected APs
            # But we need to account for APs not containing pos (they don't change)
            # The total energy change is (new_violated_count - old_violated_count)
            # because only APs containing pos can change
            delta_energy = new_violated_count - old_violated_count
            new_energy = current_energy + delta_energy
            
            # Accept or reject
            if new_energy < current_energy or rng.random() < math.exp(-delta_energy / max(T, T_min)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        break
            else:
                colors[pos] = old_color
            
            T *= cooling
        
        return best_colors, best_energy
    
    # Try simulated annealing with different k values
    best_result = None
    best_k = None
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            result, energy = simulated_annealing(k, max_iters=30000, seed=seed)
            
            if energy == 0:
                best_result = result
                best_k = k
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: use the best energy result found
        # Re-run with more iterations for the smallest k
        for k in [3, 4, 5]:
            if time.time() > deadline - 3:
                break
            result, energy = simulated_annealing(k, max_iters=40000, seed=42)
            if best_result is None or energy < count_violations(best_result):
                best_result = result
                best_k = k
    
    if best_result is not None:
        # Map colors to 0..127 range (they're already in 0..k-1 which is in 0..127)
        return best_result
    
    # Final fallback: greedy construction
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d < 1:
                    break
                idx = x - 1 - 3 * d
                if idx >= len(colors):
                    break
                # Check AP: (x-3d, x-2d, x-d, x)
                # We need at least 3 distinct colors
                # The existing 3 are at positions x-3d, x-2d, x-d (0-indexed: x-1-3d, x-1-2d, x-1-d)
                c1 = colors[x - 1 - 3 * d]
                c2 = colors[x - 1 - 2 * d]
                c3 = colors[x - 1 - d]
                # With new color, we need at least 3 distinct
                # If {c1,c2,c3} already has 3, any color works
                # If {c1,c2,c3} has 2, new color must be different from both
                # If {c1,c2,c3} has 1, new color can be anything
                distinct_existing = len({c1, c2, c3})
                if distinct_existing >= 3:
                    continue
                elif distinct_existing == 2:
                    # New color must not be in {c1,c2,c3} to make 3 distinct
                    # Wait, we need at least 3 distinct total
                    # If existing has 2, adding a 3rd distinct gives 3
                    # Adding one of the existing gives 2 (still bad)
                    if color in {c1, c2, c3}:
                        valid = False
                        break
                else:  # distinct_existing == 1
                    # Adding any color: if same, still 1; if different, 2
                    # We need 3, so this AP can't be satisfied with just this new color
                    # Unless... wait, we need at least 3 distinct in the 4-term AP
                    # With 3 existing same and 1 new: max 2 distinct. So this is always bad.
                    # But wait, maybe I'm miscounting. Let me re-examine.
                    # The AP is (x-3d, x-2d, x-d, x). That's 4 positions.
                    # We're coloring x. The other 3 are already colored.
                    # If all 3 existing are the same color, then with new color:
                    #   - same as existing: 1 distinct
                    #   - different: 2 distinct
                    # Either way < 3. So this AP will be violated regardless.
                    # But we still need to pick a color that minimizes future violations.
                    # For now, just accept it (we can't fix this AP).
                    pass
            
            if valid:
                colors.append(color)
                found = True
                break
        
        if not found:
            colors.append(0)
    
    return colors

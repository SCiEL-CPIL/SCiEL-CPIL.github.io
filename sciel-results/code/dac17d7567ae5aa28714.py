"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    tap_indices = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
            tap_indices.append(len(all_taps) - 1)
            
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    def count_violations(colors):
        """Count number of 4-term APs that violate the constraint (use < 3 distinct colors)."""
        violations = 0
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations

    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True

    def local_search(k, time_limit):
        """
        Attempt to find a valid coloring using exactly k colors (labels 0..k-1).
        Uses simulated annealing with incremental violation updates.
        """
        start_t = time.time()
        
        # Initialize with random coloring using k colors
        colors = [random.randint(0, k - 1) for _ in range(N)]
        
        # Initial violation count
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors[:]
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() - start_t < time_limit:
            if current_violations == 0:
                return colors
            
            # Pick a random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Pick a new color from 0..k-1
            if random.random() < 0.5:
                new_color = random.randint(0, k - 1)
            else:
                # Try a color different from old if possible
                new_color = (old_color + 1) % k
                if new_color == old_color and k > 1:
                    new_color = (old_color + 2) % k
            
            if new_color == old_color:
                continue
            
            # Calculate delta violations
            # This is expensive if done naively, but we can optimize by checking only affected taps.
            # For simplicity and robustness in this short time, we recompute violations for affected taps.
            
            # Store old colors of neighbors to revert if needed? No, we just compute delta.
            # To compute delta efficiently:
            # 1. For each tap involving pos, check if it was violated before and if it is violated after.
            
            delta = 0
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                # Check before
                before_colors = {colors[a], colors[b], colors[c], colors[d]}
                was_violated = len(before_colors) < 3
                
                # Temporarily assign
                colors[pos] = new_color
                
                # Check after
                after_colors = {colors[a], colors[b], colors[c], colors[d]}
                is_violated = len(after_colors) < 3
                
                # Revert for now to calculate delta cleanly? 
                # Actually, we are updating colors in place. 
                # If we update in place, we need to be careful.
                # Let's just revert and calculate delta properly.
                colors[pos] = old_color
                
                if was_violated and not is_violated:
                    delta -= 1
                elif not was_violated and is_violated:
                    delta += 1
            
            new_violations = current_violations + delta
            
            if new_violations < current_violations or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept move
                colors[pos] = new_color
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    if current_violations == 0:
                        return colors
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # If stuck for a while, maybe restart with a different initial random seed?
            # Or just keep going.
            
        return None

    # Try k=3 first
    # We might need multiple attempts for k=3 if it's hard, but usually 1-2 seconds is enough if a solution exists.
    # If k=3 fails, try k=4, etc.
    
    # To ensure we don't waste too much time on k=3 if it's impossible or very hard, 
    # we give it a specific time budget.
    
    time_per_k = 10.0
    
    # Try k=3
    res = local_search(3, time_per_k)
    if res is not None:
        return res
        
    # Try k=4
    if time.time() < deadline - 5:
        res = local_search(4, time_per_k)
        if res is not None:
            return res
            
    # Try k=5
    if time.time() < deadline - 5:
        res = local_search(5, time_per_k)
        if res is not None:
            return res
            
    # Fallback: If no valid coloring found with 5 colors (unlikely), 
    # return a valid coloring with more colors to be safe, or just the best found.
    # But the task requires validity. 
    # A simple valid coloring with many colors is i % 4 (uses 4 colors, valid? 
    # Wait, i%4: 0,1,2,3,0,1,2,3. 
    # AP: 0,1,2,3 -> 0,1,2,3 (4 colors, ok)
    # AP: 0,2,4,6 -> 0,2,0,2 (2 colors, VIOLATION)
    # So i%4 is NOT valid.
    
    # We need a guaranteed valid coloring.
    # Using distinct colors for all positions is always valid (128 colors).
    # But we want to minimize.
    # Let's try a greedy construction for a fallback that is guaranteed valid.
    
    def greedy_valid_fallback():
        colors = [-1] * N
        # We can use up to N colors.
        # Greedy: assign smallest color that doesn't create a violation with previous assignments.
        # Since we go left to right, we only need to check APs where the current position is the largest index?
        # No, APs can wrap around or have smaller indices.
        # Actually, if we assign pos, we only affect APs containing pos.
        # If we ensure that for every AP, once all 4 are assigned, it's valid.
        # If we assign in order 0..N-1, when we assign pos, some other members of the AP might be unassigned.
        # We can't check validity yet.
        # So greedy left-to-right is tricky unless we delay validation.
        
        # Better fallback: Random restart local search with high k (e.g., 10) until valid.
        # Or just use 128 distinct colors? No, that's score 1.0 (seed).
        # The seed uses ~118 colors? No, seed score 1.0833 means it found a solution with ~118 colors?
        # Seed value is 128 / 1.0833 = 118.15. So seed uses ~118 colors.
        # If we return 128 colors, score is 1.0.
        # If we return a valid coloring with k colors, score is 128/k.
        
        # Let's just do a quick random search with k=10 to ensure we return *something* valid.
        for _ in range(10):
             if time.time() > deadline - 2: break
             res = local_search(10, 1.0)
             if res is not None:
                 return res
        
        # Absolute last resort: distinct colors
        return list(range(N))

    return greedy_valid_fallback()
# EVOLVE-BLOCK-END

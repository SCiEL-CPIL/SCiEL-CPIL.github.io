import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles (120 quads * 3 cycles = 360 cycles)
    all_cycles = []
    vertices = list(range(N))
    
    # Edge index helper for precomputation
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
        
    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32) # shape (360, 4)

    # 2. Precompute edge-to-cycle incidence for fast validity checks
    # edge_to_cycles[e] = list of (cycle_index, position)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for pos in range(4):
            e = cycles_arr[ci, pos]
            edge_to_cycles[e].append((ci, pos))
            
    # Convert to numpy arrays for potential vectorization or just keep lists for speed in Python loop
    # Lists of tuples are fast enough for 182 checks per move if we optimize the inner loop.
    
    def is_valid_move(edge_idx, new_color, colors):
        """
        Returns True if setting colors[edge_idx] = new_color is valid.
        Optimized for speed: checks only the ~182 cycles containing edge_idx.
        """
        old_color = colors[edge_idx]
        if new_color == old_color:
            return True
            
        for ci, pos in edge_to_cycles[edge_idx]:
            # Get the other 3 edges in this cycle
            # We can hardcode the indices or use array slicing
            # Array slicing: cycle = cycles_arr[ci]
            # others = [cycle[p] for p in range(4) if p != pos]
            
            # Manual unrolling for speed? No, let's use a small loop or precomputed offsets.
            # Precomputed offsets for 'other positions' for each pos in 0..3
            # pos 0: 1,2,3
            # pos 1: 0,2,3
            # pos 2: 0,1,3
            # pos 3: 0,1,2
            
            if pos == 0:
                e1, e2, e3 = cycles_arr[ci, 1], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 1:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 2:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 3]
            else:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 2]
                
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            # Check for 4 distinct: new_color, c1, c2, c3
            # 1. new_color must not be unique among {new, c1, c2, c3}
            if new_color == c1 or new_color == c2 or new_color == c3:
                continue
            # 2. c1, c2, c3 must not be all distinct
            if c1 == c2 or c1 == c3 or c2 == c3:
                continue
                
            # If we reach here, it's a conflict
            return False
            
        return True

    def count_colors(colors):
        return len(np.unique(colors))

    def is_fully_valid(colors):
        # Fast check: any cycle with 4 distinct?
        # Vectorized check:
        # For each cycle, get the 4 colors.
        # Check if all 4 are distinct.
        # distinct = (c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3)
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return not np.any(distinct_mask)

    # 3. Strategy: Randomized Greedy Construction + Local Repair
    # Start with a valid low-color state (all 0s).
    # Iteratively try to increase color count by:
    # a) Picking a random edge.
    # b) Picking a random color from the pool of UNUSED colors (or a large random range).
    # c) If valid, apply.
    # d) If invalid, try to "repair" by finding a neighbor edge that can be recolored to match one of the conflict colors, or just skip.
    
    # Better Strategy: "Color Expansion"
    # Maintain a set of used colors.
    # Try to introduce new colors.
    
    rng = random.Random(12345)
    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_score = 1 # At least 1 color (0)
    
    # We will run multiple restarts or a long single run?
    # A long single run with periodic "reset to best" if stuck might be good.
    # But local search on this problem can get stuck.
    # Let's try a "Simulated Annealing" like approach but focused on maximizing distinct colors.
    
    # Actually, a pure "Greedy New Color" approach often fails because it paints itself into a corner.
    # Let's use a "Random Walk with Bias to New Colors".
    
    # State: colors array
    # Objective: Maximize unique colors.
    # Move: Change color of one edge.
    # Acceptance: 
    #   - If move increases unique colors: Always accept.
    #   - If move keeps unique colors same: Accept with probability p (to explore).
    #   - If move decreases unique colors: Reject (or accept with very low prob if temp is high).
    
    # Since we start at 1 color, we want to climb to as high as possible.
    # This is a monotone improvement problem mostly.
    
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    current_used = set([0])
    
    # To speed up "is this color used?", we use a boolean array of size 120
    color_used = np.zeros(MAX_COLORS, dtype=bool)
    color_used[0] = True
    num_used = 1
    
    def add_color(c):
        nonlocal num_used
        if not color_used[c]:
            color_used[c] = True
            num_used += 1
            
    def remove_color(c):
        nonlocal num_used
        if color_used[c]:
            # Check if any other edge uses this color
            # This is expensive if done naively. 
            # Instead, maintain a count array for each color.
            pass

    # Let's use a count array for colors for O(1) add/remove
    color_counts = np.zeros(MAX_COLORS, dtype=np.int32)
    color_counts[0] = NUM_EDGES
    num_used = 1
    
    def update_color_counts(old_c, new_c):
        nonlocal num_used
        if old_c == new_c: return
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            num_used -= 1
        color_counts[new_c] += 1
        if color_counts[new_c] == 1:
            num_used += 1

    # Main Loop: Randomized Hill Climbing
    # We will try to improve the score.
    # If we get stuck (no improvement in X steps), we can try a random restart or a "kick" (accept a worsening move).
    
    steps_since_improvement = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        if steps_since_improvement > max_steps_no_improve:
            # Kick: Accept a random move even if it reduces score, to escape local optima
            # Or restart? Restarting loses progress. 
            # Let's try a "random walk" phase: accept 10 random valid moves regardless of score change.
            for _ in range(10):
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = current_colors[e]
                # Pick a random valid color
                # We need a fast way to find a valid color.
                # Just try a few random ones.
                found = False
                for _ in range(20):
                    new_c = rng.randint(0, MAX_COLORS - 1)
                    if is_valid_move(e, new_c, current_colors):
                        current_colors[e] = new_c
                        update_color_counts(old_c, new_c)
                        found = True
                        break
                if found:
                    break
            steps_since_improvement = 0
            max_steps_no_improve = int(max_steps_no_improve * 1.5) # Exponential backoff for kicks
            
        # Normal Step
        e = rng.randint(0, NUM_EDGES - 1)
        old_c = current_colors[e]
        
        # Bias towards new colors
        # Generate a candidate color
        # 50% chance: pick from unused colors
        # 50% chance: pick random color
        if rng.random() < 0.5 and num_used < MAX_COLORS:
            # Pick a random unused color
            # This is O(120) to find one if we don't have a list.
            # Let's maintain a list of unused colors? Or just try randoms.
            # Trying randoms from 0..119 and checking color_counts[c]==0 is fast.
            new_c = None
            for _ in range(10):
                c = rng.randint(0, MAX_COLORS - 1)
                if color_counts[c] == 0:
                    new_c = c
                    break
            if new_c is None:
                new_c = rng.randint(0, MAX_COLORS - 1)
        else:
            new_c = rng.randint(0, MAX_COLORS - 1)
            
        if new_c == old_c:
            continue
            
        if is_valid_move(e, new_c, current_colors):
            # Check if this improves score
            # If new_c was unused, score increases.
            # If new_c was used, score stays same (unless old_c becomes unused, then it decreases).
            
            old_count_new_c = color_counts[new_c]
            old_count_old_c = color_counts[old_c]
            
            # Apply move
            current_colors[e] = new_c
            update_color_counts(old_c, new_c)
            
            # Evaluate
            if color_counts[new_c] > old_count_new_c: # New color introduced
                steps_since_improvement = 0
                if num_used > best_score:
                    best_score = num_used
                    best_colors = current_colors.copy()
            elif color_counts[old_c] == old_count_old_c - 1 and old_count_old_c - 1 == 0:
                # A color was lost
                steps_since_improvement += 1
                # If we lost a color, we might want to revert? 
                # In pure hill climbing, we would revert. 
                # But if we are stuck, maybe accept? 
                # Let's revert if it decreases score, unless we are in a "kick" phase (handled above).
                # Actually, the "kick" phase accepts any valid move. 
                # Here, we are in normal phase. 
                # If score decreases, revert.
                current_colors[e] = old_c
                update_color_counts(new_c, old_c)
            else:
                # Score unchanged (recolored to existing color, old color still exists)
                # Accept to explore? 
                # Yes, with some probability, to shuffle state.
                if rng.random() < 0.2:
                    steps_since_improvement += 1
                else:
                    # Revert to keep search focused? 
                    # No, let's keep it. It might enable future moves.
                    # But it doesn't improve score. 
                    # Let's count it as a step without improvement.
                    steps_since_improvement += 1
        else:
            # Move invalid
            steps_since_improvement += 1

    # Final verification
    if not is_fully_valid(best_colors):
        # Fallback
        return np.zeros(NUM_EDGES, dtype=np.int32).tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")

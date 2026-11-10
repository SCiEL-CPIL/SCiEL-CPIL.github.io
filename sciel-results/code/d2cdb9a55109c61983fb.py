import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
# A 4-cycle on vertices a<b<c<d has 3 cycles:
#   (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
#   (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
#   (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 3 * C(16,4) = 3 * 1820 = 5460

# For each edge, which cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)

# Precompute for each edge, the other edges in each cycle it belongs to
# This speeds up the "safe color" calculation
EDGE_CYCLE_OTHERS = []
for ei in range(NUM_EDGES):
    others_list = []
    for ci in EDGE_TO_CYCLES[ei]:
        cyc = CYCLES[ci]
        others = [e for e in cyc if e != ei]
        others_list.append(others)
    EDGE_CYCLE_OTHERS.append(others_list)


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len(set((c1, c2, c3, c4))) == 4:
            violations += 1
    return violations


def count_distinct_colors(colors):
    return len(set(colors))


def get_safe_colors(ei, colors):
    """
    Returns a set of colors that can be assigned to edge ei without creating
    a new 4-distinct violation in any cycle containing ei, assuming the rest
    of the coloring is valid.
    """
    # Start with all possible colors
    safe_colors = set(range(MAX_COLOR + 1))
    
    others_list = EDGE_CYCLE_OTHERS[ei]
    for others in others_list:
        # Get colors of the other 3 edges
        other_colors = [colors[e] for e in others]
        unique_others = set(other_colors)
        
        if len(unique_others) == 3:
            # All 3 other edges have distinct colors.
            # To avoid 4 distinct colors, the new color MUST be one of these 3.
            safe_colors = safe_colors.intersection(unique_others)
            if not safe_colors:
                return set()
        # If len(unique_others) < 3, any color is safe with respect to this cycle
        # because even if we pick a new color, the total distinct count will be <= 3 
        # (since the others already have a duplicate).
        
    return safe_colors


def make_initial_coloring():
    """Create a valid coloring using sum-based coloring."""
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = i + j
    return colors


def greedy_improve(colors, time_budget=55.0):
    """
    Greedy local search.
    1. Compute color counts.
    2. For each edge, find safe colors.
    3. If a safe color exists that is currently unused (or less frequent), 
       and changing to it increases total distinct count, make the change.
    4. Repeat until no improvement or time runs out.
    """
    deadline = time.time() + time_budget
    colors = list(colors)
    best_colors = list(colors)
    best_distinct = count_distinct_colors(colors)
    
    # Precompute color counts
    color_counts = [0] * (MAX_COLOR + 1)
    for c in colors:
        color_counts[c] += 1
        
    distinct_count = sum(1 for c in color_counts if c > 0)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle edges to break symmetry
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline - 1.0:
                return best_colors, best_distinct
            
            current_color = colors[ei]
            safe_colors = get_safe_colors(ei, colors)
            
            # If current color is not safe, we have a problem (shouldn't happen if we start valid)
            # But let's be safe. If current is not safe, we must change.
            # However, get_safe_colors returns colors that *keep* validity.
            # If current_color is not in safe_colors, it means the current state is invalid?
            # No, get_safe_colors checks if *changing* to c creates a violation.
            # If the current state is valid, current_color must be in safe_colors.
            
            if not safe_colors:
                continue
                
            # We want to pick a color that maximizes distinct count.
            # Ideally, pick a color with count 0 (new color).
            # If none, pick a color with count 1? No, that doesn't change distinct count.
            # We want to increase distinct count.
            # So we look for a color c in safe_colors such that color_counts[c] == 0.
            # If we find one, and current_color count becomes 0 after removal, distinct increases by 1.
            # If current_color count > 1, distinct stays same? No, if we add a new color, distinct increases by 1.
            
            best_new_color = -1
            best_gain = 0
            
            # Optimization: Only check colors that are "interesting"
            # We can iterate through safe_colors.
            # To speed up, if there are many safe colors, we can sample or prioritize unused ones.
            
            # First, check if there's any unused color in safe_colors
            # This is O(|safe_colors|)
            # We can maintain a set of unused colors globally?
            # Or just check.
            
            # Let's try to find a color that increases distinct count.
            # This happens if:
            # 1. new_color is unused (count 0)
            # 2. AND (old_color count > 1 OR we don't care about losing old_color if new is new)
            # Actually, if new_color is unused, distinct count increases by 1.
            # If old_color count was 1, it becomes 0, so that color is lost.
            # Net change: +1 (new) - 1 (old lost) = 0.
            # So to INCREASE distinct count, we need:
            # - new_color is unused (count 0)
            # - old_color count > 1 (so old color remains present)
            
            # If we can't increase distinct count, we might just shuffle to escape local minima?
            # The directive says "Maximize the number of colors actually used".
            # So we only accept moves that increase distinct count.
            
            for c in safe_colors:
                if c == current_color:
                    continue
                if color_counts[c] == 0:
                    # New color added
                    if color_counts[current_color] > 1:
                        # Old color remains
                        gain = 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new_color = c
                    else:
                        # Old color disappears, new color appears. Net change 0.
                        # But maybe it helps future moves? 
                        # For now, let's strictly maximize. 
                        # We can allow gain 0 moves if we are stuck, but let's try strict first.
                        pass
            
            if best_new_color != -1:
                # Apply move
                old_c = colors[ei]
                new_c = best_new_color
                
                color_counts[old_c] -= 1
                color_counts[new_c] += 1
                
                colors[ei] = new_c
                
                # Update distinct count
                if color_counts[old_c] == 0:
                    distinct_count -= 1
                if color_counts[new_c] == 1:
                    distinct_count += 1
                
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_colors = list(colors)
                
                improved = True
    
    return best_colors, best_distinct


def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with a valid structured coloring
    initial = make_initial_coloring()
    best_colors = initial
    best_score = count_distinct_colors(initial)
    
    # Run Greedy Improvement
    if time.time() < deadline - 5.0:
        greedy_colors, greedy_distinct = greedy_improve(initial, time_budget=deadline - time.time() - 3.0)
        if count_violations(greedy_colors) == 0 and greedy_distinct > best_score:
            best_colors = greedy_colors
            best_score = greedy_distinct
    
    # Try multiple restarts with different initial colorings
    # The sum-based coloring is very rigid. Let's try adding random noise to the sum
    # and then repairing.
    
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        # Create a perturbed initial coloring
        colors = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(EDGES):
            # Base sum
            base = i + j
            # Add small random noise
            noise = random.randint(0, 5)
            colors[idx] = (base + noise) % 32 # Keep in range, might need adjustment
        
        # This coloring is likely invalid. We need to repair it.
        # Simple repair: Iterate through edges, if invalid, pick a safe color.
        # But "safe" depends on neighbors. 
        # Let's use a "relaxation" approach:
        # 1. Start with random colors.
        # 2. Repeatedly pick a random edge involved in a violation.
        # 3. Change its color to a "safe" one (one that reduces violations).
        
        # For now, let's just use the greedy_improve on a *valid* perturbation.
        # How to get a valid perturbation?
        # Take initial valid coloring, change a few edges to other valid colors.
        
        perturbed = list(initial)
        # Pick 5 random edges
        for _ in range(5):
            ei = random.randint(0, NUM_EDGES - 1)
            safe = get_safe_colors(ei, perturbed)
            if len(safe) > 1:
                # Pick a random safe color different from current
                options = [c for c in safe if c != perturbed[ei]]
                if options:
                    perturbed[ei] = random.choice(options)
        
        if count_violations(perturbed) == 0:
            d = count_distinct_colors(perturbed)
            if d > best_score:
                best_colors = perturbed
                best_score = d
            
            # Optimize from this starting point
            if time.time() < deadline - 5.0:
                budget = min(5.0, deadline - time.time() - 3.0)
                if budget > 1.0:
                    greedy_colors, greedy_distinct = greedy_improve(perturbed, time_budget=budget)
                    if count_violations(greedy_colors) == 0 and greedy_distinct > best_score:
                        best_colors = greedy_colors
                        best_score = greedy_distinct
    
    # Final check
    if count_violations(best_colors) != 0:
        # Fallback to the known valid coloring
        best_colors = make_initial_coloring()
    
    return list(best_colors)

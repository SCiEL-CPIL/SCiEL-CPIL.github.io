import time
import random
import sys

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
# For each set of 4 vertices {a,b,c,d}, there are 3 cycles.
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

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
# AND what are the other 3 edges in those cycles?
# OTHER_EDGES[e] = list of tuples (o1, o2, o3)
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations


def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    # Initial valid coloring: all 0
    colors = [0] * NUM_EDGES
    used_colors = {0}
    current_score = 1
    
    # Best found
    best_colors = list(colors)
    best_score = 1
    
    rng = random.Random(42)
    
    # Precompute list of all colors for random selection
    all_colors = list(range(MAX_COLOR + 1))
    
    stuck_counter = 0
    max_stuck = 1000
    
    while time.time() < deadline:
        if current_score == MAX_COLOR + 1:
            break
        
        # 1. Try to add a new unused color
        if len(used_colors) < MAX_COLOR + 1:
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            
            # Pick a random unused color
            # Optimization: if few unused, pick one. If many, random.
            unused = [c for c in all_colors if c not in used_colors]
            if unused:
                new_c = rng.choice(unused)
                if check_edge_change(colors, e, new_c):
                    colors[e] = new_c
                    used_colors.add(new_c)
                    current_score = len(used_colors)
                    stuck_counter = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = list(colors)
                    continue
        
        # 2. If we couldn't add a new color (or no unused colors left, which is impossible here),
        #    try a perturbation: recolor an edge to an EXISTING color to change constraints.
        
        e = rng.randint(0, NUM_EDGES - 1)
        old_c = colors[e]
        
        # Pick a random existing color different from old_c
        # If only 1 color used, we can't do this.
        if len(used_colors) > 1:
            candidates = [c for c in used_colors if c != old_c]
            if candidates:
                new_c = rng.choice(candidates)
                if check_edge_change(colors, e, new_c):
                    colors[e] = new_c
                    # used_colors set doesn't change size, but membership might if old_c was unique
                    # However, we don't strictly need to update used_colors for the "add new" logic
                    # unless we want to be precise. 
                    # Actually, if old_c was used only here, it's now removed from used_colors.
                    # If new_c was used elsewhere, it stays.
                    # Let's update used_colors properly.
                    # Count occurrences? Too slow.
                    # Simpler: just maintain the set. 
                    # If we remove old_c, we need to know if it's still used elsewhere.
                    # Given the high density, it's likely used elsewhere.
                    # But to be safe and simple:
                    # We only care about `used_colors` for picking "unused" colors.
                    # If we mistakenly think a color is unused when it is used, we just try to assign it, 
                    # and it will fail the "new color" check? No, check_edge_change doesn't care about "newness".
                    # But if we think a color is unused, we pick it, and assign it. 
                    # If it was actually used, we didn't increase the score.
                    # So it's safe to have a slightly stale used_colors set, 
                    # provided we don't think a color is unused when it IS used? 
                    # Wait. If used_colors is missing a color that IS used, we might pick it as "unused".
                    # We assign it. Score doesn't increase. No harm.
                    # If used_colors contains a color that is NO LONGER used, we won't pick it as "unused".
                    # We might miss the opportunity to "re-use" it? No, we want to ADD new colors.
                    # So it's better to have used_colors be a superset of actual used colors?
                    # Or subset?
                    # If used_colors is a superset, we might avoid picking a color that is actually free? 
                    # No, "unused" means not in used_colors.
                    # If used_colors has X, but X is not actually used, we think X is used. 
                    # We don't pick X. We pick Y (actually unused).
                    # We assign Y. Score increases.
                    # So having extra elements in used_colors is safe (just less efficient).
                    # Having missing elements in used_colors is safe (we might pick a used color, score doesn't increase).
                    
                    # So we can just update used_colors by adding new_c.
                    # We don't remove old_c. This makes used_colors a superset.
                    # This is fine.
                    used_colors.add(new_c)
                    
                    stuck_counter = 0
                    max_stuck = int(max_stuck * 1.1)
        else:
            # If only 1 color used, we can't perturb. Just wait?
            # This shouldn't happen if we start with 1 and add colors.
            # If we are stuck at 1 color, we are in a dead end?
            # No, we can always add a new color to any edge if all edges are 0.
            # Because other edges are 0,0,0. New color C. {0,0,0,C} is not rainbow.
            # So we should never be stuck at 1.
            pass

    # Final safety check
    if count_violations(best_colors) != 0:
        # Fallback to a known valid coloring
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return list(best_colors)

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to make sure the interface matches.
# The task requires: construct() -> 120 integer colors
# The code above provides that.
# EVOLVE-BLOCK-END

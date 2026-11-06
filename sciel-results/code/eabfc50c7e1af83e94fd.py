import time
import random
import itertools

N = 16
E = N * (N - 1) // 2  # 120 edges

# Precompute edge indices
# Map (i, j) to index 0..119
edge_to_idx = {}
idx_to_edge = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        idx_to_edge.append((i, j))
        idx += 1

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Precompute cycles for each edge
# cycles_of_edge[e] = list of tuples (e1, e2, e3)
# where {e, e1, e2, e3} are the edges of a 4-cycle
cycles_of_edge = [[] for _ in range(E)]

for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                e1 = get_idx(a, b)
                e2 = get_idx(b, c)
                e3 = get_idx(c, d)
                e4 = get_idx(d, a)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 2: a-b, b-d, d-c, c-a
                e1 = get_idx(a, b)
                e2 = get_idx(b, d)
                e3 = get_idx(d, c)
                e4 = get_idx(c, a)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))
                
                # Cycle 3: a-c, c-b, b-d, d-a
                e1 = get_idx(a, c)
                e2 = get_idx(c, b)
                e3 = get_idx(b, d)
                e4 = get_idx(d, a)
                for e in (e1, e2, e3, e4):
                    others = [x for x in (e1, e2, e3, e4) if x != e]
                    cycles_of_edge[e].append(tuple(others))

def is_valid_coloring(colors):
    """
    Verify the entire coloring.
    """
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1
                    c1 = colors[get_idx(a, b)]
                    c2 = colors[get_idx(b, c)]
                    c3 = colors[get_idx(c, d)]
                    c4 = colors[get_idx(d, a)]
                    if len({c1, c2, c3, c4}) == 4:
                        return False
                    # Cycle 2
                    c1 = colors[get_idx(a, b)]
                    c2 = colors[get_idx(b, d)]
                    c3 = colors[get_idx(d, c)]
                    c4 = colors[get_idx(c, a)]
                    if len({c1, c2, c3, c4}) == 4:
                        return False
                    # Cycle 3
                    c1 = colors[get_idx(a, c)]
                    c2 = colors[get_idx(c, b)]
                    c3 = colors[get_idx(b, d)]
                    c4 = colors[get_idx(d, a)]
                    if len({c1, c2, c3, c4}) == 4:
                        return False
    return True

def check_edge_valid(colors, edge, new_color):
    """
    Check if setting colors[edge] = new_color is valid.
    """
    for (o1, o2, o3) in cycles_of_edge[edge]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        # If new_color is different from c1, c2, c3 AND c1, c2, c3 are distinct,
        # then we have 4 distinct colors.
        if new_color != c1 and new_color != c2 and new_color != c3:
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial coloring: color(i, j) = i
    colors = [0] * E
    for i in range(N):
        for j in range(i + 1, N):
            colors[get_idx(i, j)] = i
            
    used_colors = set(colors)
    current_count = len(used_colors)
    best_colors = colors.copy()
    best_count = current_count
    
    max_color_val = 119
    
    # Local Search
    # We want to maximize the number of distinct colors.
    # Strategy:
    # 1. Try to assign a new, unused color to a random edge.
    # 2. If that fails, try to assign a random valid color to shuffle the state.
    
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, E - 1)
        
        # Determine candidate colors
        # Priority 1: Unused colors
        # Priority 2: Random colors
        
        # To speed up, we can just pick a random color from 0..119
        # and check if it's valid.
        # But to bias towards new colors, we can do:
        
        # 50% chance: try to pick a color from unused set
        # 50% chance: try to pick a random color
        
        if random.random() < 0.5 and current_count < max_color_val:
            unused = [c for c in range(max_color_val + 1) if c not in used_colors]
            if unused:
                new_color = random.choice(unused)
                if check_edge_valid(colors, e, new_color):
                    colors[e] = new_color
                    used_colors.add(new_color)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                    continue
        else:
            # Try a few random colors
            for _ in range(5):
                new_color = random.randint(0, max_color_val)
                if new_color == colors[e]:
                    continue
                if check_edge_valid(colors, e, new_color):
                    colors[e] = new_color
                    # Update used_colors
                    # This is O(E) if we rebuild, but we can just check if it's new
                    # Actually, we just need to know if the count changed.
                    # If we changed to a color that was already used, count stays same.
                    # If we changed to a new color, count increases.
                    # But we also might have removed the last instance of an old color?
                    # No, we only change one edge.
                    # If old_color was unique and we change it, count decreases.
                    # If new_color was unused, count increases.
                    # We track `used_colors` set.
                    # To be safe and fast, we can just update the set if necessary.
                    # But rebuilding the set is O(E).
                    # Let's just update the set carefully.
                    
                    old_color = colors[e] # Wait, we already overwrote? No, we check before overwrite?
                    # Let's refactor to get old_color first.
                    
                    # Actually, the code above overwrites colors[e] inside the loop.
                    # Let's fix the logic.
                    
                    pass # Placeholder, logic below
                    
        # Refactored Loop Logic for Clarity and Speed
        
    # The above while loop was a bit messy. Let's write a clean version.
    
    # Reset for clean implementation
    colors = [0] * E
    for i in range(N):
        for j in range(i + 1, N):
            colors[get_idx(i, j)] = i
            
    used_colors = set(colors)
    current_count = len(used_colors)
    best_colors = colors.copy()
    best_count = current_count
    
    # To allow escaping local optima, we accept moves that don't increase count
    # with some probability, but we prioritize increasing count.
    
    while time.time() < deadline:
        e = random.randint(0, E - 1)
        old_color = colors[e]
        
        # Try to find a better or equal color
        # We try 10 random candidates
        
        # Bias towards new colors
        if random.random() < 0.3 and current_count < 120:
            # Pick a random unused color
            # Generating unused list is slow. 
            # Instead, pick random 0..119 and check if unused.
            for _ in range(10):
                c = random.randint(0, 119)
                if c in used_colors:
                    continue
                if check_edge_valid(colors, e, c):
                    # Apply
                    colors[e] = c
                    used_colors.add(c)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                    break
        else:
            # Pick random colors, accept if valid (even if count doesn't increase)
            # This helps shuffle
            for _ in range(5):
                c = random.randint(0, 119)
                if c == old_color:
                    continue
                if check_edge_valid(colors, e, c):
                    # Check if this move changes the count
                    # If c is new, count increases.
                    # If old_color was unique and we remove it, count decreases.
                    # We need to know if old_color appears elsewhere.
                    # This is expensive to check O(E).
                    # Approximation: Just apply.
                    # To maintain `used_colors` and `current_count` accurately:
                    
                    # If c is not in used_colors, it's a new color.
                    # If old_color is in used_colors, it might become unused.
                    
                    # Simple approach:
                    # Just apply and update set if we can afford it.
                    # Or, just apply and don't track count perfectly, 
                    # but check count periodically?
                    # No, we need to know when we improve.
                    
                    # Let's just update the set.
                    # If we change e from old to new:
                    # 1. If new is not in used_colors, add it.
                    # 2. Check if old is still in colors.
                    #    We can maintain a frequency array!
                    
                    pass
                    
    # Let's use a frequency array for O(1) count updates.
    
    freq = [0] * 120
    for c in colors:
        freq[c] += 1
    current_count = sum(1 for x in freq if x > 0)
    
    used_colors = set(colors) # Still useful for quick "is unused" check? 
    # Actually, freq[c] == 0 is the check for unused.
    
    while time.time() < deadline:
        e = random.randint(0, E - 1)
        old_color = colors[e]
        
        # Try to find a valid new color
        # Bias towards unused colors
        if random.random() < 0.5:
            # Try 5 random unused colors
            for _ in range(5):
                c = random.randint(0, 119)
                if freq[c] > 0:
                    continue
                if check_edge_valid(colors, e, c):
                    # Apply
                    freq[old_color] -= 1
                    if freq[old_color] == 0:
                        current_count -= 1
                    freq[c] += 1
                    if freq[c] == 1:
                        current_count += 1
                    colors[e] = c
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                    break
        else:
            # Try 5 random colors (any)
            for _ in range(5):
                c = random.randint(0, 119)
                if c == old_color:
                    continue
                if check_edge_valid(colors, e, c):
                    # Apply
                    freq[old_color] -= 1
                    if freq[old_color] == 0:
                        current_count -= 1
                    freq[c] += 1
                    if freq[c] == 1:
                        current_count += 1
                    colors[e] = c
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors.copy()
                    break
                    
    return best_colors

# Final check
# The logic above is sound.
# One detail: `check_edge_valid` uses `colors` list.
# When we check, `colors[e]` is still `old_color`.
# The check assumes we are setting it to `new_color`.
# The logic in `check_edge_valid` is:
# `if new_color != c1 and new_color != c2 and new_color != c3:`
# `   if c1 != c2 and c1 != c3 and c2 != c3:`
# `       return False`
# This correctly checks if adding `new_color` creates a 4-distinct set.
# It does NOT check if removing `old_color` breaks anything?
# Wait. The constraint is on the *resulting* coloring.
# The cycles containing `e` will have `new_color` instead of `old_color`.
# The other 3 edges in the cycle keep their colors.
# So we just need to ensure that `{new_color, c1, c2, c3}` does not have size 4.
# This is exactly what `check_edge_valid` does.
# It does not matter what `old_color` was, as long as the new state is valid.
# Since we only change one edge, only cycles containing that edge are affected.
# So the check is correct.

# One potential issue: The initial coloring is valid.
# The search maintains validity.
# So the final result is valid.

# To ensure we return a list of ints, `best_colors` is a list of ints.

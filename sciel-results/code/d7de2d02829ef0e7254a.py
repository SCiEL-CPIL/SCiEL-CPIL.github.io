import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    def is_valid_change(u, v, new_color, color_matrix):
        # Check if setting color[u][v] = new_color violates any 4-cycle
        # Cycles are (u, x, y, v)
        for x in range(N):
            if x == u or x == v:
                continue
            c_ux = color_matrix[u][x]
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                c_xy = color_matrix[x][y]
                c_yv = color_matrix[y][v]
                
                # The 4 colors are: new_color, c_ux, c_xy, c_yv
                # They must NOT be all distinct.
                # So at least two must be equal.
                if new_color == c_ux or new_color == c_xy or new_color == c_yv:
                    continue
                if c_ux == c_xy or c_ux == c_yv or c_xy == c_yv:
                    continue
                return False
        return True

    def generate_base(a, b, m):
        colors = [0] * (N * (N - 1) // 2)
        cm = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c = (a * i + b * j) % m
                colors[get_idx(i, j)] = c
                cm[i][j] = c
                cm[j][i] = c
        
        # Verify validity
        valid = True
        for i in range(N):
            if not valid:
                break
            for j in range(i + 1, N):
                if not valid:
                    break
                c1 = cm[i][j]
                for x in range(N):
                    if x == i or x == j:
                        continue
                    c2 = cm[i][x]
                    for y in range(N):
                        if y == i or y == j or y == x:
                            continue
                        c3 = cm[x][y]
                        c4 = cm[y][j]
                        if len({c1, c2, c3, c4}) == 4:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        return colors, cm, valid

    # Try a few base constructions
    best_colors = None
    best_cm = None
    best_num_colors = 0
    
    # Candidates for (a, b, m)
    # We want small m to ensure validity initially, but large enough for variety?
    # Actually, small m means fewer colors. We want to start with a valid one and expand.
    # m=7, 11, 13 are primes, good for mixing.
    candidates = [
        (1, 1, 7),
        (1, 2, 7),
        (1, 3, 7),
        (2, 1, 7),
        (1, 1, 11),
        (1, 2, 11),
        (1, 3, 11),
        (1, 1, 13),
        (1, 2, 13),
        (1, 1, 5),
        (1, 2, 5),
        (1, 1, 3),
        (1, 2, 3),
    ]
    
    for a, b, m in candidates:
        if time.time() > deadline - 10:
            break
        colors, cm, valid = generate_base(a, b, m)
        if valid:
            num_colors = len(set(colors))
            if num_colors > best_num_colors:
                best_num_colors = num_colors
                best_colors = colors.copy()
                best_cm = [row[:] for row in cm]

    if best_colors is None:
        # Fallback: trivial valid coloring
        best_colors = [0] * (N * (N - 1) // 2)
        best_cm = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                c = i % 15
                best_colors[get_idx(i, j)] = c
                best_cm[i][j] = c
                best_cm[j][i] = c
        best_num_colors = len(set(best_colors))

    colors = best_colors
    cm = best_cm
    used_colors = set(colors)
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    random.seed(42)
    
    # Local Search: Try to introduce new colors
    # We iterate through edges. For each edge, if we can change it to an UNUSED color
    # that is valid, we do so.
    
    # To make it more effective, we can also try to change to any valid color
    # if it helps "unlock" something, but adding new colors is the primary goal.
    
    # Optimization: Precompute neighbors?
    # The check is local to the edge's incident 4-cycles.
    
    passes = 0
    max_passes = 100000
    
    while time.time() < deadline - 5 and passes < max_passes:
        passes += 1
        changed = False
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            
            old_color = cm[u][v]
            
            # Find valid colors for this edge
            # We prefer unused colors
            valid_unused = []
            valid_used = []
            
            # We don't need to check ALL 120 colors.
            # We can check unused colors first.
            # If we find a valid unused color, we take it.
            # If not, we might take a valid used color to shuffle state.
            
            # List of unused colors
            # To speed up, we can just iterate through 0..119
            # But checking 120 colors * 182 pairs * 6 checks is expensive.
            
            # Better: 
            # 1. Try a few random unused colors.
            # 2. If none work, try a few random used colors.
            
            # Actually, let's just iterate through a subset of candidates.
            # Candidate pool: 
            # - All unused colors (prioritized)
            # - Current color
            # - Random used colors
            
            # Let's try to pick a random unused color and see if it's valid.
            # If we have many unused colors, we can sample.
            
            unused_list = [c for c in range(120) if c not in used_colors]
            
            if unused_list:
                # Try up to 5 random unused colors
                random.shuffle(unused_list)
                found = False
                for c_new in unused_list[:5]:
                    if is_valid_change(u, v, c_new, cm):
                        # Apply
                        cm[u][v] = c_new
                        cm[v][u] = c_new
                        colors[get_idx(u, v)] = c_new
                        used_colors.add(c_new)
                        changed = True
                        found = True
                        break
                
                if not found:
                    # If no unused color worked, maybe try to change to a different used color?
                    # This might help if the current color is "blocking" others.
                    # But it doesn't increase the count. Let's skip for now to keep it simple and fast.
                    pass
            else:
                break # All colors used, max possible
        
        if not changed:
            # If we made no progress in a pass, we might be stuck.
            # We can try a "restart" of the local search phase with a different random seed
            # or just break if we've been stuck for a while.
            # Given the structure, breaking is safer to avoid wasting time.
            # However, sometimes shuffling used colors helps.
            # Let's try one more pass with random used color swaps if stuck.
            stuck_count = 0
            # Simple heuristic: if stuck, break. The base construction + greedy add is usually strong.
            break

    return colors

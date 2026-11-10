"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    # Build edge index: edge (i,j) -> index in lexicographic order
    edges = []
    edge_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    m = len(edges)  # 120 edges
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        e_idx = []
        for a, b in combinations(quad, 2):
            e_idx.append(edge_idx[(a, b)])
        k4s.append(e_idx)
    
    # Seed coloring: (i+j) % 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    num_colors = 16
    
    def count_used_colors(c):
        return len(set(c))
    
    def is_valid(c):
        """Check if every K4 has at least 5 distinct colors."""
        for e_idx in k4s:
            if len(set(c[i] for i in e_idx)) < 5:
                return False
        return True
    
    def k4_violation_count(c):
        """Count how many K4s have fewer than 5 colors."""
        count = 0
        for e_idx in k4s:
            if len(set(c[i] for i in e_idx)) < 5:
                count += 1
        return count
    
    def try_reduce_colors(c, max_time=None):
        """Try to reduce the number of colors by merging color classes."""
        if max_time is None:
            max_time = deadline
        cur_colors = list(c)
        cur_nc = count_used_colors(cur_colors)
        
        # Try to eliminate one color at a time
        for _ in range(50):  # up to 50 elimination attempts
            if time.time() > max_time:
                break
            if cur_nc <= 13:
                break
            
            used = set(cur_colors)
            # Try each color in descending order of frequency (harder to eliminate)
            # Actually try least frequent first (easier to reassign)
            from collections import Counter
            freq = Counter(cur_colors)
            target_color = min(used, key=lambda x: freq[x])
            
            # Find all edges with target_color
            target_edges = [i for i in range(m) if cur_colors[i] == target_color]
            
            # Try to reassign each target edge to another color
            success = True
            new_colors = list(cur_colors)
            for ei in target_edges:
                if time.time() > max_time:
                    success = False
                    break
                # Try all other colors
                other_colors = [x for x in used if x != target_color]
                random.shuffle(other_colors)
                placed = False
                for nc in other_colors:
                    new_colors[ei] = nc
                    if k4_violation_count(new_colors) == 0:
                        placed = True
                        break
                if not placed:
                    # Try target_color itself (keep it)
                    new_colors[ei] = target_color
                    # This means we can't fully eliminate this color
                    success = False
                    break
            
            if success:
                cur_colors = new_colors
                cur_nc = count_used_colors(cur_colors)
            else:
                # Try a different target
                pass
        
        return cur_colors, cur_nc
    
    # First verify seed is valid
    assert is_valid(colors), "Seed coloring is invalid!"
    
    # Try to reduce colors
    best_colors = list(colors)
    best_nc = num_colors
    
    # Try reducing from 16 down
    for target_nc in range(num_colors - 1, 12, -1):
        if time.time() > deadline:
            break
        # Attempt local search to get to target_nc colors
        attempt_colors, attempt_nc = try_reduce_colors(list(colors), max_time=deadline)
        if attempt_nc < best_nc:
            best_colors = attempt_colors
            best_nc = attempt_nc
        if best_nc <= 13:
            break
    
    # Additional local search: try random recoloring to reduce colors
    if time.time() < deadline - 10:
        cur = list(best_colors)
        cur_nc = count_used_colors(cur)
        for iteration in range(200):
            if time.time() > deadline:
                break
            if cur_nc <= 13:
                break
            # Pick a random edge and try to change its color
            ei = random.randint(0, m - 1)
            old_c = cur[ei]
            used = set(cur)
            # Try to assign a color that reduces the palette
            # First try colors already in use
            candidates = list(used)
            random.shuffle(candidates)
            improved = False
            for nc in candidates:
                if nc == old_c:
                    continue
                cur[ei] = nc
                new_nc = count_used_colors(cur)
                if new_nc < cur_nc and k4_violation_count(cur) == 0:
                    cur_nc = new_nc
                    improved = True
                    break
                elif k4_violation_count(cur) == 0:
                    # Keep it if valid even if not reducing
                    if random.random() < 0.1:
                        improved = True
                        break
            if not improved:
                cur[ei] = old_c
        
        if count_used_colors(cur) < best_nc and is_valid(cur):
            best_colors = cur
            best_nc = count_used_colors(cur)
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback to seed
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        best_nc = 16
    
    return best_colors

# EVOLVE-BLOCK-END

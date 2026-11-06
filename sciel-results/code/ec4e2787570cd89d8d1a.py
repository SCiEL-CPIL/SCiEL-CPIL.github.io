import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    
    num_aps = len(aps)
    
    def count_violations(colors):
        viol = 0
        for i in range(num_aps):
            c0 = colors[aps[i][0]]
            c1 = colors[aps[i][1]]
            c2 = colors[aps[i][2]]
            c3 = colors[aps[i][3]]
            # Check if < 3 distinct
            if c0 == c1 == c2 == c3:
                viol += 1
            elif c0 == c1 == c2 and c0 != c3:
                viol += 1
            elif c0 == c1 == c3 and c0 != c2:
                viol += 1
            elif c0 == c2 == c3 and c0 != c1:
                viol += 1
            elif c1 == c2 == c3 and c1 != c0:
                viol += 1
            elif c0 == c2 and c1 == c3 and c0 != c1:
                viol += 1
            elif c0 == c3 and c1 == c2 and c0 != c1:
                viol += 1
            elif c0 == c1 and c2 == c3 and c0 != c2:
                viol += 1
        return viol
    
    # Try to find a valid 3-coloring first
    best_colors = None
    best_num_colors = n  # worst case
    
    # Try random 3-colorings
    for attempt in range(5000):
        if time.time() > deadline - 10:
            break
        colors = [random.randint(0, 2) for _ in range(n)]
        v = count_violations(colors)
        if v == 0:
            best_colors = colors
            best_num_colors = 3
            break
    
    if best_colors is None:
        # Try 4-colorings
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            colors = [random.randint(0, 3) for _ in range(n)]
            v = count_violations(colors)
            if v == 0:
                best_colors = colors
                best_num_colors = 4
                break
    
    if best_colors is None:
        # Fallback: use modular pattern
        best_colors = [i % 4 for i in range(n)]
        best_num_colors = 4
    
    # Try to reduce colors further via local search
    current = best_colors[:]
    
    # Local search: try to merge colors
    for _ in range(200000):
        if time.time() > deadline - 5:
            break
        
        pos = random.randint(0, n - 1)
        old = current[pos]
        
        # Try to change to a color already in use (to not increase color count)
        colors_in_use = set(current)
        other_colors = [c for c in colors_in_use if c != old]
        
        if not other_colors:
            continue
        
        new = random.choice(other_colors)
        current[pos] = new
        
        v = count_violations(current)
        if v == 0:
            num_colors = len(set(current))
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = current[:]
        else:
            current[pos] = old
    
    # Relabel to consecutive
    unique = sorted(set(best_colors))
    mapping = {old: new for new, old in enumerate(unique)}
    result = [mapping[c] for c in best_colors]
    
    return result

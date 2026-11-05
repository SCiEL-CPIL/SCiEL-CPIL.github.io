import time
import random
import itertools

# Precompute K4 subgraphs
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

K4s = []
for v1, v2, v3, v4 in itertools.combinations(range(16), 4):
    edges = (
        edge_idx(v1, v2),
        edge_idx(v1, v3),
        edge_idx(v1, v4),
        edge_idx(v2, v3),
        edge_idx(v2, v4),
        edge_idx(v3, v4)
    )
    K4s.append(edges)

edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4_edges in enumerate(K4s):
    for e in k4_edges:
        edge_to_k4s[e].append(k4_idx)

def count_distinct_colors(colors, k4_edges):
    # Unpack for speed
    c0, c1, c2, c3, c4, c5 = (colors[e] for e in k4_edges)
    # Manual distinct count for 6 items
    # If all same
    if c0 == c1 and c0 == c2 and c0 == c3 and c0 == c4 and c0 == c5:
        return 1
    # If 2 distinct
    # ... this is getting complex. Set is fast enough for 6 items.
    return len({c0, c1, c2, c3, c4, c5})

def get_violation_count(colors):
    total = 0
    for k4 in K4s:
        d = count_distinct_colors(colors, k4)
        if d < 5:
            total += (5 - d)
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Known valid solution with 17 colors
    best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_k = 17
    
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        found_valid = False
        # Multiple restarts
        # Limit time per restart
        max_time_per_restart = 2.0
        
        while time.time() < deadline - 5:
            restart_start = time.time()
            
            # Initialize random colors
            colors = [random.randint(0, K - 1) for _ in range(120)]
            current_violations = get_violation_count(colors)
            
            if current_violations == 0:
                best_colors = colors
                best_k = K
                found_valid = True
                break
            
            # Local Search
            no_improve_count = 0
            last_improve_time = time.time()
            
            while time.time() < restart_start + max_time_per_restart and time.time() < deadline - 5:
                if current_violations == 0:
                    best_colors = colors
                    best_k = K
                    found_valid = True
                    break
                
                # Pick a random violating K4
                # Find one
                violating_k4_idx = -1
                for idx, k4 in enumerate(K4s):
                    if count_distinct_colors(colors, k4) < 5:
                        violating_k4_idx = idx
                        break
                
                if violating_k4_idx == -1:
                    # Should not happen if current_violations > 0
                    best_colors = colors
                    best_k = K
                    found_valid = True
                    break
                
                k4 = K4s[violating_k4_idx]
                e_idx = random.choice(k4)
                old_color = colors[e_idx]
                
                # Try changing to a random color
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta = 0
                affected_k4s = edge_to_k4s[e_idx]
                for k4_idx in affected_k4s:
                    old_d = count_distinct_colors(colors, K4s[k4_idx])
                    # Simulate
                    colors[e_idx] = new_color
                    new_d = count_distinct_colors(colors, K4s[k4_idx])
                    old_v = max(0, 5 - old_d)
                    new_v = max(0, 5 - new_d)
                    delta += (new_v - old_v)
                
                if delta < 0:
                    # Accept
                    current_violations += delta
                    no_improve_count = 0
                    last_improve_time = time.time()
                elif delta == 0:
                    # Accept with prob 0.5
                    if random.random() < 0.5:
                        current_violations += delta
                        no_improve_count += 1
                    else:
                        colors[e_idx] = old_color # Revert
                        no_improve_count += 1
                else:
                    # Reject
                    colors[e_idx] = old_color # Revert
                    no_improve_count += 1
                
                if no_improve_count > 500:
                    # Restart this local search iteration
                    break

        if found_valid:
            break

    return best_colors

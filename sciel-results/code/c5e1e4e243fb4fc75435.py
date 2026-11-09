import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)
    
    # Map (u,v) to edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_validity(colors):
        for k4_idx in range(num_k4s):
            e_list = k4s[k4_idx]
            cset = set(colors[ei] for ei in e_list)
            if len(cset) < 5:
                return False
        return True
    
    def init_penalty_state(colors):
        k4_color_counts = [{} for _ in range(num_k4s)]
        penalty = 0
        for k4_idx in range(num_k4s):
            e_list = k4s[k4_idx]
            counts = {}
            for ei in e_list:
                c = colors[ei]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            if len(counts) < 5:
                penalty += 1
        return k4_color_counts, penalty
    
    def get_delta_penalty(e_idx, old_c, new_c, k4_color_counts):
        delta = 0
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            old_distinct = len(counts)
            
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            
            if new_c not in counts:
                counts[new_c] = 0
            counts[new_c] += 1
            
            new_distinct = len(counts)
            
            old_bad = old_distinct < 5
            new_bad = new_distinct < 5
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
        
        return delta
    
    def apply_change(colors, e_idx, old_c, new_c, k4_color_counts):
        colors[e_idx] = new_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            if new_c not in counts:
                counts[new_c] = 0
            counts[new_c] += 1
    
    def revert_change(colors, e_idx, old_c, new_c, k4_color_counts):
        colors[e_idx] = old_c
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            counts[new_c] -= 1
            if counts[new_c] == 0:
                del counts[new_c]
            if old_c not in counts:
                counts[old_c] = 0
            counts[old_c] += 1
    
    best_penalty = float('inf')
    best_count = float('inf')
    best_colors = None
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # (i+j) mod p
    for p in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i + j) % p
        initial_solutions.append(init)
    
    # (i-j) mod p
    for p in [13, 14, 15, 16, 17, 18, 19]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i - j) % p
        initial_solutions.append(init)
    
    # (i*j) mod p
    for p in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i * j) % p
        initial_solutions.append(init)
    
    # (i + 2*j) mod p
    for p in [13, 14, 15, 16, 17, 18, 19]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i + 2 * j) % p
        initial_solutions.append(init)
    
    # (i + 3*j) mod p
    for p in [13, 14, 15, 16, 17]:
        init = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            init[idx] = (i + 3 * j) % p
        initial_solutions.append(init)
    
    # Random initial solutions with small palette
    random.seed(42)
    for _ in range(5):
        palette = random.randint(13, 18)
        init = [random.randint(0, palette - 1) for _ in range(num_edges)]
        initial_solutions.append(init)
    
    # Check all initial solutions, keep the best
    for init_sol in initial_solutions:
        if time.time() > deadline - 20.0:
            break
        nc = count_colors(init_sol)
        if nc < best_count:
            valid = check_validity(init_sol)
            if valid:
                best_count = nc
                best_colors = init_sol[:]
                best_penalty = 0
                if nc <= 13:
                    break
            elif nc < best_count:
                # Track invalid but low-color solutions for SA starting point
                pass
    
    # If we found a valid solution with <= 13 colors, we're done
    if best_colors is not None and best_penalty == 0 and best_count <= 13:
        return compact(best_colors)[0]
    
    # Simulated annealing: try multiple restarts
    T_max = 8.0
    T_min = 0.02
    cooling_rate = 0.9997
    
    # Use the best initial solution as starting point for SA
    if best_colors is not None and best_penalty == 0:
        # Already valid, just try to reduce colors
        start_sol = best_colors[:]
    else:
        # Pick the initial solution with fewest colors (even if invalid)
        min_nc = float('inf')
        best_init = None
        for init_sol in initial_solutions:
            nc = count_colors(init_sol)
            if nc < min_nc:
                min_nc = nc
                best_init = init_sol[:]
        if best_init is not None:
            start_sol = best_init
        else:
            start_sol = [random.randint(0, 14) for _ in range(num_edges)]
    
    colors = start_sol[:]
    k4_color_counts, current_penalty = init_penalty_state(colors)
    current_count = count_colors(colors)
    
    # Update best
    if current_penalty < best_penalty or (current_penalty == best_penalty and current_count < best_count):
        best_penalty = current_penalty
        best_count = current_count
        best_colors = colors[:]
    
    T = T_max
    while T > T_min and time.time() < deadline - 15.0:
        e_idx = random.randint(0, num_edges - 1)
        old_c = colors[e_idx]
        
        # Choose new color: bias toward reducing palette
        if current_penalty > 0:
            # Focus on fixing violations: try colors that appear in neighboring K4s
            if random.random() < 0.5:
                new_c = random.randint(0, 14)
            else:
                # Try a color from a random K4 containing this edge
                if edge_k4s[e_idx]:
                    k4_idx = random.choice(edge_k4s[e_idx])
                    e_list = k4s[k4_idx]
                    neighbor_colors = [colors[ei] for ei in e_list if ei != e_idx]
                    if neighbor_colors:
                        new_c = random.choice(neighbor_colors)
                    else:
                        new_c = random.randint(0, 14)
                else:
                    new_c = random.randint(0, 14)
        else:
            # Penalty is 0: try to reduce color count
            used_colors = set(colors)
            if len(used_colors) > 13:
                # Try to replace old_c with another used color
                candidates = [c for c in used_colors if c != old_c]
                if candidates:
                    new_c = random.choice(candidates)
                else:
                    new_c = random.randint(0, 14)
            else:
                new_c = random.randint(0, 14)
        
        if new_c == old_c:
            T *= cooling_rate
            continue
        
        delta = get_delta_penalty(e_idx, old_c, new_c, k4_color_counts)
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            apply_change(colors, e_idx, old_c, new_c, k4_color_counts)
            current_penalty += delta
            new_count = count_colors(colors)
            
            if current_penalty < best_penalty or (current_penalty == best_penalty and new_count < best_count):
                best_penalty = current_penalty
                best_count = new_count
                best_colors = colors[:]
                if best_penalty == 0 and best_count <= 13:
                    break
        T *= cooling_rate
    
    # Final polish: if penalty is 0, aggressively reduce colors
    if best_colors is not None and best_penalty == 0 and best_count > 13:
        colors = best_colors[:]
        k4_color_counts, current_penalty = init_penalty_state(colors)
        current_count = best_count
        
        polish_steps = 0
        while current_count > 13 and time.time() < deadline - 5.0:
            # Find edges with colors that appear only once (or few times)
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Target colors with lowest count
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
            target_color = sorted_colors[0][0]
            
            # Find an edge with this color
            target_edges = [i for i, c in enumerate(colors) if c == target_color]
            if not target_edges:
                break
            
            e_idx = random.choice(target_edges)
            old_c = colors[e_idx]
            
            # Try to change to a more common color
            used_colors = [c for c in color_counts if c != old_c]
            if not used_colors:
                break
            new_c = random.choice(used_colors)
            
            delta = get_delta_penalty(e_idx, old_c, new_c, k4_color_counts)
            if delta <= 0:
                apply_change(colors, e_idx, old_c, new_c, k4_color_counts)
                current_penalty += delta
                new_count = count_colors(colors)
                if new_count < best_count:
                    best_count = new_count
                    best_colors = colors[:]
                    current_count = new_count
                polish_steps += 1
            else:
                # Try another edge with same color
                continue
        
        # Recheck validity after polish
        if check_validity(colors):
            final_count = count_colors(colors)
            if final_count < best_count:
                best_count = final_count
                best_colors = colors[:]
    
    if best_colors is None:
        # Fallback: use (i+j) % 17
        best_colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            best_colors[idx] = (i + j) % 17
    
    return compact(best_colors)[0]

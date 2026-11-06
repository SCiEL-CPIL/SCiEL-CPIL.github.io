import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles and their edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        # Cycle 2: a-b-d-c-a
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        # Cycle 3: a-c-b-d-a
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    # Map edge index to list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # Vectorized validity checking
    def is_valid_change_vec(colors, edge, new_color):
        """Check if changing edge to new_color is valid using vectorized operations."""
        c_indices = edge_to_cycles[edge]
        if not c_indices:
            return True
        
        # Get all edges in these cycles
        cycle_edges = [cycles[i] for i in c_indices]
        all_edges = set()
        for c in cycle_edges:
            all_edges.update(c)
        
        # Get colors of all edges except the one being changed
        colors_arr = np.array([colors[e] for e in all_edges])
        edge_arr = np.array(sorted(all_edges))
        
        # Build lookup
        color_lookup = {e: colors[e] for e in all_edges}
        
        # Check each cycle
        for c in cycle_edges:
            cols = []
            for e in c:
                if e == edge:
                    cols.append(new_color)
                else:
                    cols.append(color_lookup[e])
            if len(set(cols)) == 4:
                return False
        return True

    # Faster validity check using precomputed structures
    def is_valid_change_fast(colors, edge, new_color):
        for c_idx in edge_to_cycles[edge]:
            c = cycles[c_idx]
            # Check if this cycle would have 4 distinct colors
            col_set = set()
            for e in c:
                if e == edge:
                    col_set.add(new_color)
                else:
                    col_set.add(colors[e])
                if len(col_set) > 3:
                    return False
        return True

    # Generate seed colorings using recursive partitioning
    def get_seed_colors(seed_val):
        random.seed(seed_val)
        colors = [0] * NUM_EDGES
        color_id = 0
        
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 2:
                return
            # Vary split based on randomness
            if len(vertices) == 3:
                mid = 1
            else:
                mid = len(vertices) // 2 + random.randint(-1, 1)
                mid = max(1, min(mid, len(vertices) - 1))
            
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        
        process_partition(list(range(N)))
        return colors, color_id

    # Try multiple seeds and keep best
    best_seed = None
    best_seed_score = 0
    for s in range(10):
        seed_colors, num_colors = get_seed_colors(42 + s)
        score = len(set(seed_colors))
        if score > best_seed_score:
            best_seed_score = score
            best_seed = seed_colors
    
    current_colors = best_seed.copy()
    best_score = best_seed_score
    best_colors = best_seed.copy()
    
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
    
    random.seed(42)
    
    used_colors = set(current_colors)
    current_score = len(used_colors)
    
    temperature = 3.0
    cooling_rate = 0.99999
    min_temp = 0.005
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 1500
    
    # Precompute edge cycle counts for weighting
    edge_cycle_count = np.array([len(edge_to_cycles[e]) for e in range(NUM_EDGES)])
    
    # Precompute adjacency for faster constraint checking
    # For each edge, precompute which other edges share a cycle with it
    edge_conflicts = [[] for _ in range(NUM_EDGES)]
    for e1 in range(NUM_EDGES):
        for c_idx in edge_to_cycles[e1]:
            c = cycles[c_idx]
            for e2 in c:
                if e2 != e1 and e2 not in edge_conflicts[e1]:
                    edge_conflicts[e1].append(e2)
    
    while time.time() < deadline:
        steps += 1
        
        # Select edge to recolor
        # Weight towards less constrained edges and rare colors
        weights = np.ones(NUM_EDGES)
        weights *= (1.0 / (1.0 + edge_cycle_count))
        weights *= (1.0 / (1.0 + np.array(freq)[current_colors]))
        total_w = weights.sum()
        r = random.random() * total_w
        cumulative = 0
        e = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                e = i
                break
        
        old_color = current_colors[e]
        
        # Choose new color - bias toward unused colors
        if current_score < MAX_COLORS:
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if unused and random.random() < 0.9:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
        else:
            new_color = random.randint(0, MAX_COLORS - 1)
        
        if new_color == old_color:
            continue
            
        # Check validity
        if is_valid_change_fast(current_colors, e, new_color):
            old_score = current_score
            score_change = 0
            if freq[old_color] == 1:
                score_change -= 1
            if freq[new_color] == 0:
                score_change += 1
            new_score = old_score + score_change
            
            # Acceptance criteria
            if new_score > current_score:
                accept = True
            elif new_score == current_score:
                accept = random.random() < 0.5
            else:
                delta = current_score - new_score
                accept = random.random() < np.exp(-delta / max(temperature, 0.001))
            
            if accept:
                freq[old_color] -= 1
                freq[new_color] += 1
                current_colors[e] = new_color
                
                if freq[old_color] == 0:
                    used_colors.discard(old_color)
                if freq[new_color] == 1:
                    used_colors.add(new_color)
                current_score = len(used_colors)
                
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current_colors.copy()
                    no_improve_steps = 0
                    temperature = min(3.5, temperature * 1.2)
                else:
                    no_improve_steps += 1
                
                temperature *= cooling_rate
                if temperature < min_temp:
                    temperature = min_temp
                    
                # Diversification
                if no_improve_steps > max_no_improve:
                    current_colors = best_colors.copy()
                    freq = [0] * MAX_COLORS
                    for c in current_colors:
                        freq[c] += 1
                    used_colors = set(current_colors)
                    current_score = len(used_colors)
                    temperature = 3.0
                    no_improve_steps = 0
                    
                    # Perturbation
                    num_perturb = random.randint(8, 15)
                    perturb_edges = random.sample(range(NUM_EDGES), num_perturb)
                    for pe in perturb_edges:
                        if current_score < MAX_COLORS:
                            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                            if unused and random.random() < 0.8:
                                pc = random.choice(unused)
                            else:
                                pc = random.randint(0, MAX_COLORS - 1)
                        else:
                            pc = random.randint(0, MAX_COLORS - 1)
                        
                        if pc != current_colors[pe] and is_valid_change_fast(current_colors, pe, pc):
                            old_c = current_colors[pe]
                            freq[old_c] -= 1
                            freq[pc] += 1
                            current_colors[pe] = pc
                            if freq[old_c] == 0:
                                used_colors.discard(old_c)
                            if freq[pc] == 1:
                                used_colors.add(pc)
                            current_score = len(used_colors)

    # Final polish: greedy color insertion
    if time.time() < deadline - 0.5:
        for polish_pass in range(5):
            if time.time() > deadline - 0.3:
                break
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if not unused:
                break
            
            # Sort edges by constraint count
            edge_order = sorted(range(NUM_EDGES), key=lambda e: edge_cycle_count[e])
            for new_c in unused:
                if time.time() > deadline - 0.3:
                    break
                for e in edge_order:
                    if current_colors[e] != new_c and is_valid_change_fast(current_colors, e, new_c):
                        old_c = current_colors[e]
                        freq[old_c] -= 1
                        freq[new_c] += 1
                        current_colors[e] = new_c
                        if freq[old_c] == 0:
                            used_colors.discard(old_c)
                        if freq[new_c] == 1:
                            used_colors.add(new_c)
                        current_score = len(used_colors)
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = current_colors.copy()
                        break

    # Final aggressive search for unused colors
    if time.time() < deadline - 0.2:
        unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
        for new_c in unused:
            if time.time() > deadline - 0.1:
                break
            for e in range(NUM_EDGES):
                if current_colors[e] != new_c and is_valid_change_fast(current_colors, e, new_c):
                    old_c = current_colors[e]
                    freq[old_c] -= 1
                    freq[new_c] += 1
                    current_colors[e] = new_c
                    if freq[old_c] == 0:
                        used_colors.discard(old_c)
                    if freq[new_c] == 1:
                        used_colors.add(new_c)
                    current_score = len(used_colors)
                    if current_score > best_score:
                        best_score = current_score
                        best_colors = current_colors.copy()
                    break

    return best_colors

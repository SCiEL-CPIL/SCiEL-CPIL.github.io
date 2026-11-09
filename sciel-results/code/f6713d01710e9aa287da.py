import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
ALL_COLORS = set(range(120))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    edge_to_other_edges = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for k in range(4):
            e_idx = cyc[k]
            others = [cyc[j] for j in range(4) if j != k]
            edge_to_other_edges[e_idx].append(tuple(others))

    def is_valid(colors):
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))
        
    random.seed(42)
    
    def get_allowed_colors(colors, e_idx):
        allowed = None
        for others in edge_to_other_edges[e_idx]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            
            if c1 != c2 and c1 != c3 and c2 != c3:
                s = {c1, c2, c3}
                if allowed is None:
                    allowed = s
                else:
                    allowed = allowed.intersection(s)
                    if not allowed:
                        break
        
        if allowed is None:
            return ALL_COLORS
        return allowed

    def greedy_construct():
        colors = [-1] * NUM_EDGES
        used_colors = set()
        
        edges_order = list(range(NUM_EDGES))
        random.shuffle(edges_order)
        
        for e_idx in edges_order:
            allowed = None
            for others in edge_to_other_edges[e_idx]:
                c1 = colors[others[0]]
                c2 = colors[others[1]]
                c3 = colors[others[2]]
                
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue
                
                if c1 != c2 and c1 != c3 and c2 != c3:
                    s = {c1, c2, c3}
                    if allowed is None:
                        allowed = s
                    else:
                        allowed = allowed.intersection(s)
                        if not allowed:
                            break
            
            if allowed is not None and not allowed:
                new_color = random.randint(0, 119)
            elif allowed is not None:
                unused_in_allowed = [c for c in allowed if c not in used_colors]
                if unused_in_allowed:
                    new_color = random.choice(unused_in_allowed)
                else:
                    new_color = random.choice(list(allowed))
            else:
                if len(used_colors) < 120:
                    unused = [c for c in range(120) if c not in used_colors]
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, 119)
            
            colors[e_idx] = new_color
            used_colors.add(new_color)
            
        return colors

    best_score = -1
    best_colors = None
    
    # Try greedy construction
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        colors = greedy_construct()
        if is_valid(colors):
            score = count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]

    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_score = 1

    current = best_colors[:]
    current_score = best_score
    current_used = set(current)
    
    stagnation_counter = 0
    max_stagnation_passes = 10
    
    while time.time() < deadline - 5:
        improved_in_pass = False
        
        edges_order = list(range(NUM_EDGES))
        random.shuffle(edges_order)
        
        for e_idx in edges_order:
            if time.time() > deadline - 5:
                break
                
            allowed = get_allowed_colors(current, e_idx)
            if not allowed:
                continue
                
            current_color = current[e_idx]
            
            new_color = None
            candidates = [c for c in allowed if c not in current_used]
            
            if candidates:
                new_color = random.choice(candidates)
            else:
                new_color = random.choice(list(allowed))
                
            if new_color == current_color:
                continue
                
            old_color = current[e_idx]
            current[e_idx] = new_color
            
            new_used = set(current)
            new_score = len(new_used)
            
            if new_score > current_score:
                current_score = new_score
                current_used = new_used
                improved_in_pass = True
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current[:]
            elif new_score == current_score:
                current_used = new_used
                improved_in_pass = True
            else:
                current[e_idx] = old_color
                
        if not improved_in_pass:
            break
            
        if current_score > best_score:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter > max_stagnation_passes:
            for _ in range(10):
                e_idx = random.randint(0, NUM_EDGES - 1)
                allowed = get_allowed_colors(current, e_idx)
                if allowed:
                    current[e_idx] = random.choice(list(allowed))
            
            current_used = set(current)
            current_score = len(current_used)
            
            if current_score > best_score:
                best_score = current_score
                best_colors = current[:]
            
            stagnation_counter = 0

    return best_colors

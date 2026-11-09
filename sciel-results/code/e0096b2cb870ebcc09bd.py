import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    cycles = []
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        cycles.append([e12, e23, e34, e14])
        cycles.append([e12, e24, e34, e13])
        cycles.append([e13, e23, e24, e14])
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    
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
    
    def get_allowed_colors(colors, e_idx):
        allowed = None
        for ci in edge_to_cycles[e_idx]:
            cyc = cycles[ci]
            others = [cyc[j] for j in range(4) if cyc[j] != e_idx]
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            
            if c1 != c2 and c1 != c3 and c2 != c3:
                s = {c1, c2, c3}
                if allowed is None:
                    allowed = set(s)
                else:
                    allowed = allowed & s
                    if not allowed:
                        break
        
        if allowed is None:
            return set(range(120))
        return allowed
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        colors = [-1] * NUM_EDGES
        used_colors = set()
        
        edges_order = list(range(NUM_EDGES))
        rng.shuffle(edges_order)
        
        for e_idx in edges_order:
            allowed = None
            for ci in edge_to_cycles[e_idx]:
                cyc = cycles[ci]
                others = [cyc[j] for j in range(4) if cyc[j] != e_idx]
                c1 = colors[others[0]]
                c2 = colors[others[1]]
                c3 = colors[others[2]]
                
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue
                
                if c1 != c2 and c1 != c3 and c2 != c3:
                    s = {c1, c2, c3}
                    if allowed is None:
                        allowed = set(s)
                    else:
                        allowed = allowed & s
                        if not allowed:
                            break
            
            if allowed is not None and not allowed:
                new_color = rng.randint(0, 119)
            elif allowed is not None:
                unused_in_allowed = [c for c in allowed if c not in used_colors]
                if unused_in_allowed:
                    new_color = rng.choice(unused_in_allowed)
                else:
                    new_color = rng.choice(list(allowed))
            else:
                if len(used_colors) < 120:
                    unused = [c for c in range(120) if c not in used_colors]
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, 119)
            
            colors[e_idx] = new_color
            used_colors.add(new_color)
        
        return colors
    
    best_score = 0
    best_colors = None
    
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        colors = greedy_construct(seed)
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
    
    # Local search to improve
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        edges_order = list(range(NUM_EDGES))
        random.shuffle(edges_order)
        
        for e_idx in edges_order:
            if time.time() > deadline - 5:
                break
            
            allowed = get_allowed_colors(current, e_idx)
            if not allowed:
                continue
            
            current_color = current[e_idx]
            
            candidates = [c for c in allowed if c not in current_used]
            if candidates:
                new_color = random.choice(candidates)
            else:
                continue
            
            if new_color == current_color:
                continue
            
            current[e_idx] = new_color
            current_used.add(new_color)
            new_score = len(current_used)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current[:]
            else:
                current[e_idx] = current_color
                current_used.discard(new_color)
        
        if not improved:
            break
    
    return best_colors

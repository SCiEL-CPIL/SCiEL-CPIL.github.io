import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
            
    cycles = []
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1
                    e1, e2, e3, e4 = edge_idx[(a, b)], edge_idx[(b, c)], edge_idx[(c, d)], edge_idx[(d, a)]
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4): cycles_by_edge[e].append(len(cycles) - 1)
                    # Cycle 2
                    e1, e2, e3, e4 = edge_idx[(a, b)], edge_idx[(b, d)], edge_idx[(c, d)], edge_idx[(a, c)]
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4): cycles_by_edge[e].append(len(cycles) - 1)
                    # Cycle 3
                    e1, e2, e3, e4 = edge_idx[(a, c)], edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(a, d)]
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4): cycles_by_edge[e].append(len(cycles) - 1)
    
    total_cycles = len(cycles)
    
    colors = list(range(MAX_COLOR + 1))
    random.shuffle(colors)
    
    color_counts = [0] * (MAX_COLOR + 1)
    for c in colors:
        color_counts[c] += 1
        
    is_violating = [False] * total_cycles
    viol_count = 0
    for i in range(total_cycles):
        c = cycles[i]
        if len({colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]}) == 4:
            is_violating[i] = True
            viol_count += 1
            
    current_distinct = len(set(colors))
    
    def apply_change(e, new_c):
        nonlocal viol_count, current_distinct
        old_c = colors[e]
        if old_c == new_c:
            return 0
        
        color_counts[old_c] -= 1
        color_counts[new_c] += 1
        
        delta_distinct = 0
        if color_counts[old_c] == 0:
            delta_distinct -= 1
        if color_counts[new_c] == 1:
            delta_distinct += 1
            
        colors[e] = new_c
        
        delta_viol = 0
        for c_idx in cycles_by_edge[e]:
            was_v = is_violating[c_idx]
            cyc = cycles[c_idx]
            c1, c2, c3, c4 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
            is_v = len({c1, c2, c3, c4}) == 4
            if was_v and not is_v:
                delta_viol -= 1
                is_violating[c_idx] = False
            elif not was_v and is_v:
                delta_viol += 1
                is_violating[c_idx] = True
                
        viol_count += delta_viol
        current_distinct += delta_distinct
        
        return delta_distinct - 1000 * delta_viol

    while time.time() < deadline:
        if viol_count > 0:
            # Find violating cycle
            c_idx = -1
            for i in range(total_cycles):
                if is_violating[i]:
                    c_idx = i
                    break
            if c_idx == -1: continue
            
            cyc = cycles[c_idx]
            e1, e2, e3, e4 = cyc
            moves = [
                (e1, colors[e2]), (e1, colors[e3]), (e1, colors[e4]),
                (e2, colors[e1]), (e2, colors[e3]), (e2, colors[e4]),
                (e3, colors[e1]), (e3, colors[e2]), (e3, colors[e4]),
                (e4, colors[e1]), (e4, colors[e2]), (e4, colors[e3])
            ]
            random.shuffle(moves)
            
            accepted = False
            for e, new_c in moves:
                if colors[e] == new_c: continue
                old_c = colors[e]
                delta = apply_change(e, new_c)
                if delta > 0:
                    accepted = True
                    break
                else:
                    apply_change(e, old_c)
                    
            if not accepted:
                # Random kick
                e = random.choice([e1, e2, e3, e4])
                new_c = random.randint(0, MAX_COLOR)
                old_c = colors[e]
                delta = apply_change(e, new_c)
                if delta <= 0:
                    apply_change(e, old_c)

        else:
            if current_distinct == MAX_COLOR + 1:
                break
                
            unused_colors = [i for i in range(MAX_COLOR + 1) if color_counts[i] == 0]
            if not unused_colors:
                break
                
            e = random.randint(0, NUM_EDGES - 1)
            new_c = random.choice(unused_colors)
            old_c = colors[e]
            delta = apply_change(e, new_c)
            if delta <= 0:
                apply_change(e, old_c)
                
    return colors

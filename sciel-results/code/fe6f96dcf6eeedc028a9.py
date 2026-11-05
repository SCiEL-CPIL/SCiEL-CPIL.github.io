import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        e1 = [get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)]
        e2 = [get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)]
        e3 = [get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)]
        cycles.append(tuple(e1))
        cycles.append(tuple(e2))
        cycles.append(tuple(e3))
    return cycles

def build_edge_to_cycles(cycles):
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return edge_to_cycles

_CYCLES = build_cycles()
_EDGE_TO_CYCLES = build_edge_to_cycles(_CYCLES)

def is_rainbow(colors, cycle):
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    return len({c0, c1, c2, c3}) == 4

def is_rainbow_override1(colors, cycle, edge_idx, override_color):
    c0 = colors[cycle[0]] if cycle[0] != edge_idx else override_color
    c1 = colors[cycle[1]] if cycle[1] != edge_idx else override_color
    c2 = colors[cycle[2]] if cycle[2] != edge_idx else override_color
    c3 = colors[cycle[3]] if cycle[3] != edge_idx else override_color
    return len({c0, c1, c2, c3}) == 4

def is_rainbow_override2(colors, cycle, e1, c1, e2, c2):
    c0 = colors[cycle[0]]
    if cycle[0] == e1: c0 = c1
    elif cycle[0] == e2: c0 = c2
    
    c1_val = colors[cycle[1]]
    if cycle[1] == e1: c1_val = c1
    elif cycle[1] == e2: c1_val = c2
    
    c2_val = colors[cycle[2]]
    if cycle[2] == e1: c2_val = c1
    elif cycle[2] == e2: c2_val = c2
    
    c3_val = colors[cycle[3]]
    if cycle[3] == e1: c3_val = c1
    elif cycle[3] == e2: c3_val = c2
    
    return len({c0, c1_val, c2_val, c3_val}) == 4

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    colors = [0] * NUM_EDGES
    current_violations = 0
    current_distinct = 1
    current_score = current_distinct - 1000 * current_violations
    
    best_colors = colors[:]
    best_distinct = 1
    best_score = current_score
    
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    # Pre-compute set of colors for fast distinct count?
    # len(set(colors)) is O(N). N=120. Fast enough.
    
    while time.time() < deadline and T > min_T:
        if random.random() < 0.5:
            # Recolor
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Bias towards new colors to increase distinct
            # Get unused colors
            used = set(colors)
            unused = [c for c in range(MAX_COLOR) if c not in used]
            
            if unused and random.random() < 0.7:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLOR - 1)
                
            if new_color == old_color:
                continue
            
            colors[e] = new_color
            
            delta_v = 0
            for ci in _EDGE_TO_CYCLES[e]:
                cycle = _CYCLES[ci]
                old_r = is_rainbow_override1(colors, cycle, e, old_color)
                new_r = is_rainbow(colors, cycle)
                
                if old_r and not new_r:
                    delta_v -= 1
                elif not old_r and new_r:
                    delta_v += 1
            
            new_violations = current_violations + delta_v
            new_distinct = len(set(colors))
            new_score = new_distinct - 1000 * new_violations
            delta_score = new_score - current_score
            
            if delta_score > 0 or (T > 0 and random.random() < pow(2.71828, delta_score / T)):
                current_violations = new_violations
                current_distinct = new_distinct
                current_score = new_score
                
                if new_violations == 0 and new_distinct > best_distinct:
                    best_distinct = new_distinct
                    best_colors = colors[:]
                    best_score = new_score
            else:
                colors[e] = old_color
        else:
            # Swap
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            if c1 == c2:
                continue
            
            colors[e1] = c2
            colors[e2] = c1
            
            affected = set(_EDGE_TO_CYCLES[e1])
            affected.update(_EDGE_TO_CYCLES[e2])
            
            delta_v = 0
            for ci in affected:
                cycle = _CYCLES[ci]
                old_r = is_rainbow_override2(colors, cycle, e1, c1, e2, c2)
                new_r = is_rainbow(colors, cycle)
                
                if old_r and not new_r:
                    delta_v -= 1
                elif not old_r and new_r:
                    delta_v += 1
            
            new_violations = current_violations + delta_v
            # Distinct doesn't change
            new_score = current_distinct - 1000 * new_violations
            delta_score = new_score - current_score
            
            if delta_score > 0 or (T > 0 and random.random() < pow(2.71828, delta_score / T)):
                current_violations = new_violations
                current_score = new_score
                
                if new_violations == 0 and current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors[:]
                    best_score = new_score
            else:
                colors[e1] = c1
                colors[e2] = c2
        
        T *= cooling
        
    return best_colors

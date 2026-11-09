import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    # Three 4-cycles on vertices {a,b,c,d}
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        c0 = new_color if cycle[0] == edge else colors[cycle[0]]
        c1 = new_color if cycle[1] == edge else colors[cycle[1]]
        c2 = new_color if cycle[2] == edge else colors[cycle[2]]
        c3 = new_color if cycle[3] == edge else colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def greedy_coloring():
    """Greedy: assign colors one edge at a time, preferring unused colors."""
    colors = [0] * NUM_EDGES
    used = set()
    for e in range(NUM_EDGES):
        # Try unused colors first
        unused = [c for c in range(120) if c not in used]
        random.shuffle(unused)
        found = False
        for nc in unused:
            if can_take_color(colors, e, nc):
                colors[e] = nc
                used.add(nc)
                found = True
                break
        if not found:
            # Fall back to any valid color
            for nc in range(120):
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    found = True
                    break
            if not found:
                colors[e] = 0
    return colors

def local_search_v2(colors, deadline):
    """More aggressive local search with multiple strategies."""
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Escape: random perturbation
            num_kicks = random.randint(5, 15)
            for _ in range(num_kicks):
                e = random.randint(0, NUM_EDGES - 1)
                all_colors = list(range(120))
                random.shuffle(all_colors)
                for nc in all_colors:
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            used_colors = set(colors)
            no_improve = 0
            max_no_improve = random.randint(200, 800)
            continue

        # Try to add a new unused color
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break
            random.shuffle(unused)
            
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
                    # Continue searching for more improvements
                    break
        
        if not improved:
            no_improve += 1
    
    return best_colors, best_score

def simulated_annealing(colors, deadline):
    """Simulated annealing to escape local optima."""
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    temp = 10.0
    cooling = 0.995
    
    while time.time() < deadline and temp > 0.01:
        # Pick a random edge and try a new color
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to find a valid new color, preferring unused
        all_colors = list(range(120))
        random.shuffle(all_colors)
        
        new_color = old_color
        for nc in all_colors:
            if nc != old_color and can_take_color(colors, e, nc):
                new_color = nc
                break
        
        if new_color == old_color:
            temp *= cooling
            continue
        
        # Accept if it improves or with some probability
        old_used = len(used_colors)
        colors[e] = new_color
        new_used = len(set(colors))
        
        if new_used >= old_used:
            used_colors = set(colors)
            if new_used > best_score:
                best_score = new_used
                best_colors = list(colors)
        else:
            # Reject with some probability based on temperature
            if random.random() < np.exp(-1.0 / max(temp, 0.01)):
                used_colors = set(colors)
            else:
                colors[e] = old_color
        
        temp *= cooling
    
    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Multiple random restarts with different strategies
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Choose a strategy
        strategy = restart % 4
        
        if strategy == 0:
            colors = greedy_coloring()
        elif strategy == 1:
            # Random starting point
            colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
            # Fix invalid assignments
            for e in range(NUM_EDGES):
                for nc in range(120):
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
        elif strategy == 2:
            # Structured: use vertex-based coloring
            perm = list(range(N))
            random.shuffle(perm)
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = (perm[i] * 13 + perm[j] * 7) % 120
        else:
            # Hybrid: start with greedy, then perturb
            colors = greedy_coloring()
            for _ in range(10):
                e = random.randint(0, NUM_EDGES - 1)
                for nc in range(120):
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
        
        if not is_valid(colors):
            continue
        
        # Run local search
        ls_deadline = min(time.time() + 0.5, deadline)
        improved_colors, score = local_search_v2(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
        
        # If we have a decent score, try simulated annealing
        if best_score >= 15 and time.time() < deadline - 5:
            sa_deadline = min(time.time() + 2.0, deadline)
            sa_colors, sa_score = simulated_annealing(best_colors, sa_deadline)
            if sa_score > best_score:
                best_score = sa_score
                best_colors = sa_colors
        
        # Early stop if we reach a high score
        if best_score >= 35:
            break
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback: all zeros
        return [0] * NUM_EDGES
        
    return best_colors

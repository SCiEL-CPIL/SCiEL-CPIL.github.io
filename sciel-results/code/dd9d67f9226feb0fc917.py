import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # Precomputation
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
            
    all_cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    all_cycles.append((edge_to_idx[(a,b)], edge_to_idx[(b,c)], edge_to_idx[(c,d)], edge_to_idx[(d,a)]))
                    # Cycle 2: a-c-b-d-a
                    all_cycles.append((edge_to_idx[(a,c)], edge_to_idx[(c,b)], edge_to_idx[(b,d)], edge_to_idx[(d,a)]))
                    # Cycle 3: a-b-d-c-a
                    all_cycles.append((edge_to_idx[(a,b)], edge_to_idx[(b,d)], edge_to_idx[(d,c)], edge_to_idx[(c,a)]))
                    
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in all_cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)

    def solve_one_run():
        colors = [0] * NUM_EDGES
        used_colors = {0}
        next_color = 1
        
        def is_valid_change(edge_idx, new_color):
            for cycle in cycles_by_edge[edge_idx]:
                c_set = {new_color}
                for e in cycle:
                    if e != edge_idx:
                        c_set.add(colors[e])
                if len(c_set) == 4:
                    return False
            return True

        # Greedy add
        while time.time() < deadline - 5: # Leave 5s for finalization
            added = False
            edges_to_try = list(range(NUM_EDGES))
            random.shuffle(edges_to_try)
            for e in edges_to_try:
                if is_valid_change(e, next_color):
                    colors[e] = next_color
                    used_colors.add(next_color)
                    next_color += 1
                    added = True
                    break
            if not added:
                # Try to recolor to break deadlock
                # Pick a random edge and change to a random valid existing color
                e = random.randint(0, NUM_EDGES - 1)
                old_c = colors[e]
                # Try a few
                for _ in range(10):
                    new_c = random.choice(list(used_colors))
                    if new_c != old_c and is_valid_change(e, new_c):
                        colors[e] = new_c
                        break
                        
        return colors, len(used_colors)

    best_colors = [0] * NUM_EDGES
    best_count = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        c, count = solve_one_run()
        if count > best_count:
            best_count = count
            best_colors = c
            
    return best_colors

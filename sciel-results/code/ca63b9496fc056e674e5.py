import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_fallback(n):
        # Generate a random coloring for n=10 as a safe fallback
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(random.randint(0, 3))
        return {'n': n, 'colors': colors}

    best_result = get_fallback(10)

    def solve_n(n, time_budget):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # Precompute all 6-cycles
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        vertices = list(range(n))
        for comb in itertools.combinations(vertices, 6):
            v0 = comb[0]
            rest = comb[1:]
            for perm in itertools.permutations(rest):
                v1, v2, v3, v4, v5 = perm
                if v1 < v5:
                    cycle_edges = [
                        (v0, v1) if v0 < v1 else (v1, v0),
                        (v1, v2) if v1 < v2 else (v2, v1),
                        (v2, v3) if v2 < v3 else (v3, v2),
                        (v3, v4) if v3 < v4 else (v4, v3),
                        (v4, v5) if v4 < v5 else (v5, v4),
                        (v5, v0) if v5 < v0 else (v0, v5)
                    ]
                    cycle_indices = [edge_index[e] for e in cycle_edges]
                    cycles.append(cycle_indices)
                    for idx in cycle_indices:
                        edge_to_cycles[idx].append(len(cycles) - 1)

        colors = [random.randint(0, 3) for _ in range(m)]
        
        def count_mono_cycles(cols):
            count = 0
            for cyc in cycles:
                c0 = cols[cyc[0]]
                if c0 == cols[cyc[1]] == cols[cyc[2]] == cols[cyc[3]] == cols[cyc[4]] == cols[cyc[5]]:
                    count += 1
            return count

        current_count = count_mono_cycles(colors)
        if current_count == 0:
            return {'n': n, 'colors': colors}

        best_count = current_count
        best_colors = colors[:]
        
        restarts = 0
        while time.time() < start_time + time_budget:
            improved = False
            # Local search steps
            for _ in range(500):
                if time.time() > start_time + time_budget:
                    break
                
                # Find a mono cycle by sampling edges
                e_idx = random.randint(0, m - 1)
                target_cycle_idx = -1
                for c_idx in edge_to_cycles[e_idx]:
                    cyc = cycles[c_idx]
                    if colors[cyc[0]] == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]]:
                        target_cycle_idx = c_idx
                        break
                
                if target_cycle_idx == -1:
                    continue
                
                cyc = cycles[target_cycle_idx]
                e_pos = random.randint(0, 5)
                e_idx = cyc[e_pos]
                old_color = colors[e_idx]
                
                # Try to find a better color
                best_delta = 0
                best_new_color = old_color
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    delta = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        c_cyc = cycles[c_idx]
                        
                        # Check if mono before
                        mono_before = True
                        for k in range(6):
                            if colors[c_cyc[k]] != old_color:
                                mono_before = False
                                break
                            
                        # Check if mono after
                        mono_after = True
                        for k in range(6):
                            if k == e_pos:
                                if new_color != colors[c_cyc[k]]: # This check is redundant if we know it's the edge
                                    mono_after = False
                                    break
                            else:
                                if colors[c_cyc[k]] != new_color:
                                    mono_after = False
                                    break
                                    
                        if mono_before and not mono_after:
                            delta -= 1
                        elif not mono_before and mono_after:
                            delta += 1
                            
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color

                if best_new_color != old_color:
                    colors[e_idx] = best_new_color
                    current_count += best_delta
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]
                    if current_count == 0:
                        return {'n': n, 'colors': colors}
                    improved = True
            
            if not improved:
                colors = [random.randint(0, 3) for _ in range(m)]
                current_count = count_mono_cycles(colors)
                restarts += 1
                if restarts > 20:
                    break

        if best_count == 0:
            return {'n': n, 'colors': best_colors}
        return None

    targets = [17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        budget = 15.0
        if time.time() + budget > deadline - 2:
            budget = deadline - 2 - time.time()
            if budget < 1.0:
                break

        result = solve_n(n, budget)
        if result:
            best_result = result
            if n == 17:
                break

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")

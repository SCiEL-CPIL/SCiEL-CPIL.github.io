import time
import sys
import random
import itertools
import math

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles and their edge indices
    # There are C(16,4) = 1820 quadruples. Each has 3 distinct 4-cycles.
    # Total cycles = 1820 * 3 = 5460.
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

    def is_valid_change(colors, edge, new_color):
        # Check all cycles containing this edge
        for c_idx in edge_to_cycles[edge]:
            c = cycles[c_idx]
            # Get colors of the 4 edges in this cycle
            col0 = new_color if c[0] == edge else colors[c[0]]
            col1 = new_color if c[1] == edge else colors[c[1]]
            col2 = new_color if c[2] == edge else colors[c[2]]
            col3 = new_color if c[3] == edge else colors[c[3]]
            
            # Check if 4 distinct
            if col0 != col1 and col0 != col2 and col0 != col3 and \
               col1 != col2 and col1 != col3 and \
               col2 != col3:
                return False
        return True

    def get_initial_colors(strategy, rng):
        colors = [0] * NUM_EDGES
        if strategy == 0:
            # Recursive partition
            color_id = 0
            def process_partition(vertices):
                nonlocal color_id
                if len(vertices) <= 1:
                    return
                mid = len(vertices) // 2
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
        elif strategy == 1:
            # Random
            for i in range(NUM_EDGES):
                colors[i] = rng.randint(0, MAX_COLORS - 1)
        elif strategy == 2:
            # Greedy random: try to assign distinct colors where possible
            # This is hard to do simply, so we'll just use random but with a bias
            # or a simple heuristic: assign colors 0..119 randomly to edges
            perm = list(range(MAX_COLORS))
            rng.shuffle(perm)
            # Assign first 120 edges unique colors, rest random
            for i in range(NUM_EDGES):
                if i < MAX_COLORS:
                    colors[i] = perm[i]
                else:
                    colors[i] = perm[i % MAX_COLORS]
        return colors

    def count_colors(colors):
        return len(set(colors))

    # Main Search Loop: Multi-start Simulated Annealing
    best_score_global = 0
    best_colors_global = [0] * NUM_EDGES
    
    # We will run multiple restarts within the time budget
    rng = random.Random(42)
    
    # Initial best from a few quick strategies
    for strat in range(3):
        init_colors = get_initial_colors(strat, rng)
        # Check validity of init (random might be invalid)
        # If invalid, we need to fix it or discard. 
        # For simplicity, we assume the SA process will fix it, 
        # but starting from an invalid state is tricky for SA which assumes current state is valid.
        # So, let's only start SA from valid states.
        
        # Check validity of initial state
        valid = True
        for c in cycles:
            cols = {init_colors[e] for e in c}
            if len(cols) == 4:
                valid = False
                break
        
        if valid:
            score = count_colors(init_colors)
            if score > best_score_global:
                best_score_global = score
                best_colors_global = init_colors.copy()

    # If no valid initial state found (unlikely for partition, possible for random),
    # force a valid one using the partition strategy which is always valid?
    # Partition strategy: Edges between partitions get unique colors.
    # Does partition strategy ever create a 4-cycle with 4 distinct colors?
    # Let's verify: 
    # In a recursive partition, vertices are grouped. 
    # A 4-cycle involves 4 vertices. 
    # If all 4 are in the same smallest partition, they have colors from deeper levels.
    # Actually, the partition strategy assigns colors based on the level of separation.
    # It is known that this structure often avoids C4 with 4 distinct colors if done carefully,
    # but simple recursive bisection might not guarantee it.
    # However, the seed program used it and got 19, so it's valid.
    
    # Simulated Annealing
    # We will run SA for a fixed number of steps or time, then restart.
    
    current_best_colors = best_colors_global.copy()
    current_best_score = best_score_global
    
    # If we didn't find any valid start, use the partition one and fix if needed?
    # The partition one is likely valid. Let's assume current_best_colors is valid.
    
    while time.time() < deadline:
        # Start a new SA run from current_best_colors
        current_colors = current_best_colors.copy()
        current_score = current_best_score
        
        # Frequency of colors
        freq = [0] * MAX_COLORS
        for c in current_colors:
            freq[c] += 1
            
        T = 1.0
        T_min = 0.01
        alpha = 0.9999
        steps_per_run = 50000 # Approx steps, will be cut by time
        
        no_improve_count = 0
        local_best_score = current_score
        local_best_colors = current_colors.copy()
        
        for step in range(steps_per_run):
            if time.time() > deadline - 5.0:
                break
                
            # Pick random edge
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Pick new color
            # Bias towards colors that are currently unused or rare to increase distinct count?
            # Or just uniform. Uniform is safer for SA dynamics.
            new_color = rng.randint(0, MAX_COLORS - 1)
            
            if new_color == old_color:
                continue
                
            if is_valid_change(current_colors, e, new_color):
                # Calculate delta in distinct colors
                # If freq[old_color] == 1, we lose a color.
                # If freq[new_color] == 0, we gain a color.
                delta = 0
                if freq[old_color] == 1:
                    delta -= 1
                if freq[new_color] == 0:
                    delta += 1
                    
                # Acceptance probability
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5 # Explore neutral moves
                else: # delta < 0
                    # Accept worse move with prob exp(delta / T)
                    # delta is -1 usually.
                    if T > 0:
                        accept = rng.random() < math.exp(delta / T)
                    else:
                        accept = False
                
                if accept:
                    # Update state
                    current_colors[e] = new_color
                    freq[old_color] -= 1
                    freq[new_color] += 1
                    current_score += delta
                    
                    if current_score > local_best_score:
                        local_best_score = current_score
                        local_best_colors = current_colors.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    # Cool down
                    T *= alpha
                    if T < T_min:
                        T = T_min
                        
                    # If we haven't improved in a long time, restart SA from local best
                    if no_improve_count > 20000:
                        break
            else:
                # Invalid move, do nothing. 
                # Optionally cool down slightly even on invalid? No, standard SA only cools on accepted moves.
                pass
        
        # End of SA run
        if local_best_score > best_score_global:
            best_score_global = local_best_score
            best_colors_global = local_best_colors.copy()
            current_best_colors = local_best_colors.copy()
            current_best_score = local_best_score
        else:
            # If no improvement, we might want to perturb the current best to escape
            # Simple perturbation: change a few random edges to random valid colors
            # Or just restart SA from the same best state with a different random seed?
            # The randomness in the loop handles this.
            pass

    return best_colors_global

# Run a quick self-test to ensure it works
if __name__ == "__main__":
    import time
    start = time.time()
    colors = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Distinct colors: {len(set(colors))}")
    
    # Verify validity
    N = 16
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    valid = True
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles_to_check = [
            (edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]),
            (edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]),
            (edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a])
        ]
        for cyc in cycles_to_check:
            cols = {colors[e] for e in cyc}
            if len(cols) == 4:
                valid = False
                print(f"Invalid cycle found: {quad}, {cyc}, colors: {[colors[e] for e in cyc]}")
                break
        if not valid:
            break
            
    print(f"Valid: {valid}")

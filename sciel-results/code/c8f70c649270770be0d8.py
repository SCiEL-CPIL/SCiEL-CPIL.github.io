import time
import random
import math
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def get_seed_solution():
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    def build_cycle_data(n):
        """Precompute all C6s and edge-to-cycle mapping for K_n."""
        if n < 6:
            return [], []
        
        edges = []
        edge_idx = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_idx[(i,j)] = len(edges)
                edges.append((i,j))
        
        num_edges = len(edges)
        cycles = []
        
        # Generate all 6-cycles
        # A 6-cycle is defined by 6 vertices.
        # Number of distinct cycles on 6 labeled vertices is (6-1)! / 2 = 60.
        # We generate them by fixing the smallest vertex to break rotational symmetry,
        # and ensuring the second vertex is smaller than the second-to-last to break reflection symmetry.
        
        for verts in itertools.combinations(range(n), 6):
            v0 = verts[0]
            rest = verts[1:]
            # Permutations of the remaining 5 vertices
            # To avoid duplicates:
            # 1. Fix v0 as start.
            # 2. Ensure perm[0] < perm[4] (second vertex < fifth vertex in the cycle order v0-v1-v2-v3-v4-v5)
            # This handles the reflection symmetry.
            for perm in itertools.permutations(rest, 5):
                if perm[0] > perm[4]:
                    continue
                
                cycle_verts = (v0,) + perm
                edges_in_cycle = []
                valid = True
                for k in range(6):
                    u = cycle_verts[k]
                    v = cycle_verts[(k+1)%6]
                    if u > v:
                        u, v = v, u
                    e_idx = edge_idx.get((u,v))
                    if e_idx is None:
                        valid = False
                        break
                    edges_in_cycle.append(e_idx)
                
                if valid:
                    cycles.append(edges_in_cycle)
        
        # Map edge to list of cycle indices containing it
        edge_to_cycles = [[] for _ in range(num_edges)]
        for idx, cyc in enumerate(cycles):
            for e in cyc:
                edge_to_cycles[e].append(idx)
                
        return cycles, edge_to_cycles

    def count_mono_cycles(colors, cycles):
        """Count the number of monochromatic C6s."""
        count = 0
        for cyc in cycles:
            c0 = colors[cyc[0]]
            mono = True
            for e in cyc[1:]:
                if colors[e] != c0:
                    mono = False
                    break
            if mono:
                count += 1
        return count

    def local_search(n, initial_colors, cycles, edge_to_cycles, time_limit):
        """Perform local search to eliminate mono C6s."""
        num_edges = len(initial_colors)
        colors = initial_colors[:]
        
        # Precompute current mono status for each cycle to speed up delta calculation?
        # Or just calculate delta on the fly.
        # Delta calculation:
        # When changing edge e from old_c to new_c:
        # For each cycle containing e:
        #   - If it was mono old_c, it becomes non-mono (unless new_c == old_c, which is false). Delta -= 1
        #   - If it was not mono old_c, check if it becomes mono new_c.
        #     To be mono new_c, all other 5 edges must be new_c.
        
        def is_mono_cycle(cyc, color):
            return all(colors[e] == color for e in cyc)
        
        current_cost = count_mono_cycles(colors, cycles)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 1.0
        cool_rate = 0.999
        min_T = 0.001
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit:
            if T < min_T:
                break
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            delta = 0
            
            # Check affected cycles
            for cyc_idx in edge_to_cycles[e_idx]:
                cyc = cycles[cyc_idx]
                
                # Check if it was mono old_c
                # Since e_idx is old_c, check if other 5 are old_c
                is_mono_old = True
                is_mono_new = True
                for e in cyc:
                    if e == e_idx:
                        continue
                    if colors[e] != old_c:
                        is_mono_old = False
                    if colors[e] != new_c:
                        is_mono_new = False
                    if not is_mono_old and not is_mono_new:
                        break
                
                if is_mono_old:
                    delta -= 1
                if is_mono_new:
                    delta += 1
            
            # Acceptance criterion
            if delta <= 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_c
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
                    if best_cost == 0:
                        return best_colors
            
            T *= cool_rate
            
        return best_colors if best_cost == 0 else None

    def greedy_construct(n, cycles, edge_to_cycles):
        """Construct a coloring greedily, choosing colors that minimize immediate conflicts."""
        num_edges = n * (n - 1) // 2
        colors = [None] * num_edges
        
        # We need to assign colors to edges.
        # A simple greedy approach: iterate edges in lexicographic order.
        # For each edge, pick the color that results in the fewest mono C6s involving that edge.
        # However, checking all C6s is expensive.
        # Instead, we can just pick the color that minimizes the count of mono C6s *currently formed* 
        # that include this edge.
        
        # To make it faster, we can use a heuristic: 
        # Pick the color that appears least often in the already-colored edges of the cycles containing this edge?
        # Or just random with local repair.
        
        # Let's try a simple random initialization followed by local search, 
        # but with multiple restarts.
        # The greedy part might be: start with a random coloring, then local search.
        
        return None # Placeholder, we will use random restarts in main loop

    def try_solve(n, time_limit):
        if n < 6:
            return [0] * (n * (n - 1) // 2)
        
        cycles, edge_to_cycles = build_cycle_data(n)
        num_edges = n * (n - 1) // 2
        
        # If no cycles, any coloring is valid
        if not cycles:
            return [0] * num_edges
        
        best_result = None
        
        # Strategy: Random restarts with local search
        # We allocate time based on n. Larger n is harder.
        # For n=17, C(17,6)*60 = 12376 * 60 = 742,560 cycles.
        # For n=18, C(18,6)*60 = 18564 * 60 = 1,113,840 cycles.
        # Memory is fine. Time for local search steps depends on degree of edges in cycle graph.
        
        start_ts = time.time()
        attempts = 0
        
        while time.time() - start_ts < time_limit:
            # Random initial coloring
            init_colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Quick check: if already valid (unlikely for random)
            if count_mono_cycles(init_colors, cycles) == 0:
                return init_colors
            
            # Local search
            # Time limit per attempt: decrease as we go? Or fixed?
            # Let's give each attempt a small slice, e.g., 0.5s for n<=16, 1.0s for n=17, 2.0s for n=18
            # But we need to finish before deadline.
            remaining = time_limit - (time.time() - start_ts)
            if remaining < 0.1:
                break
            
            # Heuristic for time per attempt
            if n <= 14:
                attempt_time = 0.1
            elif n <= 16:
                attempt_time = 0.2
            elif n == 17:
                attempt_time = 0.5
            else:
                attempt_time = 1.0
            
            attempt_time = min(attempt_time, remaining)
            
            result = local_search(n, init_colors, cycles, edge_to_cycles, attempt_time)
            if result is not None:
                return result
            
            attempts += 1
            # Randomize seed for next attempt
            # random.seed(attempts) # Not necessary, random is already seeded
            
        return None

    # Main logic
    # Try to find the largest n possible.
    # Known best is 17. 18 is the target for improvement.
    # We should try 18 first? If we fail, fall back to 17.
    # But 18 is much harder. 
    # Strategy:
    # 1. Try n=18 for a short time (e.g., 30s). If found, return it.
    # 2. Try n=17 for the remaining time.
    # 3. If 17 fails, try 16, etc.
    
    # However, the task says "Maximize n". 
    # If we can't find 18, we should definitely try to find 17.
    # The seed returns 9. We need to jump to 17.
    
    # Let's reserve time for 17 to ensure we at least get 17.
    # Time budget: 140s.
    # Allocate 40s to n=18.
    # Allocate 100s to n=17.
    
    # First, let's see if we can quickly construct a valid 17.
    # If we don't find 18, we must return 17.
    
    # Actually, it's safer to try 17 first to guarantee a good score, 
    # then use leftover time for 18?
    # No, if we find 17 quickly, we can use the rest for 18.
    # If we spend all time on 18 and fail, we might not have time for 17.
    # So:
    # 1. Try n=17 for 10s. If found, store it.
    # 2. Try n=18 for remaining time - 10s safety.
    # 3. If 18 found, return it. Else return 17.
    
    # Let's refine:
    # Try n=17 with a short budget to get a baseline.
    # Then try n=18 with the rest.
    
    # But 17 is also non-trivial. Random restarts might take a while.
    # Let's try n=17 for 20s.
    # If found, great.
    # Then try n=18 for 110s.
    
    # Actually, let's try n=18 first? No, 17 is more likely.
    # Let's do:
    # 1. Try n=17 for 15s.
    # 2. If found, try n=18 for 115s.
    # 3. If 18 found, return 18.
    # 4. If 18 not found, return 17.
    # 5. If 17 not found, try n=16 for remaining time.
    
    # Let's implement this hierarchy.
    
    best_n = 9
    best_colors = get_seed_solution()['colors']
    
    # Try 17
    t_start = time.time()
    t_limit_17 = 20.0
    res_17 = try_solve(17, t_limit_17)
    if res_17 is not None:
        best_n = 17
        best_colors = res_17
    
    # Try 18
    elapsed = time.time() - t_start
    remaining = 140.0 - elapsed - 5.0 # 5s safety margin
    if remaining > 10.0:
        res_18 = try_solve(18, remaining)
        if res_18 is not None:
            best_n = 18
            best_colors = res_18
    else:
        # If we didn't get 17, try 16 with remaining time
        if best_n < 17 and remaining > 5.0:
            res_16 = try_solve(16, remaining)
            if res_16 is not None:
                best_n = 16
                best_colors = res_16
        elif best_n < 17:
            # Try 15
            res_15 = try_solve(15, remaining)
            if res_15 is not None:
                best_n = 15
                best_colors = res_15
        elif best_n < 17:
            # Try 14
            res_14 = try_solve(14, remaining)
            if res_14 is not None:
                best_n = 14
                best_colors = res_14
        elif best_n < 17:
            # Try 13
            res_13 = try_solve(13, remaining)
            if res_13 is not None:
                best_n = 13
                best_colors = res_13
        elif best_n < 17:
            # Try 12
            res_12 = try_solve(12, remaining)
            if res_12 is not None:
                best_n = 12
                best_colors = res_12
        elif best_n < 17:
            # Try 11
            res_11 = try_solve(11, remaining)
            if res_11 is not None:
                best_n = 11
                best_colors = res_11
        elif best_n < 17:
            # Try 10
            res_10 = try_solve(10, remaining)
            if res_10 is not None:
                best_n = 10
                best_colors = res_10

    # Final check: ensure we don't exceed deadline
    if time.time() > deadline - 2.0:
        # Emergency return
        pass

    return {'n': best_n, 'colors': best_colors}

# Keep the function name and signature as required
# The seed program had `construct()`

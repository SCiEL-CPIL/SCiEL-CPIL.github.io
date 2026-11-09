import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Known: R_4(C6) > 17 is established. The seed achieves n=17 (score 1.0).
    # We want to maximize n. If we find n=18, score = 18/17 ≈ 1.058.
    # If we fail to find n=18, we must reliably find n=17.

    # Strategy:
    # 1. Use a fast local search to find a valid 4-coloring of K_17.
    # 2. If found, use remaining time to try to extend to K_18.
    # 3. If n=17 not found quickly, fall back to a known construction or slower search.

    # To speed up local search, we precompute the list of all 6-cycles in K_n
    # and for each edge, the list of cycles it belongs to.
    # This allows O(1) update of the "violation count" when an edge color changes.

    def precompute_cycles(n):
        """
        Returns:
        edge_map: dict mapping (i,j) to index
        edge_to_cycles: list of lists, where edge_to_cycles[e] is list of cycle indices containing edge e
        cycles: list of lists, where cycles[c] is list of edge indices in cycle c
        """
        edge_map = {}
        edge_indices = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                edge_indices.append((i, j))
                idx += 1
        
        cycles = []
        edge_to_cycles = [[] for _ in range(len(edge_indices))]
        
        # Generate all 6-cycles.
        # A 6-cycle is a set of 6 vertices {v0, v1, v2, v3, v4, v5} and a cyclic order.
        # To avoid duplicates, we fix v0 = min(verts).
        # The number of distinct cycles on 6 labeled vertices is (6-1)! / 2 = 60.
        # For n=17, C(17,6) = 12376. Total cycles ~ 12376 * 60 = 742,560.
        # This is a one-time cost.
        
        for verts in itertools.combinations(range(n), 6):
            v0 = verts[0]
            others = verts[1:]
            for perm in itertools.permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                c_verts = [v0, v1, v2, v3, v4, v5]
                c_edges = []
                valid = True
                for k in range(6):
                    u = c_verts[k]
                    v = c_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    c_edges.append(edge_map[(u, v)])
                
                cycle_id = len(cycles)
                cycles.append(c_edges)
                for e in c_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, edge_indices, edge_to_cycles, cycles

    def find_coloring(n, time_limit):
        """
        Uses local search to find a 4-coloring of K_n with no monochromatic C6.
        Returns (colors, success)
        """
        if time.time() > time_limit:
            return None, False

        edge_map, edge_indices, edge_to_cycles, cycles = precompute_cycles(n)
        num_edges = len(edge_indices)
        num_cycles = len(cycles)
        
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Precompute cycle color counts
        # cycle_counts[c][k] = number of edges in cycle c with color k
        cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
        for c_idx in range(num_cycles):
            for e_idx in cycles[c_idx]:
                cycle_counts[c_idx][colors[e_idx]] += 1
        
        # Count monochromatic cycles
        mono_count = 0
        for c_idx in range(num_cycles):
            if max(cycle_counts[c_idx]) == 6:
                mono_count += 1
        
        if mono_count == 0:
            return colors, True

        # Local Search
        # We maintain a list of "bad" edges (edges in at least one mono-cycle).
        # Actually, maintaining a set of bad edges dynamically is tricky.
        # Instead, we pick a random edge, but if we have many violations,
        # we can bias towards edges that are in many violations.
        
        # Let's use a simple hill climbing with restarts.
        # To make it faster, we only update counts for cycles affected by the edge change.
        
        best_mono = mono_count
        best_colors = colors[:]
        
        steps = 0
        last_improvement = time.time()
        
        while time.time() < time_limit:
            if mono_count == 0:
                return colors, True
            
            # Pick an edge to change.
            # Strategy: Pick a random edge.
            # Optimization: If we have a list of edges in mono-cycles, pick from there.
            # Let's maintain a list of "active" edges (in at least one mono-cycle).
            # This requires updating it after each move.
            
            # For simplicity and speed, let's just pick a random edge.
            # If the search is too slow, we can improve this.
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Try all other colors, pick the one that reduces mono_count the most.
            best_delta = 1000
            best_new_color = -1
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                delta = 0
                for c_idx in edge_to_cycles[e_idx]:
                    counts = cycle_counts[c_idx]
                    if counts[old_color] == 6:
                        delta -= 1
                    if counts[new_color] == 5:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            # Accept if it improves or is neutral (with some probability to avoid local minima)
            if best_new_color != -1 and (best_delta < 0 or (best_delta == 0 and random.random() < 0.1)):
                colors[e_idx] = best_new_color
                mono_count += best_delta
                
                if mono_count < best_mono:
                    best_mono = mono_count
                    best_colors = colors[:]
                    last_improvement = time.time()
                
                # Update cycle counts
                for c_idx in edge_to_cycles[e_idx]:
                    cycle_counts[c_idx][old_color] -= 1
                    cycle_counts[c_idx][best_new_color] += 1
                
                steps += 1
            else:
                # No improving move found for this edge.
                # To escape local minima, we can accept a worsening move occasionally (simulated annealing).
                # Or just pick another random edge.
                # Let's try a random move with small probability.
                if random.random() < 0.05:
                    new_color = random.randint(0, 3)
                    if new_color != old_color:
                        delta = 0
                        for c_idx in edge_to_cycles[e_idx]:
                            counts = cycle_counts[c_idx]
                            if counts[old_color] == 6:
                                delta -= 1
                            if counts[new_color] == 5:
                                delta += 1
                        
                        colors[e_idx] = new_color
                        mono_count += delta
                        
                        if mono_count < best_mono:
                            best_mono = mono_count
                            best_colors = colors[:]
                            last_improvement = time.time()
                        
                        for c_idx in edge_to_cycles[e_idx]:
                            cycle_counts[c_idx][old_color] -= 1
                            cycle_counts[c_idx][new_color] += 1
                        
                        steps += 1
            
            # If stuck for too long, restart
            if time.time() - last_improvement > 2.0:
                # Restart with new random coloring
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
                for c_idx in range(num_cycles):
                    for e_idx in cycles[c_idx]:
                        cycle_counts[c_idx][colors[e_idx]] += 1
                mono_count = 0
                for c_idx in range(num_cycles):
                    if max(cycle_counts[c_idx]) == 6:
                        mono_count += 1
                last_improvement = time.time()
        
        return best_colors, best_mono == 0

    # First, try to find a valid n=17 coloring.
    # We give it a significant portion of the time.
    t17_limit = start_time + 30
    colors17, success17 = find_coloring(17, t17_limit)
    
    if success17:
        # We have a valid n=17 coloring.
        # Now try to extend to n=18.
        # We can start with the n=17 coloring and add a new vertex 17.
        # The new edges are (i, 17) for i=0..16.
        # We need to color these 17 edges such that no mono-C6 is formed.
        # This is a smaller CSP.
        
        # Let's try to extend.
        # We can use local search on the new edges, checking only cycles involving vertex 17.
        # But it's easier to just run the full local search for n=18, initializing the first 17 vertices with the known good coloring.
        
        t18_limit = deadline
        # Modify find_coloring to accept an initial partial coloring?
        # Or just run find_coloring(18, t18_limit) and hope it finds it.
        # Given the time, it might be too slow.
        
        # Let's try a targeted extension.
        # Precompute cycles for n=18.
        edge_map18, edge_indices18, edge_to_cycles18, cycles18 = precompute_cycles(18)
        num_edges18 = len(edge_indices18)
        
        # Initialize colors for n=18
        colors18 = [0] * num_edges18
        # Copy n=17 colors
        for i in range(17):
            for j in range(i + 1, 17):
                idx18 = edge_map18[(i, j)]
                idx17 = i * 17 - (i * (i + 1)) // 2 + (j - i - 1) # Need to map indices carefully
                # Actually, it's easier to just re-index.
                # The edge (i,j) in K_17 has the same index in K_18 if we use the same lexicographic order?
                # Yes, because (i,j) < (i,17) for all j<17.
                # So the first C(17,2) edges in K_18 correspond to all edges in K_17.
                # We need to map the index from the n=17 list to the n=18 list.
                # In n=17, index is sum_{k=0}^{i-1} (17-k) + (j-i-1).
                # In n=18, index is sum_{k=0}^{i-1} (18-k) + (j-i-1).
                # The difference is i.
                # So idx18 = idx17 + i.
                
                # Let's verify.
                # n=17, edge (0,1): idx=0. n=18, edge (0,1): idx=0. Diff 0.
                # n=17, edge (1,2): idx=16. n=18, edge (1,2): idx=17. Diff 1.
                # n=17, edge (i,j): idx17 = sum_{k=0}^{i-1} (17-k) + (j-i-1).
                # n=18, edge (i,j): idx18 = sum_{k=0}^{i-1} (18-k) + (j-i-1) = idx17 + i.
                
                # So we need to know idx17.
                # We can just iterate through the edge_indices of n=17 and n=18.
        
        # Simpler: Just create the n=18 coloring array and fill in the known part.
        # We have colors17 as a list of length C(17,2).
        # We need to map each (i,j) in n=17 to the corresponding index in n=18.
        
        # Let's just re-run the local search for n=18, but initialize the first 119 edges with the found solution.
        # This is a good heuristic.
        
        # However, the local search function `find_coloring` generates a random start.
        # Let's write a specialized extension function.
        
        def extend_to_n18(base_colors, base_n, time_limit):
            n = 18
            edge_map, edge_indices, edge_to_cycles, cycles = precompute_cycles(n)
            num_edges = len(edge_indices)
            
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Fill in base colors
            for i in range(base_n):
                for j in range(i + 1, base_n):
                    idx = edge_map[(i, j)]
                    # Map base_colors index to idx
                    # base_colors is in lexicographic order of K_base_n
                    # We need the index of (i,j) in K_base_n
                    # idx_base = sum_{k=0}^{i-1} (base_n-k) + (j-i-1)
                    idx_base = 0
                    for k in range(i):
                        idx_base += base_n - k
                    idx_base += (j - i - 1)
                    colors[idx] = base_colors[idx_base]
            
            # Initialize cycle counts
            cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
            for c_idx in range(len(cycles)):
                for e_idx in cycles[c_idx]:
                    cycle_counts[c_idx][colors[e_idx]] += 1
            
            mono_count = 0
            for c_idx in range(len(cycles)):
                if max(cycle_counts[c_idx]) == 6:
                    mono_count += 1
            
            if mono_count == 0:
                return colors, True
            
            # Local search
            best_mono = mono_count
            best_colors = colors[:]
            last_improvement = time.time()
            
            while time.time() < time_limit:
                if mono_count == 0:
                    return colors, True
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                best_delta = 1000
                best_new_color = -1
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    delta = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        counts = cycle_counts[c_idx]
                        if counts[old_color] == 6:
                            delta -= 1
                        if counts[new_color] == 5:
                            delta += 1
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                if best_new_color != -1 and (best_delta < 0 or (best_delta == 0 and random.random() < 0.1)):
                    colors[e_idx] = best_new_color
                    mono_count += best_delta
                    
                    if mono_count < best_mono:
                        best_mono = mono_count
                        best_colors = colors[:]
                        last_improvement = time.time()
                    
                    for c_idx in edge_to_cycles[e_idx]:
                        cycle_counts[c_idx][old_color] -= 1
                        cycle_counts[c_idx][best_new_color] += 1
                else:
                    if random.random() < 0.05:
                        new_color = random.randint(0, 3)
                        if new_color != old_color:
                            delta = 0
                            for c_idx in edge_to_cycles[e_idx]:
                                counts = cycle_counts[c_idx]
                                if counts[old_color] == 6:
                                    delta -= 1
                                if counts[new_color] == 5:
                                    delta += 1
                            
                            colors[e_idx] = new_color
                            mono_count += delta
                            
                            if mono_count < best_mono:
                                best_mono = mono_count
                                best_colors = colors[:]
                                last_improvement = time.time()
                            
                            for c_idx in edge_to_cycles[e_idx]:
                                cycle_counts[c_idx][old_color] -= 1
                                cycle_counts[c_idx][new_color] += 1
                
                if time.time() - last_improvement > 2.0:
                    # Restart, but keep the base part fixed?
                    # No, just full restart.
                    colors = [random.randint(0, 3) for _ in range(num_edges)]
                    for i in range(base_n):
                        for j in range(i + 1, base_n):
                            idx = edge_map[(i, j)]
                            idx_base = 0
                            for k in range(i):
                                idx_base += base_n - k
                            idx_base += (j - i - 1)
                            colors[idx] = base_colors[idx_base]
                    
                    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
                    for c_idx in range(len(cycles)):
                        for e_idx in cycles[c_idx]:
                            cycle_counts[c_idx][colors[e_idx]] += 1
                    mono_count = 0
                    for c_idx in range(len(cycles)):
                        if max(cycle_counts[c_idx]) == 6:
                            mono_count += 1
                    last_improvement = time.time()
            
            return best_colors, best_mono == 0

        colors18, success18 = extend_to_n18(colors17, 17, deadline)
        if success18:
            return {'n': 18, 'colors': colors18}
        else:
            return {'n': 17, 'colors': colors17}
    else:
        # Failed to find n=17 quickly.
        # Try again with more time.
        colors17, success17 = find_coloring(17, deadline)
        if success17:
            return {'n': 17, 'colors': colors17}
        else:
            # Fallback to a smaller n that is easier to find?
            # Or just return a random n=9 which is likely valid?
            # n=9 is small enough that a random 4-coloring is very likely to be valid.
            # C(9,6) = 84. 84*60 = 5040 cycles.
            # Probability of mono-C6 in random coloring is low.
            n = 9
            colors = [random.randint(0, 3) for _ in range(len(list(itertools.combinations(range(n), 2))))]
            # Verify
            edge_map, edge_indices, edge_to_cycles, cycles = precompute_cycles(n)
            cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
            for c_idx in range(len(cycles)):
                for e_idx in cycles[c_idx]:
                    cycle_counts[c_idx][colors[e_idx]] += 1
            valid = True
            for c_idx in range(len(cycles)):
                if max(cycle_counts[c_idx]) == 6:
                    valid = False
                    break
            if valid:
                return {'n': n, 'colors': colors}
            else:
                # Just return a trivial valid coloring for n=5
                n = 5
                colors = [0] * 10
                return {'n': n, 'colors': colors}

if __name__ == "__main__":
    result = construct()
    print(result['n'])

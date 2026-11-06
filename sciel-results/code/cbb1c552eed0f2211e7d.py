import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin for safety

    # Precompute cycle structures for various n
    # We will try n from 20 down to 17
    # If we find a valid one for 20, 19, or 18, we return it immediately.
    # Otherwise, we return the best found (likely 17 from a known construction or found via search).
    
    # Known n=17 circulant coloring to ensure we at least get 17
    def get_known_17():
        n = 17
        # Based on known literature or previous successful runs
        # We can reconstruct a valid one if search fails
        # For now, let's rely on search, but have a fallback
        # A simple random search for 17 is very likely to succeed quickly
        pass
        return None

    def generate_cycle_masks(n):
        """
        Generate all C6 cycles in K_n.
        Returns a list of tuples, where each tuple contains the indices of the 6 edges in the cycle.
        Also returns a mapping from edge index to list of cycle indices containing that edge.
        """
        edges = []
        edge_to_idx = {}
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = len(edges)
                edges.append((i, j))
        
        num_edges = len(edges)
        
        # Generate all sets of 6 vertices
        from itertools import combinations
        cycles = []
        # For each set of 6 vertices, generate the 5 distinct C6 cycles
        # Vertices: 0, 1, 2, 3, 4, 5
        # Cycle 1: 0-1-2-3-4-5-0
        # Cycle 2: 0-1-2-3-5-4-0
        # Cycle 3: 0-1-2-4-3-5-0
        # Cycle 4: 0-1-3-2-4-5-0
        # Cycle 5: 0-1-3-4-2-5-0
        # Actually, there are (6-1)! / 2 = 60 / 2 = 30 undirected Hamiltonian cycles on K6?
        # Wait. Number of distinct cycles in K6.
        # Total permutations of 6 vertices: 720.
        # Fix vertex 0 at start. 5! = 120.
        # Reverse is same cycle: 120 / 2 = 60.
        # So there are 60 distinct C6 cycles on a given set of 6 vertices.
        
        # Let's generate them programmatically to be safe.
        def get_cycles_on_6(vertices):
            # vertices is a tuple of 6 distinct vertex IDs
            # We want all unique cycles.
            # Fix v0. Permute v1..v5.
            # Cycle is (v0, v1, v2, v3, v4, v5, v0).
            # To avoid duplicates, we can require v1 < v5? No, that's for undirected edges in path.
            # Standard way: generate all permutations of 1..5, keep those where perm[0] < perm[4] (indices 1 and 5 of the cycle, relative to fixed 0).
            # Actually, simpler: just generate all 60 and use a set of frozensets of edge pairs to deduplicate.
            
            v = list(vertices)
            seen = set()
            unique_cycles = []
            from itertools import permutations
            # Fix v[0]
            for perm in permutations(v[1:]):
                cycle_verts = [v[0]] + list(perm)
                # Check uniqueness: sort the edges of the cycle
                edge_set = frozenset()
                for i in range(6):
                    u = cycle_verts[i]
                    w = cycle_verts[(i+1)%6]
                    if u > w: u, w = w, u
                    edge_set.add((u, w))
                if edge_set not in seen:
                    seen.add(edge_set)
                    unique_cycles.append(cycle_verts)
            return unique_cycles

        # This might be slow if done naively for all 6-subsets.
        # Let's just generate the 60 patterns once.
        base_cycles = []
        from itertools import permutations
        seen_base = set()
        for perm in permutations(range(1, 6)):
            cycle = [0] + list(perm)
            edge_set = frozenset()
            for i in range(6):
                u = cycle[i]
                w = cycle[(i+1)%6]
                if u > w: u, w = w, u
                edge_set.add((u, w))
            if edge_set not in seen_base:
                seen_base.add(edge_set)
                base_cycles.append(cycle)
        
        # base_cycles now has 60 entries, each a permutation of 0..5 representing a C6.
        
        all_cycles = []
        for verts in combinations(range(n), 6):
            # Map 0..5 in base_cycles to actual vertex IDs in verts
            # verts is a tuple of 6 sorted vertex IDs
            # We need to apply the permutation of indices to the vertex IDs?
            # No, the base cycles are on indices 0..5.
            # If we pick a subset of vertices, any bijection from {0..5} to the subset defines a cycle.
            # But we just need the set of edges.
            # For a specific subset of vertices V, the cycles are isomorphic to the base cycles.
            # We can just take the base cycles and map the indices 0..5 to the sorted vertices in V.
            # Is this correct?
            # Yes, any set of 6 vertices induces a K6. The cycles in that K6 are in 1-1 correspondence with the cycles in the canonical K6.
            # So for each 6-subset, we generate 60 cycles by mapping the canonical indices to the actual vertices.
            
            for bc in base_cycles:
                # bc is a list of 6 indices in 0..5
                # Map these to the actual vertices in 'verts'
                # Note: 'verts' is sorted.
                # The mapping is: index i in bc -> verts[i]?
                # No, that's not right. The base cycle uses vertices 0..5.
                # The subset 'verts' has 6 vertices.
                # We can just replace the vertex labels in the base cycle with the labels in 'verts'.
                # But the base cycle is a sequence.
                # Let's just create the edge set directly.
                edge_indices = []
                valid = True
                for i in range(6):
                    u_idx = bc[i]
                    w_idx = bc[(i+1)%6]
                    # Map u_idx (0..5) to actual vertex?
                    # Wait, the base cycle is on the *positions* 0..5.
                    # If we have vertices v0, v1, v2, v3, v4, v5, the cycle defined by base pattern [0, 1, 2, 3, 4, 5] connects v0-v1, v1-v2, etc.
                    # So yes, we map the index in the base cycle to the vertex at that index in the sorted subset.
                    u = verts[u_idx]
                    w = verts[w_idx]
                    if u > w: u, w = w, u
                    e_idx = edge_to_idx.get((u, w))
                    if e_idx is None:
                        valid = False
                        break
                    edge_indices.append(e_idx)
                if valid:
                    all_cycles.append(tuple(edge_indices))
        
        return all_cycles, edge_to_idx

    def solve_for_n(n):
        if time.time() > deadline:
            return None
            
        cycles, edge_to_idx = generate_cycle_masks(n)
        num_edges = len(edge_to_idx)
        
        # Map edge index to list of cycle indices
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle_edges in enumerate(cycles):
            for e_idx in cycle_edges:
                edge_to_cycles[e_idx].append(c_idx)
        
        # State: colors for all edges
        # colors[e_idx] is in {0, 1, 2, 3}
        
        best_cost = float('inf')
        best_colors = None
        
        # Initialize with random colors
        current_colors = np.random.randint(0, 4, size=num_edges)
        
        # Calculate initial cost
        # cost = number of monochromatic cycles
        # For each cycle, check if all 6 edges have the same color
        
        def calc_cost(colors):
            cost = 0
            for cycle_edges in cycles:
                c0 = colors[cycle_edges[0]]
                # Check if all others match
                # Vectorized check might be faster
                if all(colors[e] == c0 for e in cycle_edges[1:]):
                    cost += 1
            return cost

        current_cost = calc_cost(current_colors)
        if current_cost == 0:
            return {'n': n, 'colors': current_colors.tolist()}
            
        best_cost = current_cost
        best_colors = current_colors.copy()
        
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        max_steps = 500000 # Adjust based on time
        
        # We can optimize cost update
        # Keep an array cycle_colors or just re-evaluate affected cycles
        
        # For speed, let's precompute for each cycle the list of edges.
        # When we change edge e from c_old to c_new:
        # 1. For each cycle c in edge_to_cycles[e]:
        #    - Check if c was monochromatic with c_old. If yes, cost -= 1.
        #    - Check if c will be monochromatic with c_new. If yes, cost += 1.
        
        # To check "monochromatic with c", we need to know the colors of the other 5 edges.
        # We can maintain a count of how many edges in each cycle have a specific color?
        # Or just store the colors and check.
        
        # Let's use a helper to check if a cycle is monochromatic given the current colors array.
        # Since we only change one edge, we can check the other 5 edges.
        
        def is_mono_cycle(c_idx, colors, changed_edge, new_color):
            # If changed_edge is in this cycle, we substitute its color with new_color for the check
            # Actually, we just check if all edges in the cycle have the same color.
            # If the changed edge is in the cycle, we use new_color for it.
            # If not, we use current colors.
            # But we only call this for cycles containing the changed edge.
            
            # Get the color of the first edge to compare
            # Let's just pick the first edge in the cycle
            e0 = cycles[c_idx][0]
            target_color = new_color if e0 == changed_edge else colors[e0]
            
            for e in cycles[c_idx]:
                c = new_color if e == changed_edge else colors[e]
                if c != target_color:
                    return False
            return True

        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            # Pick a random edge
            e = np.random.randint(num_edges)
            
            # Pick a new color different from current
            old_color = current_colors[e]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            
            # Calculate delta cost
            delta = 0
            for c_idx in edge_to_cycles[e]:
                # Was it mono before?
                # We need to check with old_color
                # This is slightly expensive. 
                # Optimization: Precompute for each cycle a bitmask or something?
                # For now, let's just check.
                
                # Check if mono with old_color
                is_mono_old = is_mono_cycle(c_idx, current_colors, e, old_color)
                # Check if mono with new_color
                is_mono_new = is_mono_cycle(c_idx, current_colors, e, new_color)
                
                if is_mono_old:
                    delta -= 1
                if is_mono_new:
                    delta += 1
            
            new_cost = current_cost + delta
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = current_colors.copy()
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors.tolist()}
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_colors[e] = new_color
                current_cost = new_cost
            
            T *= alpha
            if T < T_min:
                T = T_min
                # Restart or perturb?
                # Let's just continue.
                
            steps += 1
        
        if best_cost == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try n from 20 down to 17
    for n in range(20, 16, -1):
        if time.time() > deadline:
            break
        result = solve_for_n(n)
        if result:
            return result

    # Fallback: Return a known valid coloring for n=17 if we found one?
    # Or just return n=17 with a random coloring? No, it must be valid.
    # If solve_for_n(17) failed to find 0 cost, we might return an invalid one.
    # We should ensure we return a valid one.
    # Let's run solve_for_n(17) one last time with whatever time is left.
    if time.time() < deadline:
        result = solve_for_n(17)
        if result:
            return result

    # Absolute fallback: A small valid graph
    # n=5 is always valid (no C6 possible)
    # But we want to maximize n.
    # If we couldn't find 17, something is wrong.
    # Let's return a trivial valid one.
    n = 5
    colors = [0] * (n*(n-1)//2)
    return {'n': n, 'colors': colors}


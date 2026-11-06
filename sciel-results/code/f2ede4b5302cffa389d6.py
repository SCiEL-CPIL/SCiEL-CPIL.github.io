import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute 6-cycles for a given n
    def get_c6s(n):
        c6s = []
        # A 6-cycle is a sequence v0-v1-v2-v3-v4-v5-v0
        # We can represent it by the set of edges or the tuple of vertices
        # To avoid duplicates, we can fix v0 < v1 and consider rotations
        # Simpler: iterate all combinations of 6 vertices, then all permutations? No, too slow.
        # Iterate all 6-tuples (v0, ..., v5) such that v0 < v1 and v0 is the smallest?
        # Actually, for n=17, number of 6-cycles is manageable.
        # Number of 6-cycles in K_n is (n choose 6) * (6-1)! / 2 = C(n,6) * 60
        # For n=17, C(17,6) = 12376, so 12376 * 60 = 742560 cycles.
        # This is a lot to precompute and store in memory if not careful, but manageable.
        
        # Better: generate cycles on the fly or store edge indices.
        # Let's store cycles as tuples of edge indices.
        # Map (u, v) with u < v to index.
        edge_index = {}
        idx = 0
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edge_index[(i, j)] = idx
                edges.append((i, j))
                idx += 1
        
        cycles = []
        # Generate all 6-cycles
        # v0 < v1, v0 is min(v0..v5)
        # To generate all distinct cycles, we can iterate over all combinations of 6 vertices
        # and then all Hamiltonian cycles in the induced K6.
        # Number of Hamiltonian cycles in K6 is (6-1)! / 2 = 60.
        from itertools import combinations, permutations
        
        for verts in combinations(range(n), 6):
            # Generate all permutations of the 6 vertices
            # To avoid duplicates, fix the first vertex?
            # A cycle is defined by its edges.
            # Let's just generate all 60 cycles for these 6 vertices.
            # Fix v0 = min(verts).
            # The remaining 5 vertices can be permuted in (5-1)! = 24 ways?
            # No, the number of distinct cycles on 6 labeled vertices is (6-1)!/2 = 60.
            # We can generate them by fixing one vertex, say min(verts), and permuting the others.
            # But we must ensure we don't double count reversals.
            
            v_min = min(verts)
            others = [v for v in verts if v != v_min]
            # Permute others
            for perm in permutations(others):
                # Cycle: v_min, perm[0], perm[1], perm[2], perm[3], perm[4], v_min
                # Check if this cycle is a "new" one or duplicate of a reversal?
                # Reversal would be v_min, perm[4], perm[3], perm[2], perm[1], perm[0], v_min
                # We can enforce an order, e.g., perm[0] < perm[4]
                if perm[0] < perm[4]:
                    cycle_edges = []
                    path = [v_min] + list(perm)
                    for k in range(6):
                        u = path[k]
                        v = path[(k+1)%6]
                        if u > v: u, v = v, u
                        cycle_edges.append(edge_index[(u, v)])
                    cycles.append(tuple(cycle_edges))
        
        return cycles, edge_index, edges

    def check_and_search(n):
        cycles, edge_index, edges = get_c6s(n)
        num_edges = len(edges)
        num_cycles = len(cycles)
        
        # Precompute for each edge, which cycles it is in
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_cycles[e_idx].append(c_idx)
        
        best_result = None
        max_restarts = 20
        restarts = 0
        
        while restarts < max_restarts and time.time() < deadline:
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Count monochromatic cycles
            mono_count = 0
            cycle_mono = [False] * num_cycles
            for c_idx, cycle in enumerate(cycles):
                c0 = colors[cycle[0]]
                mono = True
                for e_idx in cycle[1:]:
                    if colors[e_idx] != c0:
                        mono = False
                        break
                if mono:
                    mono_count += 1
                    cycle_mono[c_idx] = True
            
            if mono_count == 0:
                # Found a valid coloring
                # Map back to lexicographic order
                # The 'edges' list is already in lexicographic order (i, j) with i < j
                return {'n': n, 'colors': colors}
            
            # Local Search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to change an edge involved in a monochromatic cycle
                # Pick a random monochromatic cycle
                mono_cycles_indices = [i for i, m in enumerate(cycle_mono) if m]
                if not mono_cycles_indices:
                    break
                c_idx = random.choice(mono_cycles_indices)
                cycle = cycles[c_idx]
                # Pick a random edge in this cycle
                e_idx = random.choice(cycle)
                old_color = colors[e_idx]
                
                # Try all other colors
                best_new_color = -1
                best_delta = 0
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    # Calculate delta
                    delta = 0
                    for nc_idx in edge_to_cycles[e_idx]:
                        nc = cycles[nc_idx]
                        # Check if this cycle was mono
                        was_mono = cycle_mono[nc_idx]
                        
                        # Check if it becomes mono or stops being mono
                        # We need to check the colors of the other 5 edges
                        # Since we are only changing e_idx, we can check the rest
                        # If was_mono, it becomes non-mono if new_color != old_color (which is true)
                        # So it loses 1 mono status.
                        # If not was_mono, it becomes mono if all other 5 edges have new_color
                        # Let's just re-evaluate the cycle status for efficiency in small n
                        # For n=17, this is fast enough.
                        
                        # Actually, simpler:
                        # If was_mono: delta -= 1 (it stops being mono)
                        # If not was_mono: check if all other edges are new_color
                        if was_mono:
                            delta -= 1
                        else:
                            all_match = True
                            for oe_idx in nc:
                                if oe_idx == e_idx:
                                    continue
                                if colors[oe_idx] != new_color:
                                    all_match = False
                                    break
                            if all_match:
                                delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                if best_new_color != -1 and best_delta < 0:
                    # Apply change
                    colors[e_idx] = best_new_color
                    # Update cycle_mono
                    for nc_idx in edge_to_cycles[e_idx]:
                        nc = cycles[nc_idx]
                        # Re-evaluate
                        c0 = colors[nc[0]]
                        mono = True
                        for oe_idx in nc[1:]:
                            if colors[oe_idx] != c0:
                                mono = False
                                break
                        if mono != cycle_mono[nc_idx]:
                            if mono:
                                mono_count += 1
                            else:
                                mono_count -= 1
                            cycle_mono[nc_idx] = mono
                    improved = True
                    if mono_count == 0:
                        return {'n': n, 'colors': colors}
        
        restarts += 1
        
        return None

    # Try n from 17 down to 9
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        result = check_and_search(n)
        if result:
            return result
    
    # Fallback to seed if all fails (should not happen)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

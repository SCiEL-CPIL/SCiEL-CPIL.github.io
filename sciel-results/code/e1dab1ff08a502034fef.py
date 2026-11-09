import random
import time
import math

def construct():
    # Target n=17
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a valid coloring for n=17
    # If it fails, try smaller n
    
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 10: # Stop if not enough time for next attempt
            break
            
        # Precompute cycles for this n
        # This is expensive, so we might want to cache or just do it once if we stick to n=17.
        # But if n=17 fails quickly, we can try n=16.
        # However, the precomputation for n=17 is the bottleneck.
        
        # Let's assume we only try n=17 first. If it takes too long, we return best effort.
        # Actually, let's just try n=17.
        
        if n == 17:
            valid_coloring = find_valid_coloring(17, deadline)
            if valid_coloring is not None:
                return {'n': 17, 'colors': valid_coloring}
        else:
            # For smaller n, we can use the seed logic or a simpler search
            # But the seed already handles n=9.
            # Let's just return the seed result for n<17 if we get here.
            # Or we can run a quick search for n=16 etc.
            # For now, if n=17 fails, we fall back to the known good n=9 from seed?
            # No, we should try to find a better one.
            # But the task says "Maximize n".
            # If we can't find 17, finding 16 is better than 9.
            
            # Let's try a quick search for the current n
            valid_coloring = find_valid_coloring(n, deadline)
            if valid_coloring is not None:
                return {'n': n, 'colors': valid_coloring}
                
    # Fallback to seed result
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def find_valid_coloring(n, deadline):
    # Precompute all 6-cycles
    # Vertices 0..n-1
    # Edges are pairs (i,j) with i<j
    # Map edge (i,j) to index
    edge_index = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edges.append((i,j))
            idx += 1
            
    num_edges = len(edges)
    
    # Generate all 6-cycles
    # A 6-cycle is a set of 6 vertices v0..v5 such that (v0,v1)..(v5,v0) is a cycle.
    # We need to iterate through all combinations of 6 vertices.
    # For each combination, there are (6-1)!/2 = 60 distinct cycles.
    # Total cycles ~ 60 * C(n,6)
    
    cycles = []
    # For efficiency, we store each cycle as a list of 6 edge indices.
    
    import itertools
    
    # This precomputation can be slow for n=17.
    # C(17,6) = 12376. 12376 * 60 = 742560 cycles.
    # Generating 742k cycles in Python might take ~1-2 seconds.
    
    for vertices in itertools.combinations(range(n), 6):
        v = list(vertices)
        # Generate all permutations of v[1:] relative to v[0]?
        # A cycle is defined by the order of vertices.
        # Fix v[0] as start. Permute the rest.
        # (v0, p0, p1, p2, p3, p4)
        # Edges: (v0,p0), (p0,p1), (p1,p2), (p2,p3), (p3,p4), (p4,v0)
        # To avoid duplicates, we can just generate all permutations and check if it's a new set of edges?
        # Or just generate all 60 unique ones.
        # The number of unique cycles on 6 labeled vertices is 60.
        # We can generate them by permuting v[1:5] and fixing v[5]? No.
        # Standard way: fix v0. The remaining 5 vertices form a path from v1 to v5.
        # The number of such paths is 5! / 2? No, the cycle is undirected.
        # Let's just iterate all permutations of the 6 vertices, and use a set to store frozensets of edges to avoid duplicates?
        # That is too slow.
        
        # Better:
        # Fix v[0]. The cycle is v[0]-x1-x2-x3-x4-x5-v[0].
        # x1..x5 is a permutation of v[1:6].
        # To avoid double counting (v0-x1-...-x5-v0 is same as v0-x5-...-x1-v0),
        # we can impose an order on the neighbors?
        # Simpler: Just generate all 120 permutations of v[1:6].
        # Each cycle will be generated twice (forward and backward).
        # We can just store them all and deduplicate later? Or just accept duplicates?
        # If we accept duplicates, our cost function will count them multiple times.
        # That's fine for finding a solution (cost=0 means no cycles at all).
        # But it increases the work.
        
        # Let's just generate all permutations of the 5 other vertices.
        for perm in itertools.permutations(v[1:]):
            cycle_edges = []
            prev = v[0]
            for node in perm:
                u, w = sorted((prev, node))
                cycle_edges.append(edge_index[(u,w)])
                prev = node
            # Close the cycle
            u, w = sorted((prev, v[0]))
            cycle_edges.append(edge_index[(u,w)])
            cycles.append(cycle_edges)
            
    # Deduplicate cycles?
    # It might be safer to deduplicate to ensure cost is accurate.
    # But for simulated annealing, relative changes are what matter.
    # If a cycle is counted 2 times, and we break it, cost drops by 2.
    # This is consistent.
    
    # Build edge_to_cycles index
    edge_to_cycles = [[] for _ in range(num_edges)]
    for i, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(i)
            
    # Simulated Annealing
    # Initial random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Calculate initial cost
    # For each cycle, check if all edges have same color
    cost = 0
    # To speed up, we can just count.
    # But we need to be able to update cost quickly.
    # We will store the color of each edge.
    
    # Helper to check if a cycle is monochromatic given the edge colors
    # We will not store cycle colors separately, we will re-evaluate on the fly?
    # No, that's too slow.
    # We need to update cost incrementally.
    
    # State:
    # colors: list of int
    # cost: int
    
    # Precompute for each cycle, the list of edge indices.
    # When we change edge e from c_old to c_new:
    # For each cycle containing e:
    #   Let other_edges = cycle_edges excluding e.
    #   If all other_edges have color c_old, the cycle was mono c_old.
    #       So it contributed 1 to cost. Now it is not mono. Cost -= 1.
    #   If all other_edges have color c_new, the cycle will be mono c_new.
    #       So it will contribute 1 to cost. Cost += 1.
    #   Note: It's possible both conditions are true? No, c_old != c_new.
    #   It's possible neither is true.
    
    # We need to quickly check "all other edges have color X".
    # We can precompute for each cycle and each color, the number of edges with that color?
    # That would be 742k * 4 integers. ~3M integers. Memory is fine.
    # But updating it on every move is O(1) per cycle.
    
    # Let's use a simpler approach:
    # For each cycle, store the edge indices.
    # When checking a cycle, iterate through its 6 edges and check colors.
    # 6 checks per cycle.
    # ~5400 cycles per edge change.
    # 5400 * 6 = 32400 operations per move.
    # 120 seconds / (32400 ops) ~ 100k moves?
    # Python is slow. 100k moves might be too few to escape local minima.
    
    # Optimization:
    # Store for each cycle:
    # count[color] = number of edges in this cycle with that color.
    # If count[c] == 6, it's monochromatic.
    # When changing edge e from c_old to c_new:
    # For each cycle containing e:
    #   count[c_old] -= 1
    #   count[c_new] += 1
    #   If count[c_old] was 6, cost -= 1
    #   If count[c_new] is 6, cost += 1
    
    # This is much faster.
    
    cycle_counts = []
    for cycle in cycles:
        counts = [0, 0, 0, 0]
        for e_idx in cycle:
            counts[colors[e_idx]] += 1
        cycle_counts.append(counts)
        
    # Recalculate initial cost based on cycle_counts
    cost = 0
    for counts in cycle_counts:
        for c in range(4):
            if counts[c] == 6:
                cost += 1
                break # Only count once per cycle
        
    # SA Loop
    temp = 1.0
    cooling = 0.9999
    best_cost = cost
    best_colors = colors[:]
    
    while time.time() < deadline - 5:
        if cost == 0:
            return colors[:]
            
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        c_old = colors[e_idx]
        
        # Pick random new color
        c_new = random.choice([c for c in range(4) if c != c_old])
        
        delta = 0
        
        # Update costs for cycles containing e_idx
        for c_idx in edge_to_cycles[e_idx]:
            counts = cycle_counts[c_idx]
            
            # Check if it was mono c_old
            was_mono_old = (counts[c_old] == 6)
            # Check if it will be mono c_new
            will_be_mono_new = (counts[c_new] == 5) # Because we are adding one c_new
            
            if was_mono_old:
                delta -= 1
            if will_be_mono_new:
                delta += 1
                
            # Update counts
            counts[c_old] -= 1
            counts[c_new] += 1
            
        # Accept/Reject
        if delta < 0 or random.random() < math.exp(-delta / temp):
            colors[e_idx] = c_new
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
        else:
            # Revert counts
            for c_idx in edge_to_cycles[e_idx]:
                counts = cycle_counts[c_idx]
                counts[c_old] += 1
                counts[c_new] -= 1
                
        temp *= cooling
        
    # If we didn't find 0 cost, return the best found?
    # No, the task requires a VALID coloring.
    # If we can't find a valid one, we should return None to trigger fallback.
    # But wait, if we return an invalid coloring, score is 0.
    # So we must only return if cost == 0.
    
    if cost == 0:
        return colors[:]
        
    return None

import time
import random
import numpy as np

N = 31
M = 91  # number of edges

def count_c4_fast(adj_bitmasks):
    """Count 4-cycles using numpy bitmasks. adj_bitmasks[i] is int with bits for neighbors of i."""
    n = len(adj_bitmasks)
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = (adj_bitmasks[u] & adj_bitmasks[v]).bit_count()
            if common >= 2:
                c4_count += common * (common - 1) // 2
    return c4_count

def build_bitmasks(edges):
    """Build adjacency bitmasks from edge list."""
    masks = [0] * N
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5) which has 31 vertices and 90 edges,
    with very few C4s.
    """
    points = []
    reps = {}
    idx = 0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = pow(a, 4, 5)
                    rep = (1, b * inv_a % 5, c * inv_a % 5)
                elif b != 0:
                    inv_b = pow(b, 4, 5)
                    rep = (0, 1, c * inv_b % 5)
                else:
                    rep = (0, 0, 1)
                
                if rep not in reps:
                    reps[rep] = idx
                    points.append(rep)
                    idx += 1
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    edges = []
    for i in range(31):
        for j in range(i + 1, 31):
            a, b, c = points[i]
            d, e, f = points[j]
            val = (a * e + b * d + c * f) % 5
            if val == 0:
                edges.append((i, j))
    
    return edges

def delta_add_edge_fast(u, v, adj_bitmasks, edge_bitmask_matrix):
    """Count new C4s created by adding edge (u,v). edge_bitmask_matrix[i] is bitmask of neighbors of i."""
    # New C4s: paths u-a-b-v where a,b are common neighbors... wait, we need u-a, a-b, b-v, v-u
    # Actually new C4s containing the new edge (u,v): u-a-b-v-u where a-b is an edge
    # So for each a in N(u) and b in N(v) with a!=b and edge(a,b) exists, we get a C4
    common = adj_bitmasks[u] & adj_bitmasks[v]
    # For each pair (a,b) where a in N(u)\{v}, b in N(v)\{u}, a!=b, and edge(a,b) exists
    # This equals: for each a in N(u)\{v}, count neighbors b of a that are in N(v)\{u} and b != a
    # Better: iterate over a in N(u), b in N(v), check if edge(a,b)
    # Using bitmasks: for each a in N(u), check if (adj[a] & adj[v] & ~adj[u]) has any bits
    # Actually simpler: 
    # new_c4s = sum over a in N(u)\{v} of |N(a) ∩ N(v) \ {u,a}|
    # But we need a-b edge AND b-v edge AND b != a
    # Let me think again: C4 is u-a-b-v-u. Edges: (u,a), (a,b), (b,v), (v,u=new)
    # So a must be neighbor of u, b must be neighbor of v, and (a,b) must be edge, a != b
    new_c4s = 0
    neighbors_u = adj_bitmasks[u] & ~(1 << v)
    neighbors_v = adj_bitmasks[v] & ~(1 << u)
    # For each a in neighbors_u, count b in neighbors_v such that edge(a,b) exists and b != a
    a_mask = neighbors_u
    while a_mask:
        a_bit = a_mask & (-a_mask)
        a = a_bit.bit_length() - 1
        a_mask ^= a_bit
        # b must be in neighbors_v, b != a, and edge(a,b) exists
        b_candidates = neighbors_v & ~(1 << a) & adj_bitmasks[a]
        new_c4s += b_candidates.bit_count()
    return new_c4s

def count_c4s_containing_edge_fast(u, v, adj_bitmasks):
    """Count C4s that contain edge (u,v). C4: u-a-b-v-u, edges (u,a),(a,b),(b,v),(v,u)"""
    neighbors_u = adj_bitmasks[u] & ~(1 << v)
    neighbors_v = adj_bitmasks[v] & ~(1 << u)
    count = 0
    a_mask = neighbors_u
    while a_mask:
        a_bit = a_mask & (-a_mask)
        a = a_bit.bit_length() - 1
        a_mask ^= a_bit
        b_candidates = neighbors_v & ~(1 << a) & adj_bitmasks[a]
        count += b_candidates.bit_count()
    return count

def local_search_improve(edges, time_budget, rng):
    """
    Improve edge set via local search with fast bitmask-based delta computations.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_edges = list(edges)
    current_edge_set = set(current_edges)
    current_bitmasks = build_bitmasks(current_edges)
    
    current_c4s = count_c4_fast(current_bitmasks)
    best_c4s = current_c4s
    best_edges = list(current_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Simulated annealing parameters
    temperature = 3.0
    cooling_rate = 0.998
    min_temperature = 0.05
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and temperature > min_temperature:
        made_progress = False
        
        # Shuffle edges for diversity
        shuffled_indices = list(range(len(current_edges)))
        rng.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            if time.time() > deadline:
                break
            
            u1, v1 = current_edges[idx]
            
            # Count C4s containing edge (u1,v1)
            lost = count_c4s_containing_edge_fast(u1, v1, current_bitmasks)
            
            # Try to find a non-edge that, when added, creates fewer C4s than lost
            best_gain = float('inf')
            best_edge = None
            
            # Get non-edges
            non_edges = [(u2, v2) for u2, v2 in all_edges if (u2, v2) not in current_edge_set]
            
            # Try all non-edges (there are only 465-91=374)
            for u2, v2 in non_edges:
                gain = delta_add_edge_fast(u2, v2, current_bitmasks, current_edge_set)
                if gain < best_gain:
                    best_gain = gain
                    best_edge = (u2, v2)
            
            if best_edge is not None:
                u2, v2 = best_edge
                u2n = min(u2, v2)
                v2n = max(u2, v2)
                
                delta = best_gain - lost
                
                # Acceptance criterion: simulated annealing
                if delta < 0:
                    accept = True
                else:
                    if temperature > min_temperature:
                        accept = rng.random() < np.exp(-delta / temperature)
                    else:
                        accept = False
                
                if accept:
                    # Perform swap
                    current_edges[idx] = (u2, v2)
                    current_edge_set.discard((min(u1, v1), max(u1, v1)))
                    current_edge_set.add((u2n, v2n))
                    # Update bitmasks
                    current_bitmasks[u1] &= ~(1 << v1)
                    current_bitmasks[v1] &= ~(1 << u1)
                    current_bitmasks[u2] |= (1 << v2)
                    current_bitmasks[v2] |= (1 << u2)
                    
                    current_c4s += delta
                    made_progress = True
                    
                    if current_c4s < best_c4s:
                        best_c4s = current_c4s
                        best_edges = list(current_edges)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    break  # Restart scan after accepting a move
        
        if not made_progress:
            no_improve_count += 5
            if no_improve_count > max_no_improve:
                # Random perturbation to escape local minimum
                for _ in range(8):
                    idx = rng.randint(0, len(current_edges) - 1)
                    u1, v1 = current_edges[idx]
                    u1n, v1n = min(u1, v1), max(u1, v1)
                    
                    non_edges = [(u, v) for u, v in all_edges if (u, v) not in current_edge_set]
                    if non_edges:
                        u2, v2 = non_edges[rng.randint(0, len(non_edges)-1)]
                        u2n, v2n = min(u2, v2), max(u2, v2)
                        
                        lost = count_c4s_containing_edge_fast(u1, v1, current_bitmasks)
                        gain = delta_add_edge_fast(u2, v2, current_bitmasks, current_edge_set)
                        
                        current_edges[idx] = (u2, v2)
                        current_edge_set.discard((u1n, v1n))
                        current_edge_set.add((u2n, v2n))
                        current_bitmasks[u1] &= ~(1 << v1)
                        current_bitmasks[v1] &= ~(1 << u1)
                        current_bitmasks[u2] |= (1 << v2)
                        current_bitmasks[v2] |= (1 << u2)
                        current_c4s += gain - lost
                        
                        if current_c4s < best_c4s:
                            best_c4s = current_c4s
                            best_edges = list(current_edges)
                
                no_improve_count = 0
                temperature = max(temperature * cooling_rate, min_temperature)
            else:
                temperature = max(temperature * cooling_rate, min_temperature)
        else:
            temperature = max(temperature * cooling_rate, min_temperature)
    
    return best_edges

def construct():
    """
    Main entry point: construct a graph with 91 edges on 31 vertices minimizing C4 count.
    Strategy: Start with polarity graph of PG(2,5) (90 edges, very few C4s),
    then greedily add the best edge, then do local search with simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Step 1: Construct polarity graph
    pol_edges = polarity_graph_q5()
    
    # Step 2: If we have exactly 90 edges, greedily add 1 more
    if len(pol_edges) == 90:
        edge_set = set(pol_edges)
        adj_bitmasks = build_bitmasks(pol_edges)
        
        # Greedily add the edge that minimizes new C4s
        best_edge = None
        best_new = float('inf')
        
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edge_set:
                    continue
                new_c4s = delta_add_edge_fast(u, v, adj_bitmasks, edge_set)
                if new_c4s < best_new:
                    best_new = new_c4s
                    best_edge = (u, v)
        
        if best_edge:
            final_edges = list(pol_edges) + [best_edge]
        else:
            final_edges = list(pol_edges)
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        break
                if len(final_edges) > len(pol_edges):
                    break
    elif len(pol_edges) > 90:
        final_edges = list(pol_edges[:90])
    else:
        final_edges = list(pol_edges)
        edge_set = set(final_edges)
        for u in range(N):
            for v in range(u + 1, N):
                if len(final_edges) >= 91:
                    break
                if (u, v) not in edge_set:
                    final_edges.append((u, v))
                    edge_set.add((u, v))
            if len(final_edges) >= 91:
                break
    
    # Ensure exactly 91 edges
    if len(final_edges) != 91:
        edge_set = set(final_edges)
        if len(final_edges) < 91:
            for u in range(N):
                for v in range(u + 1, N):
                    if len(final_edges) >= 91:
                        break
                    if (u, v) not in edge_set:
                        final_edges.append((u, v))
                        edge_set.add((u, v))
                if len(final_edges) >= 91:
                    break
        else:
            final_edges = final_edges[:91]
    
    # Step 3: Local search with simulated annealing to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        improved_edges = local_search_improve(final_edges, remaining_time, rng)
        if improved_edges and len(improved_edges) == 91:
            final_edges = improved_edges
    
    # Convert to required format: list of [u, v]
    result = [[u, v] for u, v in final_edges]
    
    # Final validation
    if len(result) != 91:
        result = []
        edge_set = set()
        for u in range(N):
            for v in range(u + 1, N):
                if len(result) >= 91:
                    break
                result.append([u, v])
                edge_set.add((u, v))
            if len(result) >= 91:
                break
    
    return result

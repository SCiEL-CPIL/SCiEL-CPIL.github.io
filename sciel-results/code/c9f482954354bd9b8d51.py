import time
import random
import numpy as np

N = 31
M = 91  # number of edges

def make_bitmask(adj):
    """Convert adjacency list to numpy bitmask."""
    mask = np.zeros((N, N), dtype=np.int64)
    for u, v in adj:
        mask[u, v] = 1
        mask[v, u] = 1
    return mask

def count_c4_mask(mask):
    """Count 4-cycles using bitmask. Returns exact count."""
    # For each pair (u,v), count common neighbors
    # A 4-cycle u-a-v-b-u exists when u,v share common neighbors a,b
    # Number of C4s = sum over all unordered pairs {u,v} of C(common_neighbors(u,v), 2)
    total = 0
    for u in range(N):
        for v in range(u + 1, N):
            common = np.bitwise_and(mask[u], mask[v])
            cnt = np.sum(common)
            if cnt >= 2:
                total += cnt * (cnt - 1) // 2
    return total

def count_c4(adj_list):
    """Count 4-cycles in a graph given adjacency list."""
    mask = make_bitmask(adj_list)
    return count_c4_mask(mask)

def fast_c4_count(edges):
    """Count 4-cycles using edge list."""
    mask = make_bitmask(edges)
    return count_c4_mask(mask)

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

def count_c4s_containing_edge_mask(u, v, mask):
    """Count C4s that contain edge (u,v) using bitmask."""
    # C4s containing edge (u,v): paths u-a-b-v where a-b is an edge
    # a must be neighbor of u (not v), b must be neighbor of v (not u), and a-b must be edge
    neighbors_u = np.where(mask[u] == 1)[0]
    neighbors_v = np.where(mask[v] == 1)[0]
    
    count = 0
    for a in neighbors_u:
        if a == v:
            continue
        for b in neighbors_v:
            if b == u or b == a:
                continue
            if mask[a, b] == 1:
                count += 1
    return count

def delta_add_edge_mask(u, v, mask):
    """Count new C4s created by adding edge (u,v) using bitmask."""
    # New C4s: u-a-b-v where a is neighbor of u, b is neighbor of v, a!=b, and a-b is edge
    # Note: this counts C4s that use the new edge (u,v)
    neighbors_u = np.where(mask[u] == 1)[0]
    neighbors_v = np.where(mask[v] == 1)[0]
    
    new_c4s = 0
    for a in neighbors_u:
        for b in neighbors_v:
            if a == b:
                continue
            if mask[a, b] == 1:
                new_c4s += 1
    return new_c4s

def local_search_improve(edges, time_budget):
    """
    Improve edge set via local search: try swapping edges to reduce C4 count.
    Uses numpy bitmask for fast delta computation.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_edges = list(edges)
    current_edge_set = set(current_edges)
    mask = make_bitmask(current_edges)
    
    current_c4s = count_c4_mask(mask)
    best_c4s = current_c4s
    best_edges = list(current_edges)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        made_progress = False
        
        for idx, (u1, v1) in enumerate(current_edges):
            if time.time() > deadline:
                break
            
            u1n = min(u1, v1)
            v1n = max(u1, v1)
            
            # Count C4s containing edge (u1,v1)
            lost = count_c4s_containing_edge_mask(u1, v1, mask)
            
            # Try to find a non-edge that, when added, creates fewer C4s than lost
            best_gain = float('inf')
            best_edge = None
            
            for u2, v2 in all_edges:
                if (u2, v2) in current_edge_set:
                    continue
                
                gain = delta_add_edge_mask(u2, v2, mask)
                
                if gain < best_gain:
                    best_gain = gain
                    best_edge = (u2, v2)
            
            if best_edge is not None and best_gain < lost:
                u2, v2 = best_edge
                u2n = min(u2, v2)
                v2n = max(u2, v2)
                
                # Perform swap
                current_edges[idx] = (u2, v2)
                current_edge_set.discard((u1n, v1n))
                current_edge_set.add((u2n, v2n))
                mask[u1, v1] = 0
                mask[v1, u1] = 0
                mask[u2, v2] = 1
                mask[v2, u2] = 1
                
                delta = best_gain - lost
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
                break
            # Try random perturbation to escape local minima
            if no_improve_count >= 10:
                for _ in range(2):
                    idx = random.randint(0, len(current_edges) - 1)
                    u1, v1 = current_edges[idx]
                    u1n, v1n = min(u1, v1), max(u1, v1)
                    
                    # Find a random non-edge
                    non_edges = [(u, v) for u, v in all_edges if (u, v) not in current_edge_set]
                    if non_edges:
                        u2, v2 = random.choice(non_edges)
                        u2n, v2n = min(u2, v2), max(u2, v2)
                        
                        lost = count_c4s_containing_edge_mask(u1, v1, mask)
                        gain = delta_add_edge_mask(u2, v2, mask)
                        
                        current_edges[idx] = (u2, v2)
                        current_edge_set.discard((u1n, v1n))
                        current_edge_set.add((u2n, v2n))
                        mask[u1, v1] = 0
                        mask[v1, u1] = 0
                        mask[u2, v2] = 1
                        mask[v2, u2] = 1
                        current_c4s += gain - lost
                        
                        if current_c4s < best_c4s:
                            best_c4s = current_c4s
                            best_edges = list(current_edges)
                            no_improve_count = 0
    
    return best_edges

def construct():
    """
    Main entry point: construct a graph with 91 edges on 31 vertices minimizing C4 count.
    Strategy: Start with polarity graph of PG(2,5) (90 edges, very few C4s),
    then greedily add the best edge, then do local search.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Step 1: Construct polarity graph
    pol_edges = polarity_graph_q5()
    
    # Step 2: If we have exactly 90 edges, greedily add 1 more
    if len(pol_edges) == 90:
        edge_set = set(pol_edges)
        mask = make_bitmask(pol_edges)
        
        # Greedily add the edge that minimizes new C4s
        best_edge = None
        best_new = float('inf')
        
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edge_set:
                    continue
                new_c4s = delta_add_edge_mask(u, v, mask)
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
    
    # Step 3: Local search to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        improved_edges = local_search_improve(final_edges, remaining_time)
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

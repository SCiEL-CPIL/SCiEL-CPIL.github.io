import random
import time

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency bitmasks from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _max_independent_set_bnb(adj, n, time_limit=10.0, initial_bound=None):
    """
    Find the Maximum Independent Set (MIS) using Branch and Bound.
    Returns the size of the MIS and the set of vertices in the MIS.
    """
    start_time = time.time()
    best_size = 0
    best_set = 0
    
    if initial_bound is not None:
        # initial_bound is the size of a known independent set
        best_size = initial_bound
        # We don't easily have the set from just the bound, so we might need to 
        # reconstruct or just use size for pruning. 
        # Actually, for the final answer, we need the set. 
        # Let's assume the greedy step provides the set.
        # If we pass initial_bound, we should also pass the set.
        # Let's modify signature or handle internally.
        pass

    # Greedy coloring to get an upper bound on the size of MIS in a subgraph
    # represented by a bitmask.
    def greedy_color_bound(candidates):
        """
        Greedy coloring of the subgraph induced by 'candidates'.
        The number of colors used is an upper bound on the size of the MIS.
        """
        if candidates == 0:
            return 0
        
        colors = 0
        remaining = candidates
        # We want to partition 'remaining' into independent sets (color classes).
        # Each color class is an independent set.
        # The max size of an independent set in the subgraph is <= number of color classes?
        # No, the size of the MIS is <= number of vertices.
        # The bound is: The MIS is a subset of one color class? No.
        # Theorem: In any graph, the size of the MIS is at most the number of colors in a valid coloring?
        # No, that's for Vertex Cover. 
        # For Independent Set: A valid coloring partitions V into independent sets.
        # The MIS is contained in one of these independent sets? No.
        # The MIS can take vertices from multiple color classes.
        
        # Correct Bound for MIS using Coloring:
        # If we color the graph with k colors, the MIS size is NOT bounded by k.
        # Wait, the standard B&B for MIS uses coloring of the *complement*?
        # Let's stick to the standard: 
        # To bound MIS, we can use the fact that the graph can be covered by k cliques? No.
        
        # Let's use the standard technique for Maximum Independent Set B&B:
        # The bound is derived from a greedy coloring of the candidate graph.
        # Actually, the common bound is:
        # If we color the candidate graph with k colors, the size of the MIS is <= k?
        # NO. The size of the MIS is <= the number of vertices.
        # The size of the *Vertex Cover* is >= n - k (if k is colors of complement?).
        
        # Let's re-evaluate.
        # Problem: Max Independent Set.
        # Equivalent: Min Vertex Cover.
        # B&B for Min Vertex Cover:
        # Bound: If we can cover the remaining graph with k edges (matching), then VC >= k.
        # Or, if we color the graph, VC >= n - k (where k is number of colors? No).
        
        # Let's use the "Branch on Vertex" approach with a simple bound:
        # Upper bound on MIS size = number of remaining vertices. (Weak)
        # Better bound: Greedy coloring of the *complement* graph gives a lower bound on VC?
        
        # Let's use a simpler, robust B&B for MIS:
        # 1. Pick a vertex v.
        # 2. Branch 1: Include v. Remove v and its neighbors.
        # 3. Branch 2: Exclude v. Remove v.
        # 4. Bound: If current_size + popcount(remaining) <= best_size, prune.
        # This is exponential but n=40 is small. With good ordering, it's fast.
        
        # To improve, we can use a greedy heuristic to find a large independent set in 'candidates' 
        # to update best_size immediately (incumbent).
        
        pass

    # Let's implement a standard B&B for MIS with incumbent updates.
    
    def popcount(x):
        return bin(x).count('1')

    # Greedy MIS to get an initial incumbent
    def greedy_mis(candidates):
        mis = 0
        rem = candidates
        while rem:
            # Pick vertex with min degree in rem? Or just first?
            # Let's pick the vertex with the fewest neighbors in rem to keep options open.
            # Or just iterate.
            v_idx = -1
            min_deg = n
            temp = rem
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = popcount(adj[v] & rem)
                if deg < min_deg:
                    min_deg = deg
                    v_idx = v
            if v_idx == -1:
                break
            mis |= (1 << v_idx)
            # Remove v and its neighbors
            rem &= ~(adj[v_idx] | (1 << v_idx))
        return mis, popcount(mis)

    # Initial incumbent
    inc_set, inc_size = greedy_mis((1 << n) - 1)
    best_size = inc_size
    best_set = inc_set

    # If n is small, we can try to solve exactly.
    # B&B state: (current_set, candidates)
    # We want to maximize popcount(current_set | independent_subset_of_candidates)
    
    def bnb(current_set, candidates):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        if candidates == 0:
            if popcount(current_set) > best_size:
                best_size = popcount(current_set)
                best_set = current_set
            return
        
        # Bound: Max possible size is current_size + popcount(candidates)
        if popcount(current_set) + popcount(candidates) <= best_size:
            return
            
        # Update incumbent with greedy solution on candidates
        g_set, g_size = greedy_mis(candidates)
        if popcount(current_set | g_set) > best_size:
            best_size = popcount(current_set | g_set)
            best_set = current_set | g_set
            
        # Select a vertex to branch on
        # Heuristic: Pick vertex with highest degree in candidates? 
        # Or vertex that appears in many potential conflicts?
        # Let's pick the vertex with the highest degree in the induced subgraph of candidates.
        v_idx = -1
        max_deg = -1
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = popcount(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = v
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        # New candidates: candidates minus v_idx and its neighbors
        new_candidates = candidates & ~(adj[v_idx] | v_bit)
        bnb(current_set | v_bit, new_candidates)
        
        # Branch 2: Exclude v_idx
        # New candidates: candidates minus v_idx
        new_candidates = candidates & ~v_bit
        bnb(current_set, new_candidates)

    bnb(0, (1 << n) - 1)
    
    return best_size, best_set

def _min_vertex_cover(adj, n, time_budget=10.0):
    """
    Find Minimum Vertex Cover by finding Maximum Independent Set.
    MVC = V \ MIS.
    """
    mis_size, mis_set = _max_independent_set_bnb(adj, n, time_limit=time_budget)
    
    # The vertex cover is the complement of the independent set
    all_vertices = (1 << n) - 1
    vc_set = all_vertices & ~mis_set
    
    # Convert bitmask to list of vertex indices
    vc_list = []
    temp = vc_set
    while temp:
        v = (temp & -temp).bit_length() - 1
        vc_list.append(v)
        temp &= temp - 1
        
    return sorted(vc_list)

def _validate_solution(adj, selected, n):
    """Check if selected vertices cover all edges (which implies covering all maximal cliques >= 2)."""
    selected_set = set(selected)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                if u not in selected_set and v not in selected_set:
                    return False
    return True

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time budget per instance
    # n=30 is fast, n=40 is slower. 
    # Let's give 40s each, total 120s.
    time_per_instance = 40.0
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
            # Fallback: return empty or greedy if out of time
            # Greedy is fast
            adj = _build_graph(n, p, seed)
            # Simple greedy for MVC: pick vertex with max degree, remove it and incident edges
            degs = [bin(adj[i]).count('1') for i in range(n)]
            selected = []
            remaining_edges = adj[:]
            for _ in range(n):
                if not any(remaining_edges):
                    break
                # Find max degree vertex
                max_deg = -1
                max_v = -1
                for i in range(n):
                    if degs[i] > max_deg:
                        max_deg = degs[i]
                        max_v = i
                if max_v == -1:
                    break
                selected.append(max_v)
                # Update degrees
                for j in range(n):
                    if remaining_edges[max_v] & (1 << j):
                        degs[j] -= 1
                # Remove edges incident to max_v
                for j in range(n):
                    if remaining_edges[max_v] & (1 << j):
                        remaining_edges[j] &= ~(1 << max_v)
                        remaining_edges[max_v] &= ~(1 << j)
                remaining_edges[max_v] = 0
                degs[max_v] = 0
            results.append(sorted(selected))
            continue
            
        adj = _build_graph(n, p, seed)
        
        # Use B&B for MVC
        # Adjust time budget based on n
        if n <= 30:
            t_budget = 10.0
        elif n <= 35:
            t_budget = 20.0
        else:
            t_budget = 30.0
            
        # Ensure we don't exceed global deadline
        remaining_global = deadline - time.time() - 5
        t_budget = min(t_budget, remaining_global)
        
        if t_budget < 1.0:
             # Fallback to greedy
             degs = [bin(adj[i]).count('1') for i in range(n)]
             selected = []
             remaining_edges = adj[:]
             for _ in range(n):
                 if not any(remaining_edges):
                     break
                 max_deg = -1
                 max_v = -1
                 for i in range(n):
                     if degs[i] > max_deg:
                         max_deg = degs[i]
                         max_v = i
                 if max_v == -1:
                     break
                 selected.append(max_v)
                 for j in range(n):
                     if remaining_edges[max_v] & (1 << j):
                         degs[j] -= 1
                 for j in range(n):
                     if remaining_edges[max_v] & (1 << j):
                         remaining_edges[j] &= ~(1 << max_v)
                         remaining_edges[max_v] &= ~(1 << j)
                 remaining_edges[max_v] = 0
                 degs[max_v] = 0
             results.append(sorted(selected))
             continue

        solution = _min_vertex_cover(adj, n, time_budget=t_budget)
        
        # Validate just in case
        if not _validate_solution(adj, solution, n):
             # This should not happen if B&B is correct, but fallback to greedy
             degs = [bin(adj[i]).count('1') for i in range(n)]
             selected = []
             remaining_edges = adj[:]
             for _ in range(n):
                 if not any(remaining_edges):
                     break
                 max_deg = -1
                 max_v = -1
                 for i in range(n):
                     if degs[i] > max_deg:
                         max_deg = degs[i]
                         max_v = i
                 if max_v == -1:
                     break
                 selected.append(max_v)
                 for j in range(n):
                     if remaining_edges[max_v] & (1 << j):
                         degs[j] -= 1
                 for j in range(n):
                     if remaining_edges[max_v] & (1 << j):
                         remaining_edges[j] &= ~(1 << max_v)
                         remaining_edges[max_v] &= ~(1 << j)
                 remaining_edges[max_v] = 0
                 degs[max_v] = 0
             solution = sorted(selected)

        results.append(solution)
        
    return results

# EVOLVE-BLOCK-END

<<<<<<< SEARCH
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    idx_to_node = {i: node for i, node in enumerate(valid_nodes)}

    # Build conflict bitsets
    conflict_bits = [0] * n
    for i, node in enumerate(valid_nodes):
        for s in squares:
            j = s - node
            if 1 <= j <= N and j != node and j in valid_set:
                j_idx = node_to_idx[j]
                conflict_bits[i] |= (1 << j_idx)

    # Greedy min-degree independent set as initial lower bound
    remaining = (1 << n) - 1
    selected_bits = 0

    while remaining:
        best_node = -1
        best_degree = n + 1
        for i in range(n):
            if (remaining >> i) & 1:
                deg = bin(conflict_bits[i] & remaining).count('1')
                if deg < best_degree:
                    best_degree = deg
                    best_node = i
        selected_bits |= (1 << best_node)
        remaining &= ~(1 << best_node)
        remaining &= ~conflict_bits[best_node]

    best_size = bin(selected_bits).count('1')
    best_solution_bits = selected_bits

    # Local search: greedy add
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        for i in range(n):
            if (best_solution_bits >> i) & 1:
                continue
            if conflict_bits[i] & best_solution_bits:
                continue
            best_solution_bits |= (1 << i)
            best_size += 1
            improved = True

    # Swap-based local search with bitsets
    for _ in range(200):
        if time.time() > deadline - 15.0:
            break
        improved = False
        unselected = [i for i in range(n) if not ((best_solution_bits >> i) & 1)]
        for a in unselected:
            if (best_solution_bits >> a) & 1:
                continue
            conflicting = [b for b in range(n) if (best_solution_bits >> b) & 1 and (conflict_bits[a] >> b) & 1]
            if len(conflicting) <= 1:
                continue
            removed_bits = 0
            for b in conflicting:
                removed_bits |= (1 << b)
            candidate_bits = (best_solution_bits & ~removed_bits) | (1 << a)
            temp_bits = candidate_bits
            temp_size = bin(candidate_bits).count('1')
            for c in unselected:
                if c == a or (removed_bits >> c) & 1:
                    continue
                if (temp_bits >> c) & 1:
                    continue
                if conflict_bits[c] & temp_bits:
                    continue
                temp_bits |= (1 << c)
                temp_size += 1
            if temp_size > best_size:
                best_solution_bits = temp_bits
                best_size = temp_size
                improved = True
                break
        if not improved:
            break

    # Try multiple random restarts of greedy min-degree to find better starting points
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Randomize the order for tie-breaking in min-degree
        remaining = (1 << n) - 1
        sel_bits = 0
        
        while remaining:
            best_node = -1
            best_degree = n + 1
            # Collect all nodes with minimum degree, pick randomly among them
            min_deg_nodes = []
            for i in range(n):
                if (remaining >> i) & 1:
                    deg = bin(conflict_bits[i] & remaining).count('1')
                    if deg < best_degree:
                        best_degree = deg
                        min_deg_nodes = [i]
                    elif deg == best_degree:
                        min_deg_nodes.append(i)
            if min_deg_nodes:
                best_node = random.choice(min_deg_nodes)
            sel_bits |= (1 << best_node)
            remaining &= ~(1 << best_node)
            remaining &= ~conflict_bits[best_node]
        
        cur_size = bin(sel_bits).count('1')
        
        # Greedy add
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(n):
                if (sel_bits >> i) & 1:
                    continue
                if conflict_bits[i] & sel_bits:
                    continue
                sel_bits |= (1 << i)
                cur_size += 1
                improved = True
        
        if cur_size > best_size:
            best_size = cur_size
            best_solution_bits = sel_bits

    # Final verification
    result = [idx_to_node[i] for i in range(n) if (best_solution_bits >> i) & 1]
    result.sort()
    
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                # Fallback to a known valid set
                return list(range(1, 257, 3))
    
    if not result:
        return [1]
    
    return result
=======
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n = len(valid_nodes)
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    idx_to_node = {i: node for i, node in enumerate(valid_nodes)}

    # Build complement graph adjacency (independent set in original = clique in complement)
    # Two vertices are adjacent in complement if they are NOT in conflict in original
    complement_adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j are NOT in conflict
            if not (conflict_exists(i, j, valid_nodes, squares, node_to_idx, N)):
                complement_adj[i] |= (1 << j)
                complement_adj[j] |= (1 << i)

    def conflict_exists(i, j, valid_nodes, squares, node_to_idx, N):
        """Check if nodes i and j have a+b as perfect square"""
        node_i = valid_nodes[i]
        node_j = valid_nodes[j]
        s = node_i + node_j
        r = math.isqrt(s)
        return r * r == s

    # Rebuild complement_adj properly
    complement_adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            node_i = valid_nodes[i]
            node_j = valid_nodes[j]
            s = node_i + node_j
            r = math.isqrt(s)
            if r * r != s:
                complement_adj[i] |= (1 << j)
                complement_adj[j] |= (1 << i)

    # Maximum clique via branch-and-bound with greedy coloring
    best_clique = [0] * n
    best_clique_size = 0
    time_limit = deadline - 2.0

    def greedy_color(candidates, adj):
        """Greedy coloring of candidates, returns (color_assignment, num_colors)"""
        color_assignment = {}
        uncolored = set(candidates)
        color = 0
        while uncolored:
            color += 1
            # Find maximal independent set in remaining uncolored
            available = set(uncolored)
            while available:
                # Pick vertex with highest degree in available
                v = max(available, key=lambda x: bin(adj[x] & sum((1 << a for a in available)).__class__(0)).count('1') if False else 0)
                # Simpler: just pick first
                v = next(iter(available))
                color_assignment[v] = color
                available.discard(v)
                # Remove neighbors from available
                neigh = adj[v]
                available = {x for x in available if not (neigh >> x) & 1}
            uncolored -= available
        return color_assignment, color

    def max_clique(candidates, current_clique, adj, best_size_ref, time_limit_ref):
        """Branch and bound maximum clique"""
        nonlocal best_clique, best_clique_size
        if time.time() > time_limit_ref:
            return
        
        if not candidates:
            if len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                best_clique = current_clique[:]
            return

        # Greedy coloring bound
        cand_list = list(candidates)
        color_assignment = {}
        uncolored = set(cand_list)
        max_color = 0
        while uncolored:
            max_color += 1
            available = set(uncolored)
            while available:
                v = next(iter(available))
                color_assignment[v] = max_color
                available.discard(v)
                neigh = adj[v]
                available = {x for x in available if not (neigh >> x) & 1}
            uncolored -= available

        # Sort candidates by color (ascending) for better pruning
        ordered = sorted(cand_list, key=lambda x: color_assignment[x])

        # Branch and bound
        for i in range(len(ordered) - 1, -1, -1):
            if time.time() > time_limit_ref:
                return
            v = ordered[i]
            # Bound: if current_clique + remaining vertices <= best, prune
            if len(current_clique) + i + 1 <= best_clique_size:
                break
            if color_assignment[v] <= best_clique_size - len(current_clique):
                break

            # Include v
            new_candidates = {x for x in candidates if x != v and (adj[v] >> x) & 1}
            max_clique(new_candidates, current_clique + [v], adj, best_size_ref, time_limit_ref)

            # Exclude v
            candidates.discard(v)

    # Initial greedy clique for lower bound
    def greedy_clique(adj):
        clique = []
        candidates = set(range(n))
        while candidates:
            # Pick vertex with max degree in candidates
            v = max(candidates, key=lambda x: bin(adj[x] & ((1 << n) - 1)).count('1') if False else 0)
            # Simpler: pick first
            v = next(iter(candidates))
            clique.append(v)
            candidates = {x for x in candidates if x != v and (adj[v] >> x) & 1}
        return clique

    init_clique = greedy_clique(complement_adj)
    best_clique_size = len(init_clique)
    best_clique = init_clique[:]

    # Run branch and bound
    all_vertices = set(range(n))
    max_clique(all_vertices, [], complement_adj, best_clique_size, time_limit)

    # Convert clique back to result
    result = [idx_to_node[i] for i in best_clique]
    result.sort()

    # Final verification
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                # Fallback
                return list(range(1, 257, 3))

    if not result:
        return [1]

    return result
>>>>>>> REPLACE

Wait, I made an error - I referenced `conflict_exists` before defining it properly and the complement_adj building is duplicated. Let me fix this with a cleaner approach:


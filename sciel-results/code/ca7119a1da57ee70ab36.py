import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()

    # Try to find valid graphs for n from 36 down to 20
    for n in range(36, 19, -1):
        if time.time() > deadline - 5:
            break

        result = try_find_graph(n, deadline)
        if result and is_valid(result):
            best_result = result
            # If we found 36, that's a new record - return immediately
            if n >= 36:
                return best_result
            # If we found 35, that's the known record - return immediately
            if n >= 35:
                return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check K4: no four vertices all mutually adjacent
    # For each pair (i,j) that is an edge, check if their common neighbors contain an edge
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any two vertices in common are adjacent
                temp = common
                while temp:
                    lsb = temp & (-temp)
                    k = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check if k is adjacent to any other vertex in common
                    if common & adj[k]:
                        return False
    
    # Check independent 6-set: equivalent to K6 in complement
    mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = (~adj[i]) & mask & ~(1 << i)
    
    # Find independent set of size 6 using backtracking with bitmasks
    # Order vertices by degree in complement (descending) for better pruning
    degrees = [bin(comp_adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    def has_independent_set_k(comp_adj, n, k, order):
        """Check if there's an independent set of size k using bitmask backtracking."""
        if k == 0:
            return True
        if k == 1:
            return n > 0
        
        # Use recursive backtracking with candidate set as bitmask
        def backtrack(candidates_mask, remaining):
            if remaining == 0:
                return True
            if candidates_mask == 0:
                return False
            # Pruning: if not enough candidates
            if bin(candidates_mask).count('1') < remaining:
                return False
            
            # Pick vertex with highest degree in the induced subgraph on candidates
            # for better pruning
            best_v = -1
            best_deg = -1
            temp = candidates_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg_in_candidates = bin(comp_adj[v] & candidates_mask).count('1')
                if deg_in_candidates > best_deg:
                    best_deg = deg_in_candidates
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Try including best_v
            new_candidates = candidates_mask & comp_adj[best_v]
            new_candidates &= ~(1 << best_v)
            if backtrack(new_candidates, remaining - 1):
                return True
            
            # Try excluding best_v
            new_candidates2 = candidates_mask & ~(1 << best_v)
            if backtrack(new_candidates2, remaining - 1):
                return True
            
            return False
        
        full_mask = (1 << n) - 1
        return backtrack(full_mask, k)
    
    if has_independent_set_k(comp_adj, n, 6, order):
        return False
    
    return True

def try_find_graph(n, deadline):
    """Use WalkSAT-inspired local search to find a K4-free graph with no independent 6-set."""
    
    def bitset_to_edges(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i] & ((1 << i) - 1)
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                edges.append([i, j])
                temp ^= lsb
        return edges

    def flip_edge(adj, u, v, n):
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def find_k4(adj, n):
        """Find one K4 if it exists, return tuple of 4 vertices or None."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    temp = common
                    while temp:
                        lsb = temp & (-temp)
                        k = lsb.bit_length() - 1
                        temp ^= lsb
                        common_k = common & adj[k]
                        common_k &= ~(1 << k)
                        if common_k:
                            lsb2 = common_k & (-common_k)
                            l = lsb2.bit_length() - 1
                            return (i, j, k, l)
        return None

    def find_indep_6(adj, n):
        """Find one independent set of size 6 if it exists, return tuple or None."""
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = (~adj[i]) & mask & ~(1 << i)
        
        # Backtracking search for independent set of size 6
        # Order by complement degree descending
        degrees = [bin(comp_adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        found = [None]
        
        def backtrack(current, candidates_mask, remaining):
            if found[0] is not None:
                return
            if remaining == 0:
                found[0] = tuple(current)
                return
            if candidates_mask == 0:
                return
            if bin(candidates_mask).count('1') < remaining:
                return
            
            # Pick vertex with highest degree in induced subgraph
            best_v = -1
            best_deg = -1
            temp = candidates_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = bin(comp_adj[v] & candidates_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return
            
            # Try including best_v
            new_mask = candidates_mask & comp_adj[best_v] & ~(1 << best_v)
            backtrack(current + [best_v], new_mask, remaining - 1)
            
            if found[0] is not None:
                return
            
            # Try excluding best_v
            new_mask2 = candidates_mask & ~(1 << best_v)
            backtrack(current, new_mask2, remaining - 1)
        
        full_mask = (1 << n) - 1
        backtrack([], full_mask, 6)
        return found[0]

    def count_k4s(adj, n):
        """Count number of K4s (for scoring)."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Count edges in the induced subgraph on common
                    temp = common
                    while temp:
                        lsb = temp & (-temp)
                        k = lsb.bit_length() - 1
                        temp ^= lsb
                        if common & adj[k]:
                            count += 1
        return count

    def count_indep_6s(adj, n):
        """Count number of independent 6-sets (for scoring). Use sampling for speed."""
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = (~adj[i]) & mask & ~(1 << i)
        
        # Count K6s in complement by sampling
        count = 0
        # Use a limited search to estimate
        degrees = [bin(comp_adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Just count how many 6-sets from top candidates are independent
        top_candidates = order[:20] if len(order) > 20 else order
        for combo in combinations(top_candidates, 6):
            is_indep = True
            for a in range(6):
                for b in range(a + 1, 6):
                    if not (comp_adj[combo[a]] & (1 << combo[b])):
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                count += 1
        return count

    # Main search loop
    max_restarts = 30
    max_iters_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None
        
        # Initialize: start with a random graph biased towards lower density
        # For K4-free, average degree should be modest. For no indep 6, need enough edges.
        # Use p around 0.3-0.4 for n=35
        p = 0.35 if n >= 30 else 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search
        no_improve_count = 0
        best_score = float('inf')
        
        for it in range(max_iters_per_restart):
            if time.time() > deadline - 3:
                return None
            
            # Check for K4
            k4 = find_k4(adj, n)
            if k4 is None:
                # Check for independent 6-set
                indep6 = find_indep_6(adj, n)
                if indep6 is None:
                    # Valid!
                    edges = bitset_to_edges(adj, n)
                    return {'n': n, 'edges': edges}
                else:
                    # Have independent 6-set, need to add edges
                    i, j, k, l, m, p2 = indep6
                    # Pick a random non-edge among the 6 to add
                    pairs = [(i,j), (i,k), (i,l), (i,m), (i,p2),
                             (j,k), (j,l), (j,m), (j,p2),
                             (k,l), (k,m), (k,p2),
                             (l,m), (l,p2),
                             (m,p2)]
                    # Filter to only non-edges
                    non_edges = [(u,v) for u,v in pairs if not (adj[u] & (1 << v))]
                    if non_edges:
                        u, v = random.choice(non_edges)
                        flip_edge(adj, u, v, n)
                    else:
                        # All pairs are edges, which shouldn't happen for an independent set
                        # Pick random edge to flip
                        u, v = random.choice(pairs)
                        flip_edge(adj, u, v, n)
                    no_improve_count = 0
                    continue
            
            # Have K4, need to remove edges
            i, j, k, l = k4
            pairs = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            # Filter to only edges
            edges_in_k4 = [(u,v) for u,v in pairs if (adj[u] & (1 << v))]
            if edges_in_k4:
                u, v = random.choice(edges_in_k4)
                flip_edge(adj, u, v, n)
            else:
                # Shouldn't happen
                u, v = random.choice(pairs)
                flip_edge(adj, u, v, n)
            no_improve_count = 0
        
        # If we exhausted iterations without success, restart
        continue
    
    return None

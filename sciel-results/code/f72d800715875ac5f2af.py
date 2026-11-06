import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 285
    
    def is_clique(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def is_independent(adj, subset):
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                u, v = subset[i], subset[j]
                if (adj[u] >> v) & 1:
                    return False
        return True

    def count_offending_fast(adj, n, size=5):
        """Count K5 and independent 5-sets using bit-neighborhood intersection."""
        count = 0
        non_adj = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
        for comb in combinations(range(n), size):
            # Check clique: all vertices must be neighbors of the first
            common = adj[comb[0]]
            for v in comb[1:]:
                if not (common >> v) & 1:
                    break
                common &= adj[v]
            else:
                count += 1
            # Check independent set
            common = non_adj[comb[0]]
            for v in comb[1:]:
                if not (common >> v) & 1:
                    break
                common &= non_adj[v]
            else:
                count += 1
        return count

    def find_offending_set_fast(adj, n, size=5):
        """Find any K5 or independent 5-set. Returns the set or None."""
        for comb in combinations(range(n), size):
            if is_clique(adj, comb):
                return list(comb)
            if is_independent(adj, comb):
                return list(comb)
        return None

    def verify_graph(adj, n, size=5):
        """Full verification that no K5 or I5 exists."""
        for comb in combinations(range(n), size):
            if is_clique(adj, comb) or is_independent(adj, comb):
                return False
        return True

    def local_search(adj, n, max_iters, deadline):
        """Local search to eliminate offending sets by flipping edges."""
        best_adj = adj[:]
        best_count = count_offending_fast(adj, n)
        
        if best_count == 0:
            return best_adj, 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find an offending set
            offending = find_offending_set_fast(adj, n)
            if offending is None:
                # Verify fully
                if verify_graph(adj, n):
                    return adj, 0
                # If not valid, continue searching
            
            # Try flipping edges in the offending set (or random if none found)
            if offending:
                candidates = list(combinations(offending, 2))
                random.shuffle(candidates)
            else:
                candidates = [(random.randrange(n), random.randrange(n))]
                if candidates[0][0] == candidates[0][1]:
                    candidates[0] = (candidates[0][0], (candidates[0][0]+1) % n)
                if candidates[0][0] > candidates[0][1]:
                    candidates[0] = (candidates[0][1], candidates[0][0])
            
            best_adj_in_iter = adj
            best_count_in_iter = best_count
            
            for u, v in candidates:
                if u == v:
                    continue
                # Flip edge (u, v)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Count offending sets after flip
                new_count = count_offending_fast(adj, n)
                
                if new_count < best_count_in_iter:
                    best_count_in_iter = new_count
                    best_adj_in_iter = adj[:]
                    if new_count == 0:
                        # Unflip and return
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        return best_adj_in_iter, 0
                else:
                    # Unflip
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Update best
            if best_count_in_iter < best_count:
                best_count = best_count_in_iter
                best_adj = best_adj_in_iter
                adj = best_adj_in_iter[:]
        
        return best_adj, best_count

    def try_add_vertex(adj, n, deadline):
        """Try to add a new vertex to the graph while maintaining (5,5)-freeness."""
        new_n = n + 1
        
        # Precompute: for each 4-set S in the original graph,
        # determine which vertices are "forbidden" to connect to the new vertex
        # A vertex v is forbidden if connecting new vertex to v would create
        # a K5 or I5 involving the new vertex.
        # 
        # For a 4-set S = {a,b,c,d}:
        # - If S is a K4 (clique of 4): new vertex must NOT connect to any of S
        #   (otherwise new + S forms K5)
        # - If S is an I4 (independent 4-set): new vertex must NOT be non-adjacent
        #   to any of S (otherwise new + S forms I5), i.e., must connect to ALL of S
        #
        # So for each vertex v, we track:
        #   forbid_connect[v] = set of v that must NOT be connected to new vertex
        #   forbid_disconnect[v] = set of v that MUST be connected to new vertex
        #
        # Actually simpler: for each vertex v, compute the set of vertices that v
        # is "forced" to connect to (must_connect) and "forced" not to connect to
        # (must_not_connect) with respect to the new vertex.
        
        # For each 4-set, check if it's a K4 or I4
        # If K4: all 4 vertices must NOT connect to new vertex
        # If I4: all 4 vertices MUST connect to new vertex
        
        must_connect = [0] * n  # bitmask: vertices that MUST connect to new vertex
        must_not_connect = [0] * n  # bitmask: vertices that MUST NOT connect to new vertex
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check if K4
            is_k4 = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = quad[i], quad[j]
                    if not (adj[u] >> v) & 1:
                        is_k4 = False
                        break
                if not is_k4:
                    break
            if is_k4:
                for v in quad:
                    must_not_connect[v] |= (1 << v)  # v must not connect to new
                continue
            
            # Check if I4
            is_i4 = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = quad[i], quad[j]
                    if (adj[u] >> v) & 1:
                        is_i4 = False
                        break
                if not is_i4:
                    break
            if is_i4:
                for v in quad:
                    must_connect[v] |= (1 << v)  # v must connect to new
        
        # Now try to find a valid connection pattern
        # For each vertex v, the connection to new vertex is determined:
        # - If v is in must_connect: must connect
        # - If v is in must_not_connect: must not connect
        # - Otherwise: free choice
        #
        # If there's a conflict (v in both must_connect and must_not_connect), impossible.
        
        # Check for conflicts
        for v in range(n):
            if (must_connect[v] >> v) & 1 and (must_not_connect[v] >> v) & 1:
                return None
        
        # Build the forced connection pattern
        forced_connect = 0
        for v in range(n):
            if (must_connect[v] >> v) & 1:
                forced_connect |= (1 << v)
        
        # Free vertices: those not forced either way
        forced_mask = 0
        for v in range(n):
            if ((must_connect[v] >> v) & 1) or ((must_not_connect[v] >> v) & 1):
                forced_mask |= (1 << v)
        
        free_mask = ((1 << n) - 1) ^ forced_mask
        free_vertices = [v for v in range(n) if (free_mask >> v) & 1]
        
        # Try different assignments for free vertices
        # Use local search to find a valid assignment
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            # Start with forced connections, random for free
            new_adj = adj[:] + [0]
            for v in range(n):
                if (must_connect[v] >> v) & 1:
                    new_adj[v] |= (1 << n)
                    new_adj[n] |= (1 << v)
                elif (must_not_connect[v] >> v) & 1:
                    pass  # don't connect
                else:
                    if random.random() < 0.5:
                        new_adj[v] |= (1 << n)
                        new_adj[n] |= (1 << v)
            
            # Local search: only flip edges incident to new vertex
            for it in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Quick check: find offending set
                offending = find_offending_set_fast(new_adj, new_n)
                
                if offending is None:
                    if verify_graph(new_adj, new_n):
                        return new_adj
                    break
                
                # If offending set includes new vertex, fix it
                if n in offending:
                    others = [x for x in offending if x != n]
                    if not others:
                        break
                    
                    # Determine which flip would help
                    # If it's a K5 (new + 4 clique), we need to disconnect new from one of the 4
                    # If it's an I5 (new + 4 independent), we need to connect new to one of the 4
                    
                    # Check if it's a clique or independent
                    is_cliq = True
                    is_ind = True
                    for i in range(len(offending)):
                        for j in range(i+1, len(offending)):
                            u, v = offending[i], offending[j]
                            if (u == n or v == n):
                                other = v if u == n else u
                                if (new_adj[n] >> other) & 1:
                                    is_ind = False
                                else:
                                    is_cliq = False
                            else:
                                if (new_adj[u] >> v) & 1:
                                    is_ind = False
                                else:
                                    is_cliq = False
                    
                    if is_cliq:
                        # Need to disconnect new from one of the others
                        # Prefer vertices that are free (not forced)
                        free_others = [x for x in others if not ((must_connect[x] >> x) & 1) and not ((must_not_connect[x] >> x) & 1)]
                        if free_others:
                            target = random.choice(free_others)
                        else:
                            target = random.choice(others)
                        # Disconnect
                        new_adj[n] &= ~(1 << target)
                        new_adj[target] &= ~(1 << n)
                    elif is_ind:
                        # Need to connect new to one of the others
                        free_others = [x for x in others if not ((must_connect[x] >> x) & 1) and not ((must_not_connect[x] >> x) & 1)]
                        if free_others:
                            target = random.choice(free_others)
                        else:
                            target = random.choice(others)
                        # Connect
                        new_adj[n] |= (1 << target)
                        new_adj[target] |= (1 << n)
                    else:
                        # Mixed, just flip a random edge to new vertex
                        target = random.choice(others)
                        if (new_adj[n] >> target) & 1:
                            new_adj[n] &= ~(1 << target)
                            new_adj[target] &= ~(1 << n)
                        else:
                            new_adj[n] |= (1 << target)
                            new_adj[target] |= (1 << n)
                else:
                    # Offending set doesn't include new vertex - this shouldn't happen
                    # since the original graph was valid. But if it does, we can't fix it
                    # by changing new vertex's connections. Break.
                    break
            
            # Final check
            if verify_graph(new_adj, new_n):
                return new_adj
        
        return None

    # Start from Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = q
    current_adj = adj
    
    # Try to add vertices one at a time up to 42
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 30:
            break
        
        # Try multiple times to add a vertex
        success = False
        for attempt in range(30):
            if time.time() > deadline - 30:
                break
            
            new_adj = try_add_vertex(current_adj, current_n, deadline)
            if new_adj is not None:
                current_adj = new_adj
                current_n = target_n
                success = True
                break
        
        if not success:
            break
    
    # Final verification
    if verify_graph(current_adj, current_n):
        edges = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if (current_adj[i] >> j) & 1:
                    edges.append((i, j))
        return current_n, edges
    
    # Fallback to Paley graph on 37
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

import time
import sys

def _polarity_graph(q):
    """
    Generates the polarity graph of the projective plane of order q.
    Vertices are points of PG(2, q).
    Two points are adjacent if one lies on the polar of the other.
    This graph is C4-free.
    """
    if q < 2:
        return 0, []
    
    # Points in PG(2, q)
    # 1. Affine points: (x, y, 1)
    # 2. Points at infinity: (x, 1, 0) and (1, 0, 0)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    n = len(points)
    
    adj = [0] * n  # Bitsets for adjacency
    
    for i in range(n):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n):
            x2, y2, z2 = points[j]
            # Check if x1*x2 + y1*y2 + z1*z2 == 0 mod q
            if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    return n, adj

def _has_independent_set(adj, n, k, time_limit=1.0):
    """
    Returns True if an independent set of size k exists in the graph defined by adj bitsets.
    Uses branch and bound.
    """
    start_time = time.time()
    
    # We are looking for a clique of size k in the complement graph.
    # Complement adjacency: comp[i] has bits set for j where NOT (adj[i] & (1<<j)) and i!=j.
    # Actually, it's easier to work with the original graph directly:
    # We want a set S such that for all u, v in S, (adj[u] & (1<<v)) == 0.
    
    # Precompute non-neighbors (including self for mask convenience, but we handle i!=j)
    # Let's use the standard "available" mask approach.
    
    # Sort vertices by degree (heuristic: high degree in G means low degree in complement, 
    # so picking high degree vertices first for the independent set is harder? 
    # Actually, for independent set, we want vertices that are NOT connected.
    # So in the complement graph, we want a clique. 
    # High degree in G = Low degree in complement.
    # So vertices with low degree in G are good candidates to start with?
    # Let's just sort by degree in G ascending.
    
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    # Remap
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
        
    new_adj = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if adj[i] & (1 << j):
                bits |= (1 << remap[j])
        new_adj[remap[i]] = bits
        
    # Backtracking
    found = [False]
    
    def backtrack(start, current_size, candidates_mask):
        if found[0]:
            return
        if time.time() - start_time > time_limit:
            return
            
        if current_size == k:
            found[0] = True
            return
            
        # Pruning: if number of bits in candidates_mask + current_size < k
        if bin(candidates_mask).count('1') + current_size < k:
            return
            
        # Heuristic: pick a vertex from candidates
        # To maximize chance of finding a set, we might want to pick a vertex
        # that has many non-neighbors in the remaining candidates.
        # Or just iterate.
        
        m = candidates_mask
        while m:
            if found[0]:
                return
            if time.time() - start_time > time_limit:
                return
                
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # If we pick v, the next candidates must be:
            # 1. In candidates_mask
            # 2. > v (to avoid duplicates)
            # 3. Not adjacent to v in the original graph (i.e., not in new_adj[v])
            
            # Mask of vertices > v
            mask_gt_v = candidates_mask & ~((1 << (v + 1)) - 1)
            
            # Mask of vertices not adjacent to v
            # new_adj[v] contains neighbors.
            # We want vertices u such that (new_adj[v] & (1<<u)) == 0.
            # So, valid_next = mask_gt_v & ~new_adj[v]
            
            valid_next = mask_gt_v & ~new_adj[v]
            
            # Recurse
            backtrack(v + 1, current_size + 1, valid_next)
            
    # Initial candidates: all vertices
    initial_mask = (1 << n) - 1
    backtrack(0, 0, initial_mask)
    
    return found[0]

def _expand_graph(adj, n, target_n, deadline):
    """
    Attempts to expand the graph to target_n vertices.
    Returns (new_n, new_adj) or (n, adj) if fails.
    """
    current_n = n
    current_adj = adj[:]
    
    for _ in range(target_n - n):
        if time.time() > deadline:
            break
            
        # Try to add a new vertex 'new_v'
        # It can connect to a subset of existing vertices.
        # Constraint: If new_v connects to u and v, then u and v must not share any common neighbors.
        # Actually, the C4 condition is: No two vertices have >= 2 common neighbors.
        # Adding new_v creates new pairs (new_v, u).
        # The new common neighbors of (new_v, u) are the common neighbors of u and new_v.
        # Since new_v is new, its neighbors are a subset S of existing vertices.
        # For any u in S, the common neighbors of new_v and u are N(new_v) \cap N(u).
        # N(new_v) = S.
        # So we need |S \cap N(u)| <= 1 for all u in S?
        # Wait.
        # If u, v are in S, then u and v are both neighbors of new_v.
        # Do u and v have common neighbors?
        # The C4 constraint applies to ANY pair of vertices in the graph.
        # If we add new_v connected to S, we must ensure:
        # 1. For any u in S, the pair (new_v, u) has <= 1 common neighbor.
        #    Common neighbors of new_v and u are N(new_v) \cap N(u) = S \cap N(u).
        #    So we need |S \cap N(u)| <= 1 for all u in S.
        # 2. The existing graph was C4-free, so pairs not involving new_v are fine.
        
        # So we need to find a set S of vertices such that for all u in S, |S \cap N(u)| <= 1.
        # This is equivalent to saying that S is an independent set in the graph H where
        # H has edges between u and v if |N(u) \cap N(v)| >= 1? No.
        # Let's re-evaluate.
        # Condition: For all u in S, |S \cap N(u)| <= 1.
        # This means that in the subgraph induced by S, no vertex u has more than 1 neighbor in S?
        # No, N(u) is the set of neighbors in the ORIGINAL graph.
        # So if u and v are both in S, and they are adjacent in the original graph, then v \in N(u).
        # So if u and v are in S and adjacent, then v is counted in |S \cap N(u)|.
        # If u and v are in S and NOT adjacent, v is not in N(u).
        #
        # Let's test:
        # If S = {u, v} and u-v is an edge.
        # N(u) contains v. So S \cap N(u) contains v. Size 1. OK.
        # N(v) contains u. So S \cap N(v) contains u. Size 1. OK.
        #
        # If S = {u, v, w} and u-v, v-w are edges (path).
        # For v: N(v) contains u, w. S \cap N(v) = {u, w}. Size 2. FAIL.
        # So S cannot contain a path of length 2 where the middle vertex is in S?
        # Actually, if v is in S, and u, w are in S and adjacent to v, then |S \cap N(v)| >= 2.
        # So v cannot have 2 neighbors in S.
        # This implies that in the subgraph induced by S, the maximum degree is 1?
        # No, if u and w are not adjacent to each other, but both adjacent to v.
        # Then v has 2 neighbors in S.
        # So the condition is: The subgraph induced by S must have max degree <= 1.
        # i.e., S induces a matching (plus isolated vertices).
        
        # So we need to find a set S of vertices such that the subgraph induced by S has max degree <= 1.
        # AND, adding S as neighbors to new_v must keep alpha <= 10.
        
        # To maximize the "strength" of the new vertex (minimize alpha), we want S to be large.
        # But S must be a "matching-like" set in the original graph.
        
        # Let's try to find a large S.
        # Start with S = empty.
        # We can greedily add vertices to S.
        # When adding u to S, we must ensure that no vertex in S is adjacent to u in a way that creates a degree > 1 in the induced subgraph?
        # Actually, if we maintain the property that S induces a matching, we just need to check:
        # If we add u to S, does u have >= 2 neighbors in S?
        # If yes, we can't add u.
        # If u has 1 neighbor v in S, then v's degree in S becomes 2 (if it was 1 before) or 1 (if it was 0).
        # Wait, if v was already in S and had degree 1 (connected to w), and we add u connected to v, then v has degree 2. Fail.
        # So if we add u, u can have at most 1 neighbor in S.
        # AND, for any neighbor v of u in S, v must currently have degree 0 in S (so it becomes 1).
        # If v already has degree 1 in S, adding u makes it 2. Fail.
        
        # So the rule for adding u to S:
        # 1. Count neighbors of u in S. Let this be cnt.
        # 2. If cnt > 1, cannot add.
        # 3. If cnt == 1, let v be the neighbor. Check if v has any other neighbors in S. If yes, cannot add.
        # 4. If cnt == 0, can add.
        
        # This allows S to be a matching.
        
        # We want to find a large S.
        # We can try to build S greedily.
        # To make the search fast, we might just try S = {u} for each u?
        # That would add vertices with degree 1.
        # Or S = {u, v} where u-v is an edge.
        
        # Let's try to find the maximum size S satisfying the condition.
        # This is similar to finding a maximum induced matching? No, just max degree <= 1.
        
        # Heuristic:
        # Try to add vertices one by one.
        # Order vertices by degree in the current graph?
        
        # Let's try a simple approach:
        # For each potential new vertex, we try to find a valid S.
        # We start with S = all vertices, and remove vertices that violate the condition?
        # Or start with S = empty and add.
        
        # Given time constraints, let's try a few specific S patterns:
        # 1. S = {u} (single vertex)
        # 2. S = {u, v} where u~v
        # 3. S = {u, v, w} where u~v, w isolated (or similar matching)
        
        # Actually, the simplest is to just try to add a vertex connected to a single existing vertex.
        # This always satisfies the C4 condition (degree 1 in induced subgraph).
        # We can do this 8 times.
        
        # But will alpha stay <= 10?
        # Adding a vertex connected to only 1 vertex might increase alpha.
        # If the new vertex is not connected to many vertices, it might be part of a large independent set.
        # To minimize this, we want the new vertex to be connected to as many vertices as possible.
        # But the C4 constraint limits this.
        
        # Let's try to find the largest S for each new vertex.
        
        best_S = None
        best_S_size = 0
        
        # We can iterate through subsets? No, too slow.
        # Let's try to build S greedily.
        # Sort vertices by "potential" (e.g., degree).
        
        # For now, let's just try to add a vertex connected to a single vertex u.
        # We choose u such that the resulting graph has the smallest alpha.
        # Or just the first one that works?
        
        # Let's try to be smarter.
        # We want to connect the new vertex to a set S.
        # Let's try S of size 1, then 2, etc.
        
        found_valid_S = False
        
        # Try S of size 1
        # We want to pick u such that adding new_v connected to u keeps alpha <= 10.
        # We can test this.
        
        candidates_for_S = []
        
        # Let's just try to add a vertex connected to the vertex with the highest degree?
        # Or random?
        
        # Let's try to find a valid S by checking all single vertices first.
        # If none work, try pairs.
        
        # Optimization: Precompute "can_connect" for pairs?
        
        # Let's just try to add a vertex connected to a single vertex u.
        # We'll iterate u and check.
        
        for u in range(current_n):
            if time.time() > deadline:
                break
                
            # Try S = {u}
            # New vertex index: current_n
            # New adj for u: add bit current_n
            # New adj for current_n: add bit u
            
            # Check alpha
            # We need to update adj and check.
            # To avoid modifying the main adj array repeatedly, we can simulate.
            # But _has_independent_set takes the whole adj array.
            # We can create a copy.
            
            new_adj_copy = current_adj[:]
            new_adj_copy[u] |= (1 << current_n)
            new_adj_copy.append(1 << u) # adj for new vertex
            
            if not _has_independent_set(new_adj_copy, current_n + 1, 11, time_limit=0.5):
                # Valid!
                current_adj = new_adj_copy
                current_n += 1
                found_valid_S = True
                break
        
        if not found_valid_S:
            # Try S of size 2?
            # This is more complex.
            # For now, if we can't add with S size 1, we stop.
            # This might limit us to n=31 or slightly more if we start with a different base.
            # But PG(2,5) is strong.
            
            # Let's try to add a vertex connected to NO vertices?
            # S = empty.
            # New vertex is isolated.
            # This will definitely increase alpha.
            # If current alpha is 6, adding isolated vertex makes it 7.
            # We can add up to 4 isolated vertices (6+4=10).
            # So we can reach n=35 easily.
            
            # Let's add isolated vertices if we can't add connected ones.
            for _ in range(4):
                if time.time() > deadline:
                    break
                if current_n >= 35:
                    break
                
                new_adj_copy = current_adj[:]
                new_adj_copy.append(0)
                
                if not _has_independent_set(new_adj_copy, current_n + 1, 11, time_limit=0.5):
                    current_adj = new_adj_copy
                    current_n += 1
                else:
                    break
                    
            # If we still haven't reached target, try S of size 2?
            # Let's try to find a pair u, v such that u~v and adding new_v connected to {u,v} works.
            
            if current_n < target_n:
                for u in range(current_n):
                    if time.time() > deadline:
                        break
                    for v in range(u + 1, current_n):
                        if time.time() > deadline:
                            break
                        # Check if u and v are adjacent
                        if current_adj[u] & (1 << v):
                            # Check C4 condition for S={u,v}
                            # u and v are in S.
                            # N(u) \cap S = {v} (since u~v). Size 1. OK.
                            # N(v) \cap S = {u}. Size 1. OK.
                            # So C4 is satisfied.
                            
                            new_adj_copy = current_adj[:]
                            new_adj_copy[u] |= (1 << current_n)
                            new_adj_copy[v] |= (1 << current_n)
                            new_adj_copy.append((1 << u) | (1 << v))
                            
                            if not _has_independent_set(new_adj_copy, current_n + 1, 11, time_limit=0.5):
                                current_adj = new_adj_copy
                                current_n += 1
                                break
                    if current_n >= target_n:
                        break

    return current_n, current_adj

def construct():
    deadline = time.time() + 110  # Leave 10s margin
    
    best_n = 0
    best_adj = []
    
    # Try q=5 (n=31)
    try:
        n, adj = _polarity_graph(5)
        
        # Check if base graph is valid (alpha <= 10)
        if not _has_independent_set(adj, n, 11, time_limit=5.0):
            if n > best_n:
                best_n = n
                best_adj = adj[:]
                
            # Expand
            if time.time() < deadline - 10:
                new_n, new_adj = _expand_graph(adj, n, 39, deadline)
                if new_n > best_n:
                    best_n = new_n
                    best_adj = new_adj
        else:
            # Base graph has alpha >= 11, try to fix? Unlikely.
            pass
    except Exception:
        pass
        
    # Try q=4 (n=21) as fallback
    if time.time() < deadline - 10 and best_n < 21:
        try:
            n, adj = _polarity_graph(4)
            if not _has_independent_set(adj, n, 11, time_limit=5.0):
                if n > best_n:
                    best_n = n
                    best_adj = adj[:]
        except Exception:
            pass
            
    if best_n == 0:
        return {'n': 1, 'edges': []}
        
    # Convert adj bitsets to edges list
    edges = []
    for u in range(best_n):
        for v in range(u + 1, best_n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return {'n': best_n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(res['n'])

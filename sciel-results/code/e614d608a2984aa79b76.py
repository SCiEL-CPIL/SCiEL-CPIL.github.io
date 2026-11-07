import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    n = 15
    
    # Adjacency masks: adj[i] is an integer where bit j is 1 if edge (i,j) exists
    # We use 15 bits (0-14)
    
    def get_edges_from_mask(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i+1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def calculate_degrees(adj):
        deg = [0] * n
        for i in range(n):
            deg[i] = bin(adj[i]).count('1')
        return deg

    def is_valid(adj):
        deg = calculate_degrees(adj)
        # Group vertices by degree
        deg_to_verts = {}
        for i in range(n):
            if deg[i] not in deg_to_verts:
                deg_to_verts[deg[i]] = []
            deg_to_verts[deg[i]].append(i)
        
        # For each group of vertices with the same degree, check if any pair has a 3-path
        for d, verts in deg_to_verts.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s = verts[i]
                    t = verts[j]
                    # Check if there is a path s-u-v-t
                    # u in N(s)\{t}, v in N(t)\{u}, u~v
                    # Optimization:
                    # Let Ns = adj[s] & ~(1<<t)
                    # Let Nt = adj[t] & ~(1<<s)
                    # We need u in Ns, v in Nt such that u~v.
                    # This is equivalent to: (Union of N(u) for u in Ns) intersects Nt?
                    # Or simpler: For each u in Ns, if (adj[u] & Nt) != 0 and u not in Nt? 
                    # Wait, if u is in Nt, then u~t, so u is a neighbor of t.
                    # Path s-u-v-t.
                    # If u~v, and u~s, v~t.
                    # If u is in Nt, then v could be u? No, simple path s-u-v-t implies u!=v.
                    # Also s!=t, u!=t, v!=s.
                    
                    Ns = adj[s] & ~(1 << t)
                    Nt = adj[t] & ~(1 << s)
                    
                    # If there is any u in Ns such that adj[u] intersects Nt at some v != u
                    # Note: if u is in Nt, then u is a neighbor of t.
                    # If we pick v=u, then s-u-u-t is not a simple path.
                    # So we need v in Nt \ {u}.
                    
                    found = False
                    # Iterate through bits in Ns
                    temp_ns = Ns
                    while temp_ns:
                        # Get lowest set bit
                        lsb = temp_ns & -temp_ns
                        u = lsb.bit_length() - 1
                        temp_ns ^= lsb
                        
                        # Check neighbors of u
                        nu = adj[u]
                        # We need a neighbor v of u that is also in Nt, and v != u
                        # Since u is not in Nt? No, u can be in Nt.
                        # If u is in Nt, then u is a valid candidate for v?
                        # Path s-u-v-t. If v=u, path is s-u-u-t (invalid).
                        # So we need v in (nu & Nt) such that v != u.
                        # But if u is in Nt, then u is a neighbor of t.
                        # Is it possible that the only intersection is u itself?
                        # If (nu & Nt) == (1<<u), then the only common neighbor is u itself.
                        # But v must be a neighbor of t. u is a neighbor of t.
                        # If we set v=u, the path is s-u-u-t. Not simple.
                        # So we need another neighbor.
                        
                        common = nu & Nt
                        # Remove u from common if u is in common (i.e., if u is in Nt)
                        # Actually, if u is in Nt, then u is a neighbor of t.
                        # But v must be a neighbor of u.
                        # If v=u, it's a loop.
                        # So we check if common has any bit set other than u.
                        if common:
                            # Check if common has a bit other than u
                            if common != (1 << u):
                                found = True
                                break
                            # If common is exactly (1<<u), then v=u is the only candidate.
                            # But v=u is invalid. So no path via this u?
                            # Wait. If u is in Nt, then u~t.
                            # We need s-u-v-t.
                            # If v=u, invalid.
                            # So if the only neighbor of u in Nt is u itself, then no valid v.
                            # But wait, is u a neighbor of u? No, no loops.
                            # So u cannot be in adj[u].
                            # Therefore, u is NEVER in adj[u].
                            # So (nu & Nt) can NEVER contain bit u.
                            # Because adj[u] does not have bit u set.
                            # So common = nu & Nt will never have bit u set.
                            # Thus, if common != 0, there is a valid v.
                            if common != 0:
                                found = True
                                break
                    if found:
                        return False
        return True

    def find_violation(adj):
        deg = calculate_degrees(adj)
        deg_to_verts = {}
        for i in range(n):
            if deg[i] not in deg_to_verts:
                deg_to_verts[deg[i]] = []
            deg_to_verts[deg[i]].append(i)
        
        for d, verts in deg_to_verts.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    s = verts[i]
                    t = verts[j]
                    Ns = adj[s] & ~(1 << t)
                    Nt = adj[t] & ~(1 << s)
                    
                    temp_ns = Ns
                    while temp_ns:
                        lsb = temp_ns & -temp_ns
                        u = lsb.bit_length() - 1
                        temp_ns ^= lsb
                        
                        nu = adj[u]
                        common = nu & Nt
                        if common:
                            # Find v
                            lsb_v = common & -common
                            v = lsb_v.bit_length() - 1
                            return (s, u, v, t)
        return None

    best_edges = []
    best_count = 0
    
    # Start with complete graph
    full_mask = (1 << n) - 1
    adj = [0] * n
    for i in range(n):
        # Set all bits except i
        adj[i] = full_mask & ~(1 << i)
        
    # Removal Phase
    # We will try to fix violations by removing edges.
    # To maximize edges, we should remove edges that are "least critical".
    # However, removing any edge in the path breaks the specific violation.
    # Removing edges incident to s or t changes degrees, which might create new violations.
    
    # Strategy:
    # 1. Find violation.
    # 2. Identify all edges in the path s-u, u-v, v-t.
    # 3. Also consider edges incident to s and t.
    # 4. Try removing each candidate edge, check validity, and keep the one that allows most future additions?
    #    For speed, just pick one randomly or the one that reduces degree of the most connected vertex?
    
    # Let's use a simple heuristic: Remove an edge from the path.
    # Prefer removing an edge that is not incident to s or t? 
    # Removing (u,v) keeps deg(s), deg(t) same.
    # Removing (s,u) changes deg(s), deg(u).
    
    # We'll do multiple restarts with different random seeds for the removal choice.
    
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Reset to complete graph
        for i in range(n):
            adj[i] = full_mask & ~(1 << i)
            
        steps = 0
        max_steps = 500
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            violation = find_violation(adj)
            if violation is None:
                break
                
            s, u, v, t = violation
            
            # Candidates for removal
            # Edges in path: (s,u), (u,v), (v,t)
            # Also maybe (s,t) if it exists? No, path is s-u-v-t.
            # Removing (s,t) doesn't break this path.
            
            candidates = []
            if adj[s] & (1 << u): candidates.append((s, u))
            if adj[u] & (1 << v): candidates.append((u, v))
            if adj[v] & (1 << t): candidates.append((v, t))
            
            # Also consider removing edges incident to s or t to change their degrees
            # This might be better if deg(s) and deg(t) are high.
            # Let's just pick a random edge from the path to start.
            # If we want to be smarter, we can try all candidates and pick the one that results in a valid graph or least new violations.
            # For now, random choice from path edges.
            
            if not candidates:
                break # Should not happen
            
            # Pick random candidate
            edge_to_remove = random.choice(candidates)
            a, b = edge_to_remove
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            steps += 1
            
        if time.time() > deadline:
            break
            
        if not is_valid(adj):
            continue
            
        # Addition Phase (Local Search)
        # Try to add edges back
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Generate list of missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        missing.append((i, j))
                        
            random.shuffle(missing)
            
            for e in missing:
                if time.time() > deadline:
                    break
                a, b = e
                # Try adding
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
                if is_valid(adj):
                    improved = True
                else:
                    # Revert
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    
        edges = get_edges_from_mask(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
    return best_edges

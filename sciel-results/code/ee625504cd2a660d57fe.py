<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import random
import sys

def _get_degrees(n, edges):
    """Compute degree of each vertex."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg

def _is_valid(n, edges):
    """Check if no 3-edge path has endpoints of equal degree."""
    if not edges:
        return True
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = [0] * n
    for u in range(n):
        deg[u] = len(adj[u])
    
    # Check all 3-edge paths: s-u-v-t where s != t
    # A 3-edge path is s-u-v-t with edges (s,u), (u,v), (v,t)
    # We need to check that deg(s) != deg(t) for all such paths
    for v in range(n):
        for u in adj[v]:
            for t in adj[v]:
                if t == u:
                    continue
                # s is a neighbor of u (s != v to avoid backtracking, but s can be t? No, s != t)
                # Actually, s is any neighbor of u except v (to make it a simple path)
                # But the problem says "simple three-edge path", so s, u, v, t must be distinct
                # Wait, re-reading: "every path s-u-v-t on edge (u, v) with s != t"
                # So s is adjacent to u, t is adjacent to v, s != t, and the path is simple
                # For simplicity, let's check: for each edge (u,v), for each s in adj[u] (s != v), 
                # for each t in adj[v] (t != u), if s != t, then deg[s] != deg[t]
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        # Check if s-u-v-t is a simple path (all vertices distinct)
                        # s != u (since s is neighbor of u and no loops), s != v (we skipped), 
                        # u != v (no loops), s != t (checked), u != t (t != u checked), v != t (t is neighbor of v, no loops)
                        # So we just need deg[s] != deg[t]
                        if deg[s] == deg[t]:
                            return False
    return True

def _count_edges(edges):
    return len(edges)

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start from complete graph and remove edges
    # Strategy 2: Greedy addition
    # Strategy 3: Random restarts with local search
    
    # First, try a greedy approach: add edges in random order, keep if valid
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        current_edges = []
        adj = [set() for _ in range(n)]
        
        for u, v in all_edges:
            # Try adding edge (u, v)
            new_edges = current_edges + [(u, v)]
            if _is_valid(n, new_edges):
                current_edges = new_edges
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
        if best_count >= 100:  # K15 has 105 edges, so 100 is very good
            break
    
    # Local search: try to add back removed edges or swap
    if best_edges and time.time() < deadline - 10:
        current = list(best_edges)
        adj = [set() for _ in range(n)]
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try adding each missing edge
        missing = set(all_edges) - set(current)
        for u, v in list(missing):
            if time.time() > deadline - 5:
                break
            new_edges = current + [(u, v)]
            if _is_valid(n, new_edges):
                current = new_edges
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    
    # Ensure types are correct
    return result

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def _is_valid_fast(n, adj, deg):
    """Fast validity check using adjacency lists and degrees."""
    for v in range(n):
        for u in adj[v]:
            for t in adj[v]:
                if t == u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t2 in adj[v]:
                        if t2 == u:
                            continue
                        if s == t2:
                            continue
                        if deg[s] == deg[t2]:
                            return False
    return True

def _check_add_edge(n, adj, deg, u, v):
    """Check if adding edge (u,v) maintains validity. Only checks new 3-edge paths involving (u,v)."""
    # New 3-edge paths that include edge (u,v):
    # Type 1: s-u-v-t where s in adj[u]\{v}, t in adj[v]\{u}, s!=t
    # Type 2: s-v-u-t where s in adj[v]\{u}, t in adj[u]\{v}, s!=t
    # (These are the same paths just traversed differently, but let's be careful)
    
    # After adding (u,v), deg[u] and deg[v] increase by 1
    new_deg_u = deg[u] + 1
    new_deg_v = deg[v] + 1
    
    # Check Type 1: paths s-u-v-t
    for s in adj[u]:
        if s == v:
            continue
        for t in adj[v]:
            if t == u:
                continue
            if s == t:
                continue
            # s and t are endpoints; their degrees don't change
            if deg[s] == deg[t]:
                return False
    
    # Check Type 2: paths s-v-u-t  
    for s in adj[v]:
        if s == u:
            continue
        for t in adj[u]:
            if t == v:
                continue
            if s == t:
                continue
            if deg[s] == deg[t]:
                return False
    
    # Also need to check: could u or v now be an endpoint of a new violating path?
    # Paths where u is an endpoint: x-?-?-u or u-?-?-x
    # The middle edge could be any edge incident to a neighbor of u's neighbors
    # This is complex; let's do a full check for safety in the main loop
    return True

def _full_check_after_add(n, adj, deg, u, v):
    """Full validity check after tentatively adding edge (u,v)."""
    # Temporarily add edge
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1
    
    # Do full check
    valid = True
    for w in range(n):
        for x in adj[w]:
            for y in adj[w]:
                if y == x:
                    continue
                for z in adj[x]:
                    if z == w:
                        continue
                    if z == y:
                        continue
                    if deg[z] == deg[y]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    # Remove edge
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1
    
    return valid

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    edge_set = set(all_edges)
    best_edges = []
    best_count = 0
    
    # Strategy: Start with a dense base graph, then local search
    # Key insight: use degree-based partitioning to guide edge selection
    
    # Try multiple random orderings with greedy + local search
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        current_edges = []
        adj = [set() for _ in range(n)]
        deg = [0] * n
        
        for u, v in all_edges:
            # Quick check: only add if it doesn't immediately create a violation
            # involving the new edge as middle
            # Check paths s-u-v-t and s-v-u-t
            ok = True
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                for s in adj[v]:
                    if s == u:
                        continue
                    for t in adj[u]:
                        if t == v:
                            continue
                        if s == t:
                            continue
                        if deg[s] == deg[t]:
                            ok = False
                            break
                    if not ok:
                        break
            
            if ok:
                current_edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        
        # Local search: try adding missing edges
        if time.time() < deadline - 5:
            missing = [e for e in all_edges if e not in set(current_edges)]
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                random.shuffle(missing)
                for u, v in missing:
                    if (u, v) in set(current_edges) or (v, u) in set(current_edges):
                        continue
                    # Try adding
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    
                    # Full check
                    valid = True
                    for w in range(n):
                        for x in adj[w]:
                            for y in adj[w]:
                                if y == x:
                                    continue
                                for z in adj[x]:
                                    if z == w:
                                        continue
                                    if z == y:
                                        continue
                                    if deg[z] == deg[y]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        current_edges.append((u, v))
                        improved = True
                        missing.remove((u, v))
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        deg[u] -= 1
                        deg[v] -= 1
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
        if best_count >= 100:
            break
    
    # Final polish: try to add any remaining missing edges
    if best_edges and time.time() < deadline - 3:
        current = list(best_edges)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        missing = [e for e in all_edges if e not in set(current)]
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            valid = True
            for w in range(n):
                for x in adj[w]:
                    for y in adj[w]:
                        if y == x:
                            continue
                        for z in adj[x]:
                            if z == w:
                                continue
                            if z == y:
                                continue
                            if deg[z] == deg[y]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                current.append((u, v))
            else:
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current
    
    # Convert to list of lists with proper types
    result = [[int(u), int(v)] for u, v in best_edges]
    
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

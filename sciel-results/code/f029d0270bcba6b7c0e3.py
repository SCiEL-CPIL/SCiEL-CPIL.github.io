import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
N = 15

def is_valid(adj, deg):
    """Check if the graph with adjacency sets and degrees is valid.
    A path s-u-v-t (three edges) is invalid if deg[s] == deg[t] and s != t."""
    for u in range(N):
        for v in adj[u]:
            if u >= v:
                continue
            # For edge (u,v), check all s in adj[u]\{v} and t in adj[v]\{u}
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s == t:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def check_add(adj, deg, u, v):
    """Check if adding edge (u,v) keeps the graph valid.
    We need to check:
    1. Existing paths that become endpoints: any path s-u-v-t where s was adjacent to u (not v)
       and t was adjacent to v (not u) - but wait, s-u-v-t means edges s-u, u-v, v-t.
       So s must be in adj[u] (s != v), t must be in adj[v] (t != u), and s != t.
       After adding u-v, deg[u] and deg[v] increase by 1.
       But also, new paths of the form s-v-u-t or t-u-v-s could form.
       
       Actually, let me think more carefully. When we add edge (u,v):
       - New paths of length 3 that use this new edge:
         * s-u-v-t where s in adj[u]\{v}, t in adj[v]\{u}, s != t
         * s-v-u-t where s in adj[v]\{u}, t in adj[u]\{v}, s != t
       
       For s-u-v-t: need deg[s] != deg[t]
       For s-v-u-t: need deg[s] != deg[t]
       
       But deg values are from BEFORE adding the edge (since we're checking if we CAN add it).
       After adding, deg[u] and deg[v] increase by 1, but s and t are not u or v, so their
       degrees don't change. So we just check deg[s] != deg[t] using current degrees.
       
       Also, we need to check that adding the edge doesn't create conflicts with existing
       paths. But existing paths don't change their endpoint degrees (only u and v's degrees
       change, and they were already endpoints of some paths). Wait - actually, u and v ARE
       endpoints of existing paths. When their degrees change, those paths might become
       invalid or valid. But we're only checking if the NEW edge creates violations.
       
       Hmm, actually the check is: after adding edge (u,v), is the whole graph valid?
       The new edge creates new 3-edge paths. But it also changes degrees of u and v,
       which affects whether EXISTING 3-edge paths that have u or v as endpoints are valid.
       
       Let me reconsider. An existing path s-a-b-t has endpoints s and t. If u or v is s or t,
       then the degree of an endpoint changed. So we need to check:
       - All 3-edge paths that have u as an endpoint (s=u or t=u)
       - All 3-edge paths that have v as an endpoint (s=v or t=v)
       
       For u as endpoint: paths u-a-b-t where a in adj[u], b in adj[a]\{u}, t in adj[b]\{a}, t != u
       After adding edge, deg[u] increases by 1. So we need deg[u]+1 != deg[t] for all such paths.
       But this is expensive to check.
       
       Alternative: just do the full check after tentatively adding the edge.
    """
    # Tentatively add
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1
    
    valid = is_valid(adj, deg)
    
    # Remove
    adj[u].remove(v)
    adj[v].remove(u)
    deg[u] -= 1
    deg[v] -= 1
    
    return valid

def fast_is_valid(adj, deg):
    """Faster validity check."""
    for u in range(N):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u:
                        continue
                    if t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def construct():
    deadline = time.time() + 85  # Leave 5s margin for 90s limit
    
    best_edges = 14  # Star graph
    best_adj = [set() for _ in range(N)]
    for v in range(1, N):
        best_adj[0].add(v)
        best_adj[v].add(0)
    best_deg = [14] + [1] * 14
    
    random.seed(42)
    
    # Try local search with random restarts
    while time.time() < deadline - 10:
        # Start from a random valid graph or the best found
        if random.random() < 0.3:
            # Start from best
            adj = [set(s) for s in best_adj]
            deg = best_deg[:]
        else:
            # Random start: sparse graph
            adj = [set() for _ in range(N)]
            deg = [0] * N
            # Add random edges until valid or can't add more
            pairs = list(combinations(range(N), 2))
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline - 10:
                    break
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                if not fast_is_valid(adj, deg):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
            
            current_edges = sum(len(adj[i]) for i in range(N)) // 2
            if current_edges >= best_edges:
                best_edges = current_edges
                best_adj = [set(s) for s in adj]
                best_deg = deg[:]
        
        # Local search: try to add edges
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            pairs = list(combinations(range(N), 2))
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline - 10:
                    break
                if v in adj[u]:
                    continue
                if fast_is_valid(adj, deg) is False:
                    break
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                if fast_is_valid(adj, deg):
                    improved = True
                    current_edges = sum(len(adj[i]) for i in range(N)) // 2
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [set(s) for s in adj]
                        best_deg = deg[:]
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
        
        # Try from best again
        if time.time() < deadline - 10:
            adj = [set(s) for s in best_adj]
            deg = best_deg[:]
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                pairs = list(combinations(range(N), 2))
                random.shuffle(pairs)
                for u, v in pairs:
                    if time.time() > deadline - 10:
                        break
                    if v in adj[u]:
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    if fast_is_valid(adj, deg):
                        improved = True
                        current_edges = sum(len(adj[i]) for i in range(N)) // 2
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [set(s) for s in adj]
                            best_deg = deg[:]
                    else:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        deg[u] -= 1
                        deg[v] -= 1
    
    # Return best result
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges
# EVOLVE-BLOCK-END

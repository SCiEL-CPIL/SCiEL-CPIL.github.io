"""K2,3-free graph on 20 vertices: maximize edges via spectral init + optimized local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy 1: Spectral initialization from random regular graphs
    for seed in range(200):
        if time.time() > deadline - 40.0:
            break
        
        random.seed(seed)
        edges = _spectral_init(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 2: Optimized local search from best result
    if time.time() < deadline - 25.0 and best_edges:
        improved = _fast_local_search(n, best_edges, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved
    
    # Strategy 3: More spectral inits with local search on promising results
    for seed in range(200, 800):
        if time.time() > deadline - 10.0:
            break
        
        edges = _spectral_init(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
        
        # Quick local search on promising results
        if score >= best_score - 3 and time.time() < deadline - 5.0:
            improved = _fast_local_search(n, edges, deadline)
            if len(improved) > best_score:
                best_score = len(improved)
                best_edges = improved[:]
    
    return best_edges


def _spectral_init(n, seed):
    """
    Initialize from a random regular graph with high girth, then greedily add edges.
    Random regular graphs tend to have fewer short cycles, which helps K2,3-freeness.
    """
    random.seed(seed)
    
    # Start with a random 8-regular graph (balanced degree)
    # Use configuration model or simple random approach
    adj = [set() for _ in range(n)]
    edges = []
    
    # Build a random regular graph of degree ~8-10
    target_degree = 9
    
    # Simple approach: randomly add edges until degrees are balanced
    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_possible)
    
    for u, v in all_possible:
        if len(adj[u]) < target_degree and len(adj[v]) < target_degree:
            # Check K2,3 constraint
            if _can_add_fast(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Now greedily add more edges
    remaining = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
    random.shuffle(remaining)
    
    for u, v in remaining:
        if _can_add_fast(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges


def _can_add_fast(adj, u, v):
    """
    Fast check if adding edge (u,v) keeps graph K2,3-free.
    Precompute common neighbor counts where possible.
    """
    # After adding (u,v), check pairs involving u or v
    
    # Pair (u,v): common neighbors
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    # For each w in adj[v] \ {u}: pair (u,w)
    # New common neighbors = (adj[u] ∪ {v}) ∩ adj[w] = (adj[u] ∩ adj[w]) ∪ ({v} if v in adj[w])
    for w in adj[v]:
        if w == u:
            continue
        # v is in adj[w] since w in adj[v]
        common = len(adj[u] & adj[w]) + 1  # +1 for v
        if common >= 3:
            return False
    
    # For each w in adj[u] \ {v}: pair (v,w)
    for w in adj[u]:
        if w == v:
            continue
        # u is in adj[w] since w in adj[u]
        common = len(adj[v] & adj[w]) + 1  # +1 for u
        if common >= 3:
            return False
    
    return True


def _fast_local_search(n, edges, deadline):
    """
    Fast local search with efficient state management.
    Uses set-based adjacency for O(1) membership checks.
    """
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    current_edges = [e[:] for e in edges]
    current_score = len(current_edges)
    
    # Phase 1: Greedy fill - add all possible missing edges
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    random.shuffle(missing)
    
    added_any = True
    while added_any and time.time() < deadline - 5.0:
        added_any = False
        for u, v in missing:
            if (u, v) in edge_set:
                continue
            if _can_add_fast(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                current_edges.append([u, v])
                current_score += 1
                added_any = True
    
    # Phase 2: Edge swap moves
    for iteration in range(400):
        if time.time() > deadline - 3.0:
            break
        
        if not current_edges:
            break
        
        idx = random.randint(0, len(current_edges) - 1)
        u_rem, v_rem = current_edges[idx]
        rem_key = (min(u_rem, v_rem), max(u_rem, v_rem))
        
        # Save state
        old_score = current_score
        
        # Remove edge
        adj[u_rem].discard(v_rem)
        adj[v_rem].discard(u_rem)
        edge_set.discard(rem_key)
        current_edges.pop(idx)
        current_score -= 1
        
        # Try to add all possible missing edges
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set]
        random.shuffle(missing2)
        
        added_count = 0
        for a, b in missing2:
            if _can_add_fast(adj, a, b):
                adj[a].add(b)
                adj[b].add(a)
                edge_set.add((a, b))
                current_edges.append([a, b])
                current_score += 1
                added_count += 1
        
        # If we didn't gain edges, revert
        if current_score <= old_score:
            # Remove any edges we added
            while added_count > 0:
                extra = current_edges.pop()
                eu, ev = extra
                adj[eu].discard(ev)
                adj[ev].discard(eu)
                edge_set.discard((min(eu, ev), max(eu, ev)))
                current_score -= 1
                added_count -= 1
            
            # Restore removed edge
            adj[u_rem].add(v_rem)
            adj[v_rem].add(u_rem)
            edge_set.add(rem_key)
            current_edges.append([u_rem, v_rem])
            current_score += 1
    
    # Phase 3: 2-opt moves
    for iteration in range(200):
        if time.time() > deadline - 3.0:
            break
        
        if len(current_edges) < 2:
            break
        
        idx1 = random.randint(0, len(current_edges) - 1)
        idx2 = random.randint(0, len(current_edges) - 1)
        if idx1 == idx2:
            continue
        
        u1, v1 = current_edges[idx1]
        u2, v2 = current_edges[idx2]
        key1 = (min(u1, v1), max(u1, v1))
        key2 = (min(u2, v2), max(u2, v2))
        
        # Save state
        old_score = current_score
        
        # Remove both edges
        adj[u1].discard(v1)
        adj[v1].discard(u1)
        edge_set.discard(key1)
        adj[u2].discard(v2)
        adj[v2].discard(u2)
        edge_set.discard(key2)
        
        # Pop in correct order
        if idx1 < idx2:
            current_edges.pop(idx2)
            current_edges.pop(idx1)
        else:
            current_edges.pop(idx1)
            current_edges.pop(idx2)
        
        current_score -= 2
        
        # Try to add all possible missing edges
        missing3 = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set]
        random.shuffle(missing3)
        
        added_count = 0
        for a, b in missing3:
            if _can_add_fast(adj, a, b):
                adj[a].add(b)
                adj[b].add(a)
                edge_set.add((a, b))
                current_edges.append([a, b])
                current_score += 1
                added_count += 1
        
        # If we didn't gain edges, revert
        if current_score <= old_score:
            # Remove any edges we added
            while added_count > 0:
                extra = current_edges.pop()
                eu, ev = extra
                adj[eu].discard(ev)
                adj[ev].discard(eu)
                edge_set.discard((min(eu, ev), max(eu, ev)))
                current_score -= 1
                added_count -= 1
            
            # Restore removed edges
            adj[u1].add(v1)
            adj[v1].add(u1)
            edge_set.add(key1)
            current_edges.append([u1, v1])
            current_score += 1
            
            adj[u2].add(v2)
            adj[v2].add(u2)
            edge_set.add(key2)
            current_edges.append([u2, v2])
            current_score += 1
    
    # Return as list of lists
    return [[u, v] for u, v in current_edges]


# EVOLVE-BLOCK-END

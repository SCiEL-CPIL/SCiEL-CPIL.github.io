import time
import sys
from random import Random

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(n, edges, time_budget, global_deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(edges)
    pieces = []
    rng = Random(42)
    t0 = time.time()
    
    # Helper to find a cycle in the current graph
    def find_cycle():
        # If no edges, no cycle
        if not remaining:
            return None
        
        # Get vertices with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return None
            
        best_cycle = None
        best_len = 0
        
        # Try multiple times to find a long cycle
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > global_deadline or time.time() - t0 > time_budget:
                break
                
            start = rng.choice(candidates)
            path = [start]
            visited = {start}
            cur = start
            
            # Greedy extension
            while True:
                # Find unvisited neighbors
                nbrs = [nb for nb in adj[cur] if nb not in visited]
                
                # If we can close the cycle and it's long enough
                if start in adj[cur] and len(path) >= 3:
                    # We found a cycle. Check if it's better than best
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    # We can stop this attempt or try to extend more?
                    # Usually, closing early is suboptimal. 
                    # Let's try to extend more if possible, but if we can't, we keep this.
                    # Actually, if we close now, we commit to this cycle.
                    # To find a *longer* cycle, we should avoid closing if we have other options.
                    # So, if we have other unvisited neighbors, we should probably take one.
                    # If nbrs is empty, we MUST close if possible, or fail.
                    if not nbrs:
                        if len(path) >= 3 and start in adj[cur]:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break
                    else:
                        # We have other neighbors. Should we close or continue?
                        # Heuristic: If the cycle is already quite long, maybe close.
                        # Otherwise, continue to try for a longer one.
                        # Let's say if len(path) > 10, close. Else continue.
                        if len(path) > 10:
                             if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                             break
                        else:
                            # Do not close, continue extending
                            pass
                            
                if not nbrs:
                    # Stuck, cannot extend
                    if len(path) >= 3 and start in adj[cur]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                
                # Select next neighbor
                # Heuristic: Pick neighbor with lowest degree to avoid traps
                # But add some randomness
                nbrs.sort(key=lambda x: len(adj[x]))
                if rng.random() < 0.3:
                    nxt = rng.choice(nbrs)
                else:
                    nxt = nbrs[0]
                
                path.append(nxt)
                visited.add(nxt)
                cur = nxt
                
        return best_cycle

    while remaining:
        if time.time() > global_deadline or time.time() - t0 > time_budget:
            break
            
        cycle = find_cycle()
        if cycle is None:
            break
            
        # Verify cycle edges are in remaining
        ok = True
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = (a, b) if a < b else (b, a)
            if e not in remaining:
                ok = False
                break
        
        if not ok:
            # This shouldn't happen if logic is correct, but just in case
            break
            
        # Commit cycle
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
            
        pieces.append(cycle)
        
    # Leftover edges
    for e in sorted(remaining):
        pieces.append(list(e))
        
    return pieces

def _validate(n, edges, pieces):
    seen = set()
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (u, v) if u < v else (v, u)
            if e in seen:
                return False
            seen.add(e)
        elif len(piece) >= 3:
            verts = piece
            if len(set(verts)) != len(verts):
                return False
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                e = (a, b) if a < b else (b, a)
                if e in seen:
                    return False
                seen.add(e)
        else:
            return False
    if seen != set(edges):
        return False
    return True

def construct():
    global_deadline = time.time() + 145.0 # Safety margin
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Allocate time roughly equally, but with some buffer
    # 150s limit. Let's give 45s per graph.
    time_per_graph = 45.0
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Adjust budget based on remaining time
        elapsed = time.time() - (global_deadline - 145.0)
        remaining_time = 145.0 - elapsed - 5.0 # 5s safety
        if i < 2:
            budget = min(time_per_graph, remaining_time)
        else:
            budget = remaining_time
            
        budget = max(1.0, budget)
        
        try:
            pieces = _decompose(n, edges, budget, global_deadline)
            if not _validate(n, edges, pieces):
                pieces = [list(e) for e in edges]
        except Exception:
            pieces = [list(e) for e in edges]
            
        results.append(pieces)
        
    return results

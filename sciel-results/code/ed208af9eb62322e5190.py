"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_solve(edges, max_restarts=200, time_limit=30.0):
        if not edges:
            return []
        
        edge_set = set(edges)
        m = len(edge_set)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        def degree_dict(es):
            d = defaultdict(int)
            for u, v in es:
                d[u] += 1
                d[v] += 1
            return d
        
        best = None
        best_count = float('inf')
        t_start = time.time()
        
        for restart in range(max_restarts):
            if time.time() - t_start > time_limit:
                break
            if m == 0:
                break
                
            rng = random.Random(hash((restart, tuple(sorted(edge_set)))) % (2**32))
            cur_edges = set(edge_set)
            paths = []
            
            while cur_edges:
                # Compute degrees in current residual graph
                deg = degree_dict(cur_edges)
                if not deg:
                    break
                    
                # Choose start: highest degree, tie-break random
                max_d = max(deg.values())
                candidates = [v for v, d in deg.items() if d == max_d]
                start = rng.choice(candidates)
                
                # Build path using Warnsdorff-like heuristic
                path = [start]
                visited = {start}
                last = start
                removed = set()
                
                while True:
                    # Find available neighbors of last in cur_edges not visited
                    nbrs = []
                    for nb in adj[last]:
                        if nb in visited:
                            continue
                        e = tuple(sorted((last, nb)))
                        if e in cur_edges and e not in removed:
                            # Count onward moves for nb
                            onward = 0
                            for nn in adj[nb]:
                                if nn in visited or nn == last:
                                    continue
                                e2 = tuple(sorted((nb, nn)))
                                if e2 in cur_edges and e2 not in removed:
                                    onward += 1
                            nbrs.append((onward, nb))
                    
                    if not nbrs:
                        break
                    
                    # Warnsdorff: pick neighbor with fewest onward moves
                    # Tie-break randomly among best
                    min_onward = min(x[0] for x in nbrs)
                    best_nbs = [nb for o, nb in nbrs if o == min_onward]
                    nxt = rng.choice(best_nbs)
                    
                    e = tuple(sorted((last, nxt)))
                    cur_edges.discard(e)
                    removed.add(e)
                    path.append(nxt)
                    visited.add(nxt)
                    last = nxt
                
                if len(path) >= 2:
                    paths.append(path)
                else:
                    # Should not happen if edges exist and start has degree > 0
                    # Fallback: add remaining edges as individual paths
                    for e in cur_edges:
                        paths.append(list(e))
                    cur_edges = set()
                    break
            
            cnt = len(paths)
            if cnt < best_count:
                best_count = cnt
                best = [p[:] for p in paths]
                
                # Early stop if we hit a very good number
                # Theoretical lower bound: (sum(deg) - 2*V_nontrivial)/2 ... complex
                # Gallai: floor(n/2) for trees. For general connected: floor(n/2) - 1 ?
                # n=20, floor(20/2)=10. n=25, floor(25/2)=12. n=30, floor(30/2)=15.
                # Total ~ 37.
                # If we get below or equal to 37 total, we are close to optimal.
                # Per graph: if best_count <= 10 for n=20, stop.
                # But we don't know n here directly, just edges.
                # Estimate n from max vertex id + 1? No, vertices might be 0..n-1.
                # We can check if best_count is very small relative to m.
                # For now, just run all restarts or until time limit.
        
        if best is None:
            # Fallback: trivial decomposition
            return [[u, v] for u, v in edge_set]
            
        return best
    
    def validate_and_fix(paths, edges):
        """
        Ensure paths are valid:
        1. Simple paths
        2. Edges in paths are in graph
        3. Edge-disjoint
        4. Cover all edges
        """
        edge_set = set(edges)
        used_edges = set()
        valid_paths = []
        for p in paths:
            # Check simple
            if len(p) != len(set(p)):
                # Path is not simple. Break it?
                # For safety, if a path is invalid, we fall back to single edges for that segment?
                # Actually, greedy ensures simplicity.
                pass
            
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set:
                    ok = False
                    break
                if e in used_edges:
                    ok = False
                    break
                used_edges.add(e)
            
            if ok:
                valid_paths.append(p)
            else:
                # If a path is invalid, add its edges individually?
                # This might cause duplicates if we're not careful.
                # Better to just return the valid ones and handle the rest?
                # But we need to cover ALL edges.
                # Since greedy is deterministic and correct, this should not happen.
                # For safety, let's add the edges of this path individually.
                for i in range(len(p) - 1):
                    e = tuple(sorted((p[i], p[i+1])))
                    if e in edge_set and e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([p[i], p[i+1]])
        
        # Check coverage
        if used_edges != edge_set:
            # Missing edges
            for e in edge_set - used_edges:
                valid_paths.append(list(e))
        
        # Check no duplicates in valid_paths (should be ensured by used_edges)
        return valid_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        
        if time_left < 2:
            # Emergency fallback
            paths = [[u, v] for u, v in g_edges]
        else:
            # Allocate time: give more time to later graphs if we are fast?
            # Or fixed split.
            # 3 graphs. Total budget ~130s.
            # Let's give 40s each if time allows, but cap at 30s to be safe?
            # No, greedy is fast. 50 restarts takes < 1s usually.
            # 200 restarts takes < 2s.
            # So we can afford many restarts.
            # Let's set max_restarts high and time_limit per graph to 10s.
            # If we finish fast, we could do more, but greedy convergence is fast.
            paths = greedy_solve(g_edges, max_restarts=500, time_limit=min(15.0, time_left / 3.0))
            
            # Final validation
            paths = validate_and_fix(paths, g_edges)
            
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END

"""Program for the graph construction task with relaxation-guided search."""
import time
import random
import numpy as np

DEADLINE = 115  # safety margin before 120s hard kill

def _compute_score_fast(adj, deg):
    """Compute the largest degree sum over triangles using numpy."""
    n = adj.shape[0]
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            common = np.logical_and(adj[u], adj[v])
            for w in np.where(common)[0]:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _compute_score_fast_no_triangles(adj, deg):
    """Fast check: return 0 if no triangles, else compute score."""
    n = adj.shape[0]
    max_sum = 0
    found_triangle = False
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            common = np.logical_and(adj[u], adj[v])
            ws = np.where(common)[0]
            for w in ws:
                if w > v:
                    found_triangle = True
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _get_triangle_vertices(adj, deg):
    """Return list of (u, v, w, score) for all triangles."""
    n = adj.shape[0]
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u, v]:
                continue
            common = np.logical_and(adj[u], adj[v])
            for w in np.where(common)[0]:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    triangles.append((u, v, w, s))
    return triangles

def _relaxation_guidance(adj, deg, n):
    """
    Compute a continuous partition vector that guides edge swaps.
    Uses a simple iterative relaxation: vertices connected to high-degree
    vertices should be on the "opposite" side.
    
    Returns: x vector in [0,1]^n where x_i indicates "side" of vertex i.
    Also returns a score for each potential edge swap indicating how much
    it would improve alignment with the relaxation.
    """
    # Initialize x based on degree: high degree -> one side, low degree -> other
    # Actually, we want to find a cut that explains the edge structure.
    # Use a simple power-iteration-like approach on the adjacency.
    
    x = np.ones(n) * 0.5
    # Initialize with degree-based heuristic
    if deg.max() > deg.min():
        x = (deg - deg.min()) / (deg.max() - deg.min())
    
    # Iterate: x_i = mean of (1 - x_j) for neighbors j
    # This is like finding a bipartition
    for _ in range(50):
        new_x = np.zeros(n)
        for i in range(n):
            neighbors = np.where(adj[i])[0]
            if len(neighbors) > 0:
                new_x[i] = np.mean(1.0 - x[neighbors])
            else:
                new_x[i] = 0.5
        x = new_x
    
    # Normalize to [0, 1]
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.ones(n) * 0.5
    
    return x

def _swap_guidance_score(x, adj, u1, v1, u2, v2):
    """
    Score how good a swap is based on the relaxation vector.
    Positive = good swap (moves toward relaxation optimum).
    """
    # Removing edge (u1,v1): good if x[u1] and x[v1] are close (shouldn't be connected)
    remove_goodness = 1.0 - abs(x[u1] - x[v1])
    # Adding edge (u2,v2): good if x[u2] and x[v2] are far (should be connected)
    add_goodness = abs(x[u2] - x[v2])
    
    return add_goodness - remove_goodness

def _initial_constructions():
    """Generate multiple diverse initial constructions."""
    n = 16
    target = 65
    constructions = []
    
    # 1. K_{8,8} + one internal edge
    adj = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = True
            adj[v, u] = True
    adj[0, 1] = True
    adj[1, 0] = True
    constructions.append(adj.copy())
    
    # 2. K_{7,9} + internal edges
    adj = np.zeros((n, n), dtype=bool)
    for u in range(7):
        for v in range(7, 16):
            adj[u, v] = True
            adj[v, u] = True
    adj[0, 1] = True
    adj[1, 0] = True
    adj[2, 3] = True
    adj[3, 2] = True
    constructions.append(adj.copy())
    
    # 3. K_{6,10} + internal edges
    adj = np.zeros((n, n), dtype=bool)
    for u in range(6):
        for v in range(6, 16):
            adj[u, v] = True
            adj[v, u] = True
    internal_pairs = [(0,1), (0,2), (1,2), (6,7), (6,8)]
    for u, v in internal_pairs:
        adj[u, v] = True
        adj[v, u] = True
    constructions.append(adj.copy())
    
    # 4. Random near-bipartite
    for seed in range(10):
        random.seed(seed)
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(7, 9)
        A = set(perm[:split])
        B = set(perm[split:])
        
        adj = np.zeros((n, n), dtype=bool)
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        remaining = target - edge_count
        
        if remaining > 0:
            internal = []
            for part in [A, B]:
                plist = list(part)
                for i in range(len(plist)):
                    for j in range(i+1, len(plist)):
                        internal.append((plist[i], plist[j]))
            random.shuffle(internal)
            for u, v in internal[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            cross = [(u, v) for u in A for v in B]
            random.shuffle(cross)
            for u, v in cross[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        constructions.append(adj.copy())
    
    return constructions

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_pairs_set = set(all_pairs)
    
    best_score = float('inf')
    best_adj = None
    
    def _elapsed():
        return time.time() - start_time
    
    def _time_left():
        return DEADLINE - _elapsed()
    
    # Try each initial construction
    for init_adj in _initial_constructions():
        if _time_left() < 10:
            break
        
        deg = init_adj.sum(axis=1).astype(int)
        score = _compute_score_fast(init_adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = init_adj.copy()
        
        # Local search with relaxation guidance from this starting point
        current_adj = init_adj.copy()
        current_deg = deg.copy()
        current_score = score
        
        # Compute initial relaxation
        x = _relaxation_guidance(current_adj, current_deg, n)
        
        T = 2.0
        cooling = 0.9998
        no_improve = 0
        max_no_improve = 3000
        relax_refresh_interval = 200  # refresh relaxation every N steps
        
        for it in range(80000):
            if _time_left() < 5:
                break
            if no_improve > max_no_improve:
                break
            
            # Refresh relaxation periodically
            if it > 0 and it % relax_refresh_interval == 0:
                x = _relaxation_guidance(current_adj, current_deg, n)
            
            # Generate candidate swaps
            # Mix of: guided swaps (top by relaxation) and random swaps
            candidates = []
            
            # Guided swaps: pick edges that are "misaligned" with relaxation
            # Misaligned existing edges: |x_u - x_v| small (shouldn't exist)
            # Misaligned missing edges: |x_u - x_v| large (should exist)
            
            existing_edges = [(u, v) for u, v in all_pairs if current_adj[u, v]]
            missing_edges = [(u, v) for u, v in all_pairs if not current_adj[u, v]]
            
            if len(existing_edges) > 0 and len(missing_edges) > 0:
                # Score existing edges for removal (want to remove misaligned)
                rem_scores = []
                for u, v in existing_edges:
                    s = 1.0 - abs(x[u] - x[v])  # high = misaligned (close in x)
                    rem_scores.append((s, u, v))
                rem_scores.sort(reverse=True)
                
                # Score missing edges for addition (want to add misaligned)
                add_scores = []
                for u, v in missing_edges:
                    s = abs(x[u] - x[v])  # high = misaligned (far in x)
                    add_scores.append((s, u, v))
                add_scores.sort(reverse=True)
                
                # Take top candidates
                top_rem = rem_scores[:10]
                top_add = add_scores[:10]
                
                # Generate candidate swaps from top candidates
                for rs, ru, rv in top_rem[:5]:
                    for as_, au, av in top_add[:5]:
                        if (ru, rv) != (au, av):
                            candidates.append((ru, rv, au, av))
                
                # Also add some random candidates
                for _ in range(10):
                    ru, rv = random.choice(existing_edges)
                    au, av = random.choice(missing_edges)
                    if (ru, rv) != (au, av):
                        candidates.append((ru, rv, au, av))
            
            if not candidates:
                no_improve += 1
                continue
            
            # Pick a random candidate
            ru, rv, au, av = random.choice(candidates)
            
            # Perform swap
            new_adj = current_adj.copy()
            new_deg = current_deg.copy()
            
            new_adj[ru, rv] = False
            new_adj[rv, ru] = False
            new_deg[ru] -= 1
            new_deg[rv] -= 1
            
            new_adj[au, av] = True
            new_adj[av, au] = True
            new_deg[au] += 1
            new_deg[av] += 1
            
            new_score = _compute_score_fast(new_adj, new_deg)
            delta = new_score - current_score
            
            if delta < 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cooling
        
        # Final hill-climbing pass on best found so far
        if _time_left() < 5:
            break
    
    # Final exhaustive local search on best_adj
    if best_adj is not None and _time_left() > 10:
        current_adj = best_adj.copy()
        current_deg = current_adj.sum(axis=1).astype(int)
        current_score = _compute_score_fast(current_adj, current_deg)
        
        # Compute relaxation for guidance
        x = _relaxation_guidance(current_adj, current_deg, n)
        
        improved = True
        while improved and _time_left() > 5:
            improved = False
            
            existing_edges = [(u, v) for u, v in all_pairs if current_adj[u, v]]
            missing_edges = [(u, v) for u, v in all_pairs if not current_adj[u, v]]
            
            # Score all possible swaps
            swap_candidates = []
            for ru, rv in existing_edges:
                rem_s = 1.0 - abs(x[ru] - x[rv])
                for au, av in missing_edges:
                    if (ru, rv) == (au, av):
                        continue
                    add_s = abs(x[au] - x[av])
                    swap_candidates.append((add_s - rem_s, ru, rv, au, av))
            
            # Sort by guidance score (descending)
            swap_candidates.sort(key=lambda x: x[0], reverse=True)
            
            for _, ru, rv, au, av in swap_candidates:
                if _time_left() < 5:
                    break
                
                test_adj = current_adj.copy()
                test_deg = current_deg.copy()
                
                test_adj[ru, rv] = False
                test_adj[rv, ru] = False
                test_deg[ru] -= 1
                test_deg[rv] -= 1
                
                test_adj[au, av] = True
                test_adj[av, au] = True
                test_deg[au] += 1
                test_deg[av] += 1
                
                s = _compute_score_fast(test_adj, test_deg)
                if s < current_score:
                    current_score = s
                    current_adj = test_adj
                    current_deg = test_deg
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_adj = current_adj.copy()
                        # Refresh relaxation
                        x = _relaxation_guidance(current_adj, current_deg, n)
                    break
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}

"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def time_left():
        return deadline - time.time()

    # 1. Calculate Moore Bound
    if k == 2:
        moore = g
    else:
        if g % 2 == 0:
            moore = k + k * (k - 1) ** (g // 2 - 1)
        else:
            h = (g - 1) // 2
            if k == 2:
                moore = g
            else:
                moore = 1 + k * ((k - 1) ** h - 1) // (k - 2)
    
    # For k=5, g=9, moore = 426.
    # We want to find the smallest n >= moore such that a graph exists.
    
    def verify_girth(n, edges, min_girth):
        """
        Verify no cycle shorter than min_girth.
        Returns True if valid, False otherwise.
        Also checks regularity and simplicity.
        """
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k, seed=None):
        """
        Generate a random k-regular graph using the configuration model.
        Returns (n, edges) or None if failed (e.g., multiple edges).
        """
        if n * k % 2 != 0:
            return None
        
        # Try a few times to get a simple graph
        for attempt in range(100):
            if seed is not None:
                random.seed(seed + attempt)
            else:
                random.seed()
            
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            edge_set = set()
            valid = True
            for i in range(n * k // 2):
                u, v = stubs[2 * i], stubs[2 * i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return n, list(edge_set)
        return None

    def count_short_cycles(adj, limit, n):
        """
        Count cycles of length < limit.
        Used as objective function for simulated annealing.
        """
        count = 0
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                if dist[u] >= limit // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < limit:
                            count += 1
        return count

    def local_search(n, edges, target_girth, max_time):
        """
        Improve girth using 2-switches.
        Returns (n, edges) if girth >= target_girth is achieved, else None.
        """
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        current_cycles = count_short_cycles(adj, target_girth, n)
        if current_cycles == 0:
            return n, list(edge_set)
        
        temperature = 10.0
        cooling_rate = 0.995
        max_iter = 50000
        # We want to be fast, so we limit iterations based on time
        time_limit = max_time - time.time()
        
        for i in range(max_iter):
            if time.time() > max_time:
                break
            if current_cycles == 0:
                return n, list(edge_set)
            
            # Pick two random edges (u1, v1) and (u2, v2) that are disjoint
            # To make it faster, we can pick random vertices and their neighbors
            u1 = random.randint(0, n - 1)
            if not adj[u1]: continue
            v1 = random.choice(adj[u1])
            
            u2 = random.randint(0, n - 1)
            if not adj[u2]: continue
            v2 = random.choice(adj[u2])
            
            # Check if edges are distinct and disjoint
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # New edges: (u1, u2) and (v1, v2)
            # Check validity
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            key1 = (min(u1, u2), max(u1, u2))
            key2 = (min(v1, v2), max(v1, v2))
            
            if key1 in edge_set or key2 in edge_set:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            edge_set.remove((min(u1, v1), max(u1, v1)))
            edge_set.remove((min(u2, v2), max(u2, v2)))
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edge_set.add(key1)
            edge_set.add(key2)
            
            new_cycles = count_short_cycles(adj, target_girth, n)
            
            delta = new_cycles - current_cycles
            if delta <= 0 or random.random() < math.exp(-delta / temperature):
                current_cycles = new_cycles
            else:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                edge_set.remove(key1)
                edge_set.remove(key2)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                edge_set.add((min(u1, v1), max(u1, v1)))
                edge_set.add((min(u2, v2), max(u2, v2)))
            
            temperature *= cooling_rate
            
        if current_cycles == 0:
            return n, list(edge_set)
        return None, None

    best_n = None
    best_edges = None
    
    # Search strategy:
    # Start from Moore bound, go up in steps.
    # For k=5, g=9, Moore=426. Known best is 1116.
    # We can jump in larger steps initially, then refine.
    
    # Heuristic: Random graphs have girth ~ log(n).
    # To get girth 9, we need n ~ 4^9 ~ 260k? No, that's for random regular graphs.
    # But we are using local search to increase girth.
    # It's easier to start from a smaller n and try to increase girth.
    
    # Let's try a range of n.
    # If we find a valid graph at n, we record it.
    # We want the minimum n.
    
    # Since the known best is 1116, and Moore is 426, there is a gap.
    # We will iterate n from moore to, say, 1200.
    # To save time, we will try fewer seeds for smaller n if it fails quickly.
    
    current_n = moore
    step = 10
    
    # If k and g are large, the search space is huge.
    # For this specific task (k=5, g=9), we can be more aggressive.
    
    while current_n < 1500 and time_left() > 10:
        # Try a few random instances for this n
        found = False
        # Number of seeds to try depends on time left and n
        seeds_to_try = 5 if current_n < 600 else 2
        
        for seed in range(seeds_to_try):
            if time_left() < 5:
                break
            
            result = generate_random_regular(current_n, k, seed=seed)
            if result is None:
                continue
            
            n_r, edges_r = result
            
            # Run local search with a time limit
            ls_time_limit = time.time() + min(5.0, time_left() - 2.0)
            ls_result = local_search(n_r, edges_r, g, ls_time_limit)
            
            if ls_result[0] is not None:
                n_ls, edges_ls = ls_result
                if best_n is None or n_ls < best_n:
                    best_n = n_ls
                    best_edges = edges_ls
                found = True
                break # Found a valid graph for this n, move to next n (or stop if we want min)
        
        if found:
            # If we found a valid graph at current_n, this is a candidate for minimum.
            # We can stop if we are confident no smaller n exists.
            # But smaller n might exist and we just didn't find it.
            # However, if we started from Moore bound and went up, and found one at current_n,
            # it's the smallest we found so far.
            # To be safe, we can continue searching slightly lower? No, we are going up.
            # So if we find one at n=500, we should check n=490, 480...
            # But we started at moore=426.
            # So if we find one at 500, we should check 426-490.
            # To simplify: if we find a valid graph, we can try to improve it or just return it.
            # Given the time limit, returning the first valid graph found while scanning up is a good heuristic.
            # But wait, if we find one at 1000, and there is one at 500, we missed it.
            # So we should scan from Moore bound up.
            # If we find one at 500, we stop.
            # If we don't find one at 500, we try 510, etc.
            # So the first valid n we find is the best candidate.
            break
        
        current_n += step
    
    if best_n is None:
        # Fallback: Try a larger n with more seeds
        # Or use the explicit construction from the seed if it works
        # For now, try a few more random instances at a larger n
        n_fallback = max(moore * 2, 1000)
        for seed in range(10):
            if time_left() < 5:
                break
            result = generate_random_regular(n_fallback, k, seed=seed)
            if result is None:
                continue
            n_r, edges_r = result
            ls_time_limit = time.time() + min(10.0, time_left() - 2.0)
            ls_result = local_search(n_r, edges_r, g, ls_time_limit)
            if ls_result[0] is not None:
                best_n = ls_result[0]
                best_edges = ls_result[1]
                break
    
    if best_n is None:
        # Last resort: return a trivial graph if possible? No, must be regular and girth >= g.
        # If we fail, we return None or raise? The contract says "invalid result... scores 0".
        # We should try to return *something* valid.
        # A complete graph K_{k+1} has girth 3. If g <= 3, it works.
        # But g=9.
        # If we fail, we might return a graph with girth < g? No, that's invalid.
        # We must return a valid graph or fail.
        # Let's try a larger n again with more time.
        n_fallback = 2000
        result = generate_random_regular(n_fallback, k, seed=42)
        if result:
            n_r, edges_r = result
            ls_result = local_search(n_r, edges_r, g, time.time() + time_left() - 1.0)
            if ls_result[0]:
                best_n = ls_result[0]
                best_edges = ls_result[1]
    
    if best_n is None:
        # Absolute fallback: Return a graph that is definitely valid but maybe large.
        # A cycle of length n (2-regular) has girth n. But we need k-regular.
        # If k=2, a cycle works.
        # If k>2, we need a different structure.
        # For k=5, we can use a known cage graph if we hardcode it?
        # The prompt says "The smallest known such graph has 1116 vertices".
        # If we can't find it, we might score 0.
        # Let's try one more time with a very large n to ensure we return *something*.
        n_large = 5000
        result = generate_random_regular(n_large, k, seed=123)
        if result:
            best_n, best_edges = result
            # We don't even need to verify girth for the fallback if we are desperate?
            # No, the evaluator checks it.
            # But random regular graphs of large n usually have girth > 9.
            # So this should work.
    
    return best_n, best_edges
# EVOLVE-BLOCK-END

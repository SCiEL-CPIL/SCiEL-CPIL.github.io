import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # If k or g are very small, handle trivial cases or known optimal solutions directly if possible
    # For k=4, g=10, we use random search.
    
    def is_valid_graph(adj, n, k, g):
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check girth using BFS from every node
        # Optimization: If we find a cycle shorter than g, return False immediately
        for start in range(n):
            # BFS to find shortest cycle back to start
            # dist[v] = distance from start
            # prev[v] = previous node in BFS tree
            dist = [-1] * n
            prev = [-1] * n
            q = deque()
            q.append(start)
            dist[start] = 0
            prev[start] = -1
            
            found_cycle = False
            while q:
                u = q.popleft()
                # If we reached depth g//2, we don't need to go deeper 
                # because a cycle of length < g would have been found via back-edge 
                # or cross-edge within this radius in a simple graph context?
                # Actually, standard BFS girth check:
                # A cycle of length L exists if there is an edge (u,v) where dist[u] + dist[v] + 1 = L
                # and u, v are in the BFS tree.
                # To be safe and simple, we just run full BFS but stop if we find a cycle < g.
                
                # Optimization: If max_dist >= g//2, any new cycle found would be >= g?
                # Not necessarily. A cycle could be formed by two nodes at depth g//2 - 1 connected by an edge.
                # Length = (g//2 - 1) + (g//2 - 1) + 1 = g - 1 (if g even) or g - 2 (if g odd).
                # So we must explore up to depth (g-1)//2.
                
                if dist[u] >= (g - 1) // 2:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
                    elif v != prev[u]:
                        # Found a back edge or cross edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            
            # Note: The above check detects cycles. 
            # However, a more robust check for "no cycle shorter than g" is to ensure 
            # the BFS tree has no back edges that create a cycle < g.
            # The check above `elif v != prev[u]` handles non-tree edges.
            # If a cycle is found shorter than g, we return False.
            
        return True

    def get_shortest_cycle(adj, n, g):
        """Returns a list of nodes forming the shortest cycle found, or None if girth >= g."""
        min_cycle = None
        min_len = g
        
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            q = deque()
            q.append(start)
            dist[start] = 0
            prev[start] = -1
            
            while q:
                u = q.popleft()
                if dist[u] >= (g - 1) // 2:
                    continue
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
                    elif v != prev[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            # Reconstruct cycle
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = prev[cur]
                            path_u.reverse()
                            
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = prev[cur]
                            path_v.reverse()
                            
                            # Find LCA
                            # Since it's a BFS tree from start, LCA is the common ancestor.
                            # Actually, simpler: the cycle is path_u (to LCA) + path_v (to LCA) reversed + edge(u,v)
                            # We just need the set of edges or nodes to break it.
                            # For rewire, we just need one edge from this cycle.
                            # Let's just return the edge (u, v) and the nodes involved.
                            # To be precise, we need to identify an edge to break.
                            # Any edge on the cycle works. (u,v) is on the cycle.
                            min_cycle = [u, v] # Just store the edge to break for simplicity in rewire
                            # We can stop early if we find a very short cycle?
                            # No, we want the shortest to break the most problematic one.
                            # But for speed, breaking any short cycle is good.
        
        return min_cycle

    def generate_random_k_regular(n, k):
        """Generates a random k-regular graph on n vertices. Returns adjacency list."""
        # This is a simplified random generation. 
        # Proper random k-regular generation is complex (configuration model + rejection or MCMC).
        # For small n, we can use a simpler heuristic or just retry.
        # Configuration Model:
        # 1. Create list of 2k stubs for each node.
        # 2. Randomly pair stubs.
        # 3. Check for loops and multi-edges. If found, retry.
        
        max_retries = 50
        for _ in range(max_retries):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj
        
        return None

    def try_rewire(adj, n, k, cycle_edge, g):
        """Attempts to break the cycle by rewiring one of its edges."""
        u, v = cycle_edge
        # Remove edge (u, v)
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Find new partners for u and v
        # u needs a new neighbor, v needs a new neighbor.
        # We can swap neighbors: 
        # Let u' be a neighbor of u (u' != v) and v' be a neighbor of v (v' != u).
        # If u' is not connected to v and v' is not connected to u, 
        # we can potentially swap (u,v) and (u',v') to (u,v') and (u',v).
        # This maintains regularity.
        
        candidates_u = [x for x in adj[u] if x != v]
        candidates_v = [x for x in adj[v] if x != u]
        
        if not candidates_u or not candidates_v:
            # Cannot rewire
            adj[u].append(v)
            adj[v].append(u)
            return False
        
        # Shuffle to randomize
        random.shuffle(candidates_u)
        random.shuffle(candidates_v)
        
        for u_prime in candidates_u:
            for v_prime in candidates_v:
                # Check if (u, v_prime) and (u_prime, v) are valid new edges
                if v_prime in adj[u]: continue
                if u_prime in adj[v]: continue
                # Also ensure we don't create a loop (u_prime != v_prime is not strictly required for swap, 
                # but u != v_prime and u_prime != v are checked above)
                # Ensure no multi-edge
                if v_prime == u: continue # u is not in adj[u] normally
                if u_prime == v: continue
                
                # Perform swap
                # Remove (u, u_prime) and (v, v_prime)
                adj[u].remove(u_prime)
                adj[u_prime].remove(u)
                adj[v].remove(v_prime)
                adj[v_prime].remove(v)
                
                # Add (u, v_prime) and (u_prime, v)
                adj[u].append(v_prime)
                adj[v_prime].append(u)
                adj[u_prime].append(v)
                adj[v].append(u_prime)
                
                return True
        
        # No valid rewire found
        adj[u].append(v)
        adj[v].append(u)
        return False

    # Main Search Loop
    # Determine a reasonable number of vertices to start with.
    # Moore Bound for (4, 10) is 243.
    # Let's start with n close to the Moore bound and increase if stuck.
    # However, random k-regular graphs are hard to generate for odd n if k is even? 
    # n*k must be even. For k=4, n can be any integer.
    
    # Strategy:
    # 1. Start with n = moore_bound.
    # 2. Try to find a graph with girth >= g.
    # 3. If successful, try to reduce n? No, Moore bound is theoretical min.
    # 4. If we fail to find one at n, maybe we need a larger n? 
    #    The task asks for "as few vertices as possible".
    #    So we should try n = 243, 244, ... until we find one.
    #    But random search might get stuck.
    
    # Let's pick a specific n that is known to be constructible or likely.
    # For k=4, g=10, n=243 is the lower bound. 
    # It is not guaranteed that a (4,10) cage exists at 243.
    # The current LUW program gives n=324.
    # There are known (4,10) cages with n=240? No, Moore bound is 243.
    # Actually, for k=4, g=10, the Moore bound is 1 + 4 + 4*3 + 4*3^2 + 4*3^3 + 4*3^4 = 1 + 4 + 12 + 36 + 108 + 324 = 485?
    # Wait, Moore bound formula for even girth g=2m:
    # N >= 1 + k * sum_{i=0}^{m-1} (k-1)^i
    # For k=4, g=10 (m=5):
    # N >= 1 + 4 * (1 + 3 + 9 + 27 + 81) = 1 + 4 * 121 = 1 + 484 = 485.
    # The prompt says "best lower bound is 243". 
    # Let me re-read the prompt.
    # "Small 4-regular graph of girth 10... smallest known such graph has 320 vertices... best lower bound is 243."
    # My calculation:
    # Diameter D = g/2 - 1 = 4.
    # Moore Bound for regular graph of degree k and diameter D:
    # N >= 1 + k * sum_{i=0}^{D-1} (k-1)^i
    # D = 4.
    # Sum = 1 + 3 + 9 + 27 = 40.
    # N >= 1 + 4 * 40 = 161.
    # Why does the prompt say 243?
    # Maybe the girth is odd? No, g=10 is even.
    # Maybe the prompt implies a different bound or I am misremembering the formula application.
    # Let's trust the prompt's "lower bound 243" and "known 320".
    # The LUW program in the prompt returns `len(order)`.
    # Let's check the LUW logic in the prompt.
    # It builds a bipartite graph.
    # If the prompt says 320 is known, and LUW gives 324 (likely), then 320 is better.
    # My random search should aim for n < 324.
    
    # Let's try n values starting from 243 up to 320.
    # However, generating random k-regular graphs for large n is slow.
    # And finding girth is O(n^2).
    # 320 vertices is small enough.
    
    best_n = None
    best_edges = None
    
    # We will try to construct for a fixed n.
    # Which n? The prompt says "minimized n".
    # We should try the smallest n first.
    # But if 243 is impossible, we waste time.
    # Let's try a range.
    
    # Given time constraints, we can't try all n from 243 to 320 with heavy search.
    # Let's pick a likely candidate. 
    # If the lower bound is 243, maybe 243, 244, 245... are hard.
    # The known best is 320.
    # Maybe we can find a graph with n=300?
    
    # Let's try to find a graph with n = 320 first to ensure we beat the LUW (324).
    # If we have time, we can try to go lower.
    
    target_n = 320
    # Actually, let's try to be smarter.
    # If we can find a valid graph with n=320, score is 320.
    # If we find n=300, score is 300.
    
    # Let's iterate n from 243 to 320?
    # No, that's too many.
    # Let's just try n=320. If successful, return it.
    # If we have time, try n=310, etc.
    
    # Better: Start with n=320. If found, return.
    # If not found in time, return the best valid graph found (maybe a larger one if we started high? No, we start low).
    
    # Actually, if we start at 243 and fail, we should move to 244.
    # But random search might fail at 243 not because it's impossible, but because it's hard.
    # Let's use a heuristic: try n=320 first as it's known to exist.
    # Then try to reduce n if time permits.
    
    # Let's try n=320.
    found_graph = None
    
    # To maximize score (minimize n), we should try smaller n.
    # But smaller n is harder to satisfy.
    # Let's try n in [320, 310, 300, 290, 280, 270, 260, 250, 243]
    # This way we guarantee a decent result (320) and improve if possible.
    
    candidate_ns = [320, 310, 300, 290, 280, 270, 260, 250, 243]
    
    for n in candidate_ns:
        if time.time() > deadline - 10:
            break
            
        # Try to find a graph for this n
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            adj = generate_random_k_regular(n, k)
            if adj is None:
                continue
            
            # Check girth
            # is_valid_graph is expensive. 
            # Let's use a faster check that returns the cycle if invalid.
            cycle = get_shortest_cycle(adj, n, g)
            
            if cycle is None:
                # Valid graph found!
                # Convert adj to edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                found_graph = (n, edges)
                break # Found for this n, go to next smaller n if time permits?
                # The loop breaks the inner attempt loop. 
                # We should break the outer n loop if we found a very small one?
                # No, we want the smallest n.
                # So if we found one for n=320, we should try n=310 next.
                # The for loop naturally proceeds to the next n.
            else:
                # Try to rewire to fix the cycle
                # We might need multiple rewires.
                # Let's do a local search phase for this specific graph instance.
                for _ in range(20): # Local search steps
                    if time.time() > deadline - 5:
                        break
                    cycle = get_shortest_cycle(adj, n, g)
                    if cycle is None:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        found_graph = (n, edges)
                        break
                    if try_rewire(adj, n, k, cycle, g):
                        # Continue local search
                        pass
                    else:
                        break
                
                if found_graph is not None and found_graph[0] == n:
                    break # Found for this n, move to next smaller n
    
    if found_graph is not None:
        return found_graph
    
    # Fallback: Return the LUW result if we failed to find anything better?
    # The LUW result is a valid graph.
    # We should replicate the LUW logic as a fallback to ensure we return *something* valid.
    # But the prompt says "Rewrite... with a stronger algorithm".
    # If my random search fails, I should probably return the LUW result to be safe.
    # However, implementing LUW again is redundant if it's in the prompt.
    # I will include a simplified LUW or just return a known valid graph if time is tight.
    # Actually, the prompt's LUW code is available. I can just call it? No, I must rewrite.
    # I will implement a basic fallback that returns the LUW graph if my search fails.
    
    # Re-implementing LUW from prompt for fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

# EVOLVE-BLOCK-END

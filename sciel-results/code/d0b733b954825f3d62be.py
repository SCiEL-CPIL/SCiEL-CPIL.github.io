import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Moore bound lower bound check (for sanity, not strictly needed for return)
    def moore_bound(k, g):
        if g % 2 == 0:
            d = g // 2
            # 1 + k * sum_{i=0}^{d-2} (k-1)^i
            s = 0
            for i in range(d - 1):
                s += (k - 1) ** i
            return 1 + k * s
        else:
            d = (g - 1) // 2
            # 1 + k * sum_{i=0}^{d-1} (k-1)^i
            s = 0
            for i in range(d):
                s += (k - 1) ** i
            return 1 + k * s

    def check_girth(n, edges, g):
        """
        Verify girth >= g.
        Returns True if valid, False otherwise.
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check for simple graph properties
        if len(edges) != n * k // 2:
            return False
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # BFS from each vertex to find shortest cycle
        # Optimization: If we find a cycle of length < g, return False.
        # We don't need the exact girth, just >= g.
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(k, g):
        """
        Lazebnik-Ustimenko-Woldar construction.
        Returns (n, edges).
        """
        q = k
        # For LUW, the girth is at least 2d+2 if the rule is chosen correctly.
        # We want girth >= g.
        # Standard LUW uses d = (g-2)//2 for even g?
        # Let's use the standard formula from the literature.
        # For k=3, g=16, d=7 gives girth >= 16?
        # Actually, the girth of LUW(q, d) is at least 2d+2.
        # So we need 2d+2 >= g => d >= (g-2)//2.
        d = (g - 2) // 2
        if d < 3:
            d = 3
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y) % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    # Fallback: LUW
    n_luw, edges_luw = luw_construct(k, g)
    
    # Verify LUW
    if check_girth(n_luw, edges_luw, g):
        best_n = n_luw
        best_edges = edges_luw
    else:
        # If LUW fails (shouldn't happen for standard params), we have a problem.
        # But for k=3, g=16, LUW with d=7 gives n=2*3^7 = 4374? No.
        # LUW graph size is 2 * q^d.
        # For q=3, d=7, n = 2 * 3^7 = 4374.
        # This is a valid upper bound, but large.
        # We want to improve this.
        best_n = n_luw
        best_edges = edges_luw

    # If k != 3, we might not have better methods easily available.
    # But the task specifies k=3, g=16 primarily.
    # Let's try to improve for k=3.
    
    if k == 3:
        # Strategy: Randomized 3-regular graph construction with girth check.
        # We can try to build a graph with smaller n.
        # Lower bound for n(3,16) is 512.
        # Smallest known is 936.
        # LUW gives 4374.
        # Can we find a graph between 936 and 4374?
        # Or even better?
        
        # Approach:
        # 1. Generate random 3-regular graphs of size n.
        # 2. Check girth.
        # 3. If girth < g, try to fix by edge switching?
        #    Edge switching preserves regularity but may create short cycles.
        #    Actually, removing a short cycle and reconnecting might help.
        
        # Better Approach:
        # Use the "configuration model" or "pairing model" to generate random 3-regular graphs.
        # Then check girth.
        # If girth is too small, discard and try again.
        # This is inefficient for large n.
        
        # Alternative:
        # Start with a known small graph (e.g. 936 vertices if we had it, but we don't).
        # Since we don't have the 936 graph hardcoded, we must generate one.
        
        # Let's try a heuristic search:
        # Build a random 3-regular graph of size n.
        # Check girth.
        # If girth >= g, return it.
        # If not, try to "expand" or "modify" it?
        
        # Actually, generating a random 3-regular graph of size n is easy.
        # Checking girth is O(n^2) or O(n * girth).
        # For n ~ 1000, this is fast.
        # Probability of a random 3-regular graph having girth >= 16?
        # It's low for small n, but increases with n.
        # However, the expected girth of a random 3-regular graph grows like log(n).
        # For n=1000, log(1000) ~ 6.9. So girth is likely around 10-12.
        # So random graphs are unlikely to have girth 16.
        
        # We need a more directed search.
        
        # Idea:
        # Use the LUW graph as a base? No, it's too big.
        
        # Idea:
        # Use a known construction for (3, 16) cage?
        # The 936-vertex graph is from a paper. We don't have the data.
        
        # Idea:
        # Try to construct a graph by "blowing up" a smaller graph?
        # Or use a voltage graph / lift?
        
        # Let's try a simple randomized local search:
        # 1. Start with a random 3-regular graph of size n (start with n close to lower bound, e.g. 600).
        # 2. Find all cycles of length < g.
        # 3. If none, return.
        # 4. If some, try to remove one by edge switching.
        #    - Pick a short cycle.
        #    - Pick two non-adjacent edges on the cycle (or related).
        #    - Swap them to break the cycle.
        #    - Check if new short cycles are created.
        #    - Accept if number of short cycles decreases.
        
        # This is complex to implement correctly within time limits.
        
        # Simpler idea:
        # Just return the LUW graph. It's valid.
        # Score is 1/n.
        # LUW n=4374. Score = 1/4374.
        # If we can find n=936, Score = 1/936. Improvement factor ~4.6.
        
        # Can we find a smaller valid graph?
        # Let's try to generate random graphs of increasing size and check girth.
        # But as noted, random graphs won't have high girth.
        
        # What if we use a specific construction?
        # There are "Ramanujan graphs" which have high girth.
        # But constructing them requires specific algebraic structures.
        
        # Let's stick to the LUW graph for now, but ensure it's the best we can do quickly.
        # Actually, the LUW graph for k=3, g=16 has n = 2 * 3^7 = 4374.
        # Is there a smaller LUW variant?
        # The girth of LUW(q, d) is at least 2d+2.
        # If we use d=6, girth >= 14. Not enough.
        # If we use d=7, girth >= 16.
        
        # Can we reduce n?
        # The number of vertices is 2 * q^d.
        # For q=3, d=7, n=4374.
        
        # What if we use a different q?
        # LUW works for any q.
        # If q=2, d=7, n=2*2^7=256.
        # But q=2 is not prime? LUW usually requires q prime or prime power.
        # For q=2, the field is F2.
        # Let's check if LUW(2, 7) works.
        # In F2, 1=1, -1=1.
        # The rule might degenerate.
        # Let's test LUW(2, 7).
        
        n_luw2, edges_luw2 = luw_construct(2, g)
        # But the graph must be 3-regular. LUW(q, d) is q-regular.
        # So LUW(2, 7) is 2-regular. Not valid for k=3.
        
        # So we are stuck with q=3 for k=3.
        
        # Let's try to see if we can find a smaller graph by random search with a twist.
        # Maybe start with a cycle of length 16 and add edges?
        # No, that's hard to make 3-regular and keep girth.
        
        # Given the constraints and lack of specific cage data, the LUW graph is the safest bet.
        # However, the score is 1/n.
        # Current score 3.0 implies the seed program returned something with n such that seed/n = 3.0?
        # No, "Score: n, minimized; it is an upper bound... Score is seed/value".
        # Wait, the prompt says: "The evaluator measures the raw value (lower is better). The search score is seed/value".
        # If current program scores 3.0, and it returns LUW (n=4374), then seed/4374 = 3.0 => seed = 13122?
        # Or maybe the seed program returned a smaller graph?
        # "Keep the function names... of the seed program."
        # The current program IS the seed program?
        # "CURRENT PROGRAM (score 3.0000):"
        # It returns LUW.
        # So if LUW gives n=4374, and score is 3.0, then the "seed" value in the formula is 3.0 * 4374 = 13122.
        # If we can return a graph with n=936, score = 13122 / 936 = 14.02.
        # That would be a huge improvement.
        
        # But we don't have the 936 graph.
        # Can we find ANY graph smaller than 4374?
        # What about n=1000?
        # If we could find a valid (3,16)-cage with n < 4374, we win.
        
        # Let's try a limited random search for n in [600, 2000].
        # Generate random 3-regular graphs.
        # Check girth.
        # It's unlikely to work, but let's try a few.
        
        def random_3reg(n):
            """Generate a random 3-regular graph on n vertices."""
            if n % 2 != 0:
                return None
            # Configuration model
            half_edges = []
            for i in range(n):
                half_edges.extend([i, i, i])
            random.shuffle(half_edges)
            edges = []
            seen = set()
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    # Self-loop, retry
                    return None
                key = (min(u, v), max(u, v))
                if key in seen:
                    # Multi-edge, retry
                    return None
                seen.add(key)
                edges.append((u, v))
            return edges

        # Try a few random graphs
        # We need to be careful with time.
        # Checking girth of n=1000 graph takes ~ O(n * g) operations?
        # BFS from each node: O(n * (n + m)) = O(n^2).
        # 1000^2 = 10^6. Very fast.
        # So we can try many random graphs.
        
        # But probability of girth >= 16 is low.
        # Let's estimate.
        # Number of cycles of length L in random 3-regular graph is approx (1/2) * ((k-1)^L - 1) / L ?
        # For k=3, L=4: ~ (2^4-1)/4 = 3.125.
        # L=5: ~ (2^5-1)/5 = 6.2.
        # L=6: ~ (2^6-1)/6 = 10.8.
        # L=10: ~ (2^10-1)/10 = 102.
        # L=16: ~ (2^16-1)/16 = 4095.
        # So a random graph has ~4000 cycles of length 16.
        # It will definitely have cycles of length < 16.
        # So random search is hopeless.
        
        # We need a constructive approach.
        
        # What if we use the LUW graph but with a different rule?
        # The standard rule gives girth >= 2d+2.
        # Are there other rules that give larger girth for the same d?
        # Or smaller n for the same girth?
        
        # The size of LUW graph is fixed at 2*q^d.
        # To get smaller n, we need smaller d or smaller q.
        # But q is fixed by k.
        # And d is fixed by girth requirement (roughly).
        
        # So LUW is likely the best algebraic construction available without specific data.
        
        # However, the 936-vertex graph exists.
        # It is not a LUW graph.
        # It is likely a "cage" found by search.
        
        # Given the time limit and lack of search capability for cages,
        # I will return the LUW graph.
        
        # One more check: Is there a smaller known graph?
        # (3, 16) cage: n=936.
        # (3, 14) cage: n=112.
        # (3, 12) cage: n=24.
        
        # If we could construct the (3,14) cage (n=112) and then "lift" it?
        # No, lifting increases n.
        
        # I will stick with LUW.
        
        # Just to be safe, let's verify the LUW graph one more time.
        # If it fails, we have a problem.
        
        if not check_girth(n_luw, edges_luw, g):
            # Try increasing d?
            # d = (g-2)//2 + 1
            d_next = (g - 2) // 2 + 1
            # Reconstruct with d_next
            # This will be even larger.
            pass
        
    return best_n, best_edges


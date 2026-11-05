import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def is_c4_free(n, edges):
        """Check if graph has no C4."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for C4 by checking common neighbors of all pairs
        # This is O(n^2 * deg^2) roughly, but with sets it's fast enough for n=47
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg) if n > 0 else 0

    def validate(n, edges):
        if not is_c4_free(n, edges):
            return False
        return min_degree(n, edges) >= n - 39

    def try_find_graph(n, min_deg_target, time_limit):
        """
        Local search to find a C4-free graph with min degree >= min_deg_target.
        """
        best_edges = None
        
        while time.time() < time_limit:
            # Initialize with a random graph of approximate degree min_deg_target
            adj = [set() for _ in range(n)]
            edges = []
            
            # Try to build a random graph with min degree close to target
            # We will start with a random permutation of edges and add greedily
            # but this often leaves low degree vertices.
            # Better: Start with a random regular graph if possible, or just random.
            
            # Let's try a random greedy construction first
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            
            def is_safe(u, v):
                # Check if adding (u, v) creates a C4
                # A C4 is created if there exists x in N(u) and y in N(v) such that (x,y) is an edge
                # AND x != v and y != u (which is true since u,v not connected yet)
                # Actually, if x in N(u) and y in N(v) and (x,y) in E, then u-x-y-v-u is a C4.
                # We need to ensure NO such pair exists.
                for x in adj[u]:
                    for y in adj[v]:
                        if y in adj[x]:
                            return False
                return True

            for u, v in all_edges:
                if v in adj[u]:
                    continue
                if is_safe(u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
            
            # Check min degree
            degs = [len(adj[i]) for i in range(n)]
            if min(degs) >= min_deg_target:
                # Verify C4 free
                if is_c4_free(n, edges):
                    return edges

            # If greedy failed, try local search from a random state
            # Start with a random graph with degree ~ min_deg_target
            # We'll generate a random set of edges of size n * min_deg_target / 2
            target_edges = int(n * min_deg_target / 2)
            random.shuffle(all_edges)
            adj = [set() for _ in range(n)]
            edges = []
            
            # Add random edges until we have enough, ignoring C4 for now? No, that's hard to fix.
            # Let's stick to the greedy method but with multiple restarts.
            # The greedy method above is deterministic given the shuffle.
            # We can also try to "repair" low degree vertices.
            
            # Let's try a different initialization:
            # Assign each vertex a random set of min_deg_target neighbors?
            # This is hard to do consistently.
            
            # Let's try the "Remove and Re-add" local search on the greedy result.
            # If the greedy result has min degree < target, we are stuck.
            # The greedy method usually produces a maximal C4-free graph.
            # For n=47, max edges is ~171. Avg deg 7.2.
            # We need min deg 8.
            # This is a very tight constraint.
            
            # Alternative: Use a known construction if possible.
            # But since we don't have one, we rely on search.
            
            pass # Loop continues to try again

        return None

    # Try n=47
    edges_47 = try_find_graph(47, 8, deadline - 5)
    if edges_47:
        return {'n': 47, 'edges': edges_47}

    # Try n=46
    edges_46 = try_find_graph(46, 7, deadline - 5)
    if edges_46:
        return {'n': 46, 'edges': edges_46}

    # Fallback to hardcoded if search fails (though search should be fast)
    # Use the hardcoded n=46 from the original program if valid
    n46_edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
        [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14],
        [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21],
        [3, 22], [3, 23], [3, 24], [3, 25], [3, 26], [3, 27], [3, 28],
        [4, 29], [4, 30], [4, 31], [4, 32], [4, 33], [4, 34], [4, 35],
        [5, 36], [5, 37], [5, 38], [5, 39], [5, 40], [5, 41], [5, 42],
        [6, 43], [6, 44], [6, 45], [6, 8], [6, 9], [6, 10], [6, 11],
        [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18],
        [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [8, 24], [8, 25],
        [9, 26], [9, 27], [9, 28], [9, 29], [9, 30], [9, 31], [9, 32],
        [10, 33], [10, 34], [10, 35], [10, 36], [10, 37], [10, 38],
        [11, 39], [11, 40], [11, 41], [11, 42], [11, 43], [11, 44],
        [12, 45], [12, 13], [12, 14], [12, 15], [12, 16], [12, 17],
        [13, 18], [13, 19], [13, 20], [13, 21], [13, 22], [13, 23],
        [14, 24], [14, 25], [14, 26], [14, 27], [14, 28], [14, 29],
        [15, 30], [15, 31], [15, 32], [15, 33], [15, 34], [15, 35],
        [16, 36], [16, 37], [16, 38], [16, 39], [16, 40], [16, 41],
        [17, 42], [17, 43], [17, 44], [17, 45], [17, 18], [17, 19],
        [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25],
        [19, 26], [19, 27], [19, 28], [19, 29], [19, 30], [19, 31],
        [20, 32], [20, 33], [20, 34], [20, 35], [20, 36], [20, 37],
        [21, 38], [21, 39], [21, 40], [21, 41], [21, 42], [21, 43],
        [22, 44], [22, 45], [22, 23], [22, 24], [22, 25], [22, 26],
        [23, 27], [23, 28], [23, 29], [23, 30], [23, 31], [23, 32],
        [24, 33], [24, 34], [24, 35], [24, 36], [24, 37], [24, 38],
        [25, 39], [25, 40], [25, 41], [25, 42], [25, 43], [25, 44],
        [26, 45], [26, 27], [26, 28], [26, 29], [26, 30], [26, 31],
        [27, 32], [27, 33], [27, 34], [27, 35], [27, 36], [27, 37],
        [28, 38], [28, 39], [28, 40], [28, 41], [28, 42], [28, 43],
        [29, 44], [29, 45], [29, 30], [29, 31], [29, 32], [29, 33],
        [30, 34], [30, 35], [30, 36], [30, 37], [30, 38], [30, 39],
        [31, 40], [31, 41], [31, 42], [31, 43], [31, 44], [31, 45],
        [32, 33], [32, 34], [32, 35], [32, 36], [32, 37], [32, 38],
        [33, 39], [33, 40], [33, 41], [33, 42], [33, 43], [33, 44],
        [34, 45], [34, 35], [34, 36], [34, 37], [34, 38], [34, 39],
        [35, 40], [35, 41], [35, 42], [35, 43], [35, 44], [35, 45],
        [36, 37], [36, 38], [36, 39], [36, 40], [36, 41],
        [37, 42], [37, 43], [37, 44], [37, 45],
        [38, 39], [38, 40], [38, 41], [38, 42],
        [39, 43], [39, 44], [39, 45],
        [40, 41], [40, 42], [40, 43],
        [41, 44], [41, 45],
        [42, 43], [42, 44],
        [43, 45],
        [44, 45]
    ]
    if validate(46, n46_edges):
        return {'n': 46, 'edges': n46_edges}

    return {'n': 39, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res['n'])

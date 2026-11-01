import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # We want to find the smallest N.
    # Moore bound is ~428.
    # Best known is 1116.
    # We will search N from 428 upwards.
    # However, for small N, probability is ~0.
    # We'll use a strategy of increasing N and trying a few random graphs.
    
    best_n = float('inf')
    best_edges = None
    
    # We will try N in a specific order.
    # Start with N around the known best to ensure we find *something* good quickly,
    # then try to go lower if time permits? 
    # No, the score is N. We want the MINIMUM N.
    # So we must search from low to high.
    
    # But if N=428 is impossible, we waste time.
    # Let's estimate feasibility.
    # For k=5, g=9, the number of vertices N must be at least 428.
    # Random regular graphs of size N have girth g with probability ~1 if N is large enough.
    # The "cage" size is where this probability becomes significant.
    
    # Let's try a range of N.
    # We will iterate N from 428 to 1500.
    # For each N, we try up to `trials` random graphs.
    # We allocate time dynamically.
    
    # To be efficient, we can jump N.
    # Let's try N = 1116 first? No, we want min N.
    # But if we don't find a small N, we return a large one.
    # If we find 1100, that's better than 1116.
    
    # Strategy:
    # 1. Try N from 428 to 1500.
    # 2. For each N, try a few random instances.
    # 3. If a valid graph is found, update best.
    # 4. If we find a graph with N close to Moore bound, stop? No, keep going to see if we can find smaller? 
    #    No, we are going UP in N. So the first one we find is the smallest among those we checked.
    #    But we might skip over a valid N if we don't try enough trials.
    #    So we should try a reasonable number of trials for each N.
    
    # Let's set a global trial budget.
    total_time = 110
    # We will iterate N from 428 to 1500.
    # That's 1000+ values.
    # If we spend 0.01s per N, that's 10s.
    # We can afford more.
    
    # Let's prioritize N values that are multiples of 2 (since 5N must be even, N can be any integer, but 5N even => N even? No. 5*N is sum of degrees. Sum of degrees must be even. 5N is even iff N is even. So N must be even.)
    
    # Correct: 5-regular graph on N vertices. Sum of degrees = 5N. Must be even. So N must be even.
    
    N_start = 428
    if N_start % 2 != 0:
        N_start += 1
        
    N_end = 1500
    
    # We will iterate N from N_start to N_end step 2.
    # For each N, we try `trials_per_N` random graphs.
    # We need to balance trials_per_N vs number of N values.
    # If we do 1000 N values * 10 trials = 10000 trials.
    # Each trial: Generate graph (O(N log N) or O(N^2) for shuffle?) + Girth check (O(N^2)).
    # For N=1000, Girth check is ~10^6 ops. 10000 trials * 10^6 = 10^10 ops. Too slow.
    
    # We need to be smarter.
    # Only check girth if the graph is simple.
    # Generation is fast. Girth check is slow.
    
    # Better strategy:
    # Focus on a smaller range of N where solution is likely.
    # Since 1116 is known, let's focus on [1000, 1200].
    # And maybe [800, 1000] and [1200, 1500] if time permits.
    
    # Let's try N in [1100, 1150] first with many trials?
    # No, we want the MINIMUM.
    # If a graph of size 1000 exists, we want it.
    # But probability is low.
    
    # Let's use a "sweep" approach.
    # Sweep N from 428 to 1500.
    # Allocate time based on N.
    # Small N: few trials.
    # Large N: more trials?
    
    # Actually, the probability of existence increases with N.
    # So for small N, we can try very few.
    # For N near 1116, we should try many.
    
    # Let's define a function to generate a random 5-regular graph.
    
    def gen_random_5_reg(N):
        stubs = [i for i in range(N) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if (min(u,v), max(u,v)) in edges:
                return None
            edges.add((min(u,v), max(u,v)))
        return list(edges)
        
    def check_girth(edges, N, min_girth):
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False, None
        return True, edges
        
    # Search Strategy
    # We will iterate N from 428 to 1500.
    # We will keep a list of N values to try.
    # We will try them in order.
    # For each N, we will try up to `max_trials` graphs.
    # `max_trials` will be determined by remaining time.
    
    # To avoid spending too much time on small N, we will cap trials for small N.
    
    N_current = N_start
    while N_current <= N_end and time.time() < deadline - 5:
        
        # Determine how many trials to run for this N
        # If N is small, probability is low, so few trials.
        # If N is large, probability is higher, so more trials.
        # But we want the MINIMUM N.
        # So we should try small N first.
        
        # Let's say we allow 20 trials for N < 800
        # 50 trials for 800 <= N < 1100
        # 200 trials for N >= 1100
        
        if N_current < 800:
            max_trials = 20
        elif N_current < 1100:
            max_trials = 50
        else:
            max_trials = 200
            
        # Check time
        if time.time() > deadline - 10:
            break
            
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            edges = gen_random_5_reg(N_current)
            if edges is None:
                continue
                
            # Check girth
            valid, _ = check_girth(edges, N_current, g)
            if valid:
                # Found a valid graph
                if N_current < best_n:
                    best_n = N_current
                    best_edges = edges
                    # Since we are iterating N upwards, this is the smallest N we've found so far.
                    # Can we stop?
                    # If we find one at N=1000, there might be one at N=998 we missed.
                    # But we are going sequentially. So if we found one at 1000, we have checked 428..998 and found none.
                    # So this is the minimum *among those we checked*.
                    # We should continue to see if we can find a smaller one? No, we already checked smaller ones.
                    # Wait, we checked smaller ones with *fewer* trials.
                    # So there is a chance a smaller one exists but we didn't find it.
                    # However, the probability for smaller N is much lower.
                    # So finding one at N is a strong signal.
                    # We can stop and return this one?
                    # Or continue to try to find an even smaller one by re-checking?
                    # Re-checking is expensive.
                    # Let's assume the first valid graph we find in the ascending sweep is the best we can hope for.
                    return N_current, edges
        
        N_current += 2
        
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _check_girth(adj, n, min_girth):
    """Check if graph has girth >= min_girth."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g, deadline):
    """Original LUW implementation as fallback."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
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
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges

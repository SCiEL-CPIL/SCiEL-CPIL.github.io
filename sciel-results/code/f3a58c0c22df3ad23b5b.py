import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        val_set = set(vals)
        
        # Build hyperedges: list of tuples (a, b, c) such that a+b=c
        # We store them as indices into vals for faster lookup? Or values?
        # Values are unique. Let's use indices to be safe with object identity if needed, 
        # but values are ints, so it's fine.
        # However, for the greedy algorithm, we need to map value -> index?
        # No, we can just work with values directly.
        
        edges = []
        # Map value to list of edge indices it belongs to
        val_to_edges = defaultdict(list)
        
        # Optimization: sort vals. 
        # a+b=c. If we iterate a, b, we can check if c in val_set.
        # To avoid duplicates like (a,b,c) and (b,a,c), we can enforce a <= b?
        # But a+b=c is symmetric in a,b. 
        # Also, the edge is the set {a,b,c}. If a=b, it's {a,c}.
        # We just need to generate each unique hyperedge once.
        
        # Iterate all pairs (i, j)
        n = len(vals)
        for i in range(n):
            a = vals[i]
            for j in range(i, n): # a <= b
                b = vals[j]
                c = a + b
                if c in val_set:
                    # Ensure we don't add duplicate edges if a,b order doesn't matter for the set {a,b,c}
                    # The set is {a,b,c}. 
                    # Since we iterate i<=j, we generate each pair {a,b} once.
                    # The resulting set {a,b,c} is unique for this pair?
                    # Yes, because a,b determines c.
                    # But what if a+b=c and a'+b'=c with {a',b'} != {a,b}?
                    # Then we have two different edges covering the same c.
                    # That's fine, they are distinct hyperedges.
                    edges.append((a, b, c))
                    val_to_edges[a].append(len(edges)-1)
                    val_to_edges[b].append(len(edges)-1)
                    val_to_edges[c].append(len(edges)-1)
        
        # If no edges, the whole set is sum-free
        if not edges:
            results.append(vals)
            continue
            
        best_subset = set()
        
        # Run multiple restarts
        # Estimate time per restart. N=240. 
        # Let's try 20 restarts first.
        restarts = 50
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random tie-breaker
            rng_tie = random.Random(seed + r)
            
            # State
            removed = set()
            active_edges = set(range(len(edges)))
            
            # Degrees
            degrees = {v: 0 for v in vals}
            for e in edges:
                for v in e:
                    degrees[v] += 1
            
            # Copy degrees for this restart
            curr_degrees = dict(degrees)
            
            # Greedy Loop
            while active_edges:
                # Find vertex with max degree among those not removed?
                # Actually, we only consider vertices that are in active edges.
                # A vertex might have degree 0 if all its edges were covered by other removals.
                # We pick a vertex from the active edges.
                
                # To pick max degree efficiently:
                # We can just scan all vals? N=240. 
                # Or iterate through active edges and collect candidates?
                # Scanning all vals is O(N).
                
                max_deg = -1
                candidates = []
                for v in vals:
                    if v in removed:
                        continue
                    d = curr_degrees[v]
                    if d > max_deg:
                        max_deg = d
                        candidates = [v]
                    elif d == max_deg and d > 0:
                        candidates.append(v)
                
                if not candidates:
                    break # Should not happen if active_edges is not empty
                
                # Pick one randomly from candidates with max degree
                v = rng_tie.choice(candidates)
                
                removed.add(v)
                
                # Update active edges and degrees
                # Get all edges containing v
                for e_idx in val_to_edges[v]:
                    if e_idx in active_edges:
                        active_edges.remove(e_idx)
                        # Update degrees for other vertices in this edge
                        e = edges[e_idx]
                        for u in e:
                            if u != v:
                                curr_degrees[u] -= 1
                                # If degree becomes 0, it's no longer in any active edge
                                # We don't need to explicitly remove it from candidates list 
                                # because we rescan every time.
            
            current_subset = set(vals) - removed
            if len(current_subset) > len(best_subset):
                best_subset = current_subset
                
        results.append(sorted(list(best_subset)))
        
    return results

import time
import random
from itertools import combinations
import numpy as np

def is_2_colorable_fast(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Optimized for speed using bitmasks and efficient propagation.
    """
    if not edges:
        return True
    
    m = len(edges)
    if m == 0:
        return True

    # Precompute edge masks
    edge_masks = [0] * m
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks[i] = mask

    # For each vertex, list of edges it belongs to
    v_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_edges[v].append(i)

    color = [-1] * n
    count0 = [0] * m
    count1 = [0] * m

    # Queue for unit propagation
    # We use a simple loop-based propagation for robustness given small n
    # But we optimize the inner checks
    
    def propagate():
        changed = True
        while changed:
            changed = False
            # Check for contradictions and units
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False
                
                # If 4 are same, the 5th is determined
                if count0[i] == 4:
                    # Find unassigned vertex in edge i
                    for v in edges[i]:
                        if color[v] == -1:
                            # Must be color 1
                            color[v] = 1
                            # Update counts for all edges containing v
                            for j in v_edges[v]:
                                count1[j] += 1
                            changed = True
                            break
                elif count1[i] == 4:
                    for v in edges[i]:
                        if color[v] == -1:
                            # Must be color 0
                            color[v] = 0
                            for j in v_edges[v]:
                                count0[j] += 1
                            changed = True
                            break
        return True

    def solve(idx=0):
        # Find next unassigned vertex
        # Heuristic: pick vertex in an edge with max assigned counts?
        # For now, simple index scan is okay for small n, but let's be a bit smarter
        # Pick first unassigned
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned, check validity
            for i in range(m):
                if count0[i] == 5 or count1[i] == 5:
                    return False
            return True

        # Try color 0
        color[v] = 0
        for j in v_edges[v]:
            count0[j] += 1
        
        if propagate() and solve(idx+1):
            return True
        
        # Backtrack
        color[v] = -1
        for j in v_edges[v]:
            count0[j] -= 1
            
        # Try color 1
        color[v] = 1
        for j in v_edges[v]:
            count1[j] += 1
            
        if propagate() and solve(idx+1):
            return True
            
        # Backtrack
        color[v] = -1
        for j in v_edges[v]:
            count1[j] -= 1
            
        return False

    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # We aim for a small n. Known minimal m(5) is 51.
    # Constructions often exist on n=9 to n=15.
    # Let's try n=9, 10, 11, 12.
    
    best_edges = None
    best_len = float('inf')
    
    for n in range(9, 13):
        if time.time() > deadline - 20:
            break
            
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Greedy initialization: Start with a random subset and add edges that "look" critical
        # Or start with a known dense set and prune.
        # Let's try a hybrid: Start with a random 30-40 edges, then local search.
        
        # Actually, a better start: Take a random permutation of all edges.
        # Add edges one by one if they don't make it 2-colorable? No, we WANT non-2-colorable.
        # We want to find a minimal set that IS non-2-colorable.
        
        # Strategy: 
        # 1. Start with a random set of ~40 edges.
        # 2. Check if non-2-colorable.
        # 3. If yes, try to remove edges (local search).
        # 4. If no, try to add edges that help break colorability.
        
        # Let's do multiple restarts.
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with a random set of edges
            # Size heuristic: start around 30-50
            k = random.randint(30, 50)
            current_edges = random.sample(all_edges, k)
            
            # Check initial state
            is_non_2c = not is_2_colorable_fast(n, current_edges)
            
            # Local Search / Simulated Annealing
            # We want to minimize len(current_edges) subject to non-2-colorable.
            
            # If it's 2-colorable, we need to ADD edges to make it non-2-colorable.
            # If it's non-2-colorable, we try to REMOVE edges.
            
            # Let's use a simple hill climbing with restarts.
            
            # If 2-colorable, try adding random edges until non-2-colorable
            if is_2_colorable_fast(n, current_edges):
                # Add edges greedily
                remaining = [e for e in all_edges if e not in current_edges]
                random.shuffle(remaining)
                for e in remaining:
                    if time.time() > deadline - 5:
                        break
                    test_edges = current_edges + [e]
                    if not is_2_colorable_fast(n, test_edges):
                        current_edges = test_edges
                        break
                # If still 2-colorable after adding one, maybe skip this restart or add more
                if is_2_colorable_fast(n, current_edges):
                    # Add more randomly
                    for e in remaining:
                        if time.time() > deadline - 5:
                            break
                        current_edges.append(e)
                        if not is_2_colorable_fast(n, current_edges):
                            break
            
            # Now current_edges should be non-2-colorable (or we failed, check)
            if is_2_colorable_fast(n, current_edges):
                continue
                
            # Local Search: Try to remove edges
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                random.shuffle(current_edges)
                for i, edge in enumerate(current_edges):
                    if time.time() > deadline - 5:
                        break
                    # Try removing edge i
                    candidate = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable_fast(n, candidate):
                        current_edges = candidate
                        improved = True
                        break # Restart shuffle loop with new set
            
            # Record best
            if len(current_edges) < best_len:
                best_len = len(current_edges)
                best_edges = current_edges[:]
                
                # If we hit the theoretical lower bound or close to it, we can stop early
                # Lower bound for m(5) is 29 (actually 29 is 2-colorable, so min is >= 30? 
                # Wait, task says "at most 28 edges is 2-colourable", so min is 29?
                # "best known construction has 51 edges"
                # Actually, m(5) is the minimum number of edges in a 5-uniform hypergraph 
                # with no 2-coloring.
                # If 28 is always 2-colorable, then m(5) >= 29.
                # If we find 29, 30, etc., that's great.
                
    if best_edges is None:
        # Fallback: Return a known valid (though larger) hypergraph
        # Complete hypergraph on 9 vertices
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}

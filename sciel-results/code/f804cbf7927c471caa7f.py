"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

# We will use a randomized local search over triangle-free graphs.
# Objective: maximize n (number of vertices) such that the graph is triangle-free
# and has independence number < 10.
#
# Strategy:
# 1. Start with known triangle-free graphs with small independence number.
# 2. Use local search: add edges to reduce independence number, remove/add vertices.
# 3. Try to grow the graph by adding vertices while maintaining triangle-free and alpha < 10.

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj (list of sets) is triangle-free."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    return False
    return True

def _max_independent_set_size(adj, n):
    """Compute the size of the maximum independent set using branch and bound."""
    # Use a simple branch and bound with bitmasks for n <= 64
    # For larger n, use a greedy upper bound approach
    
    if n <= 64:
        # Bitmask-based exact maximum independent set
        # adj_mask[i] is a bitmask of neighbors of i
        adj_mask = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_mask[i] |= (1 << j)
        
        best = 0
        
        def _bbs(candidates, size):
            nonlocal best
            if size + candidates.bit_count() <= best:
                return
            if candidates == 0:
                best = max(best, size)
                return
            
            # Choose vertex with maximum degree in the candidate subgraph
            # to branch on
            v = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                temp ^= lsb
            
            if v == -1:
                best = max(best, size)
                return
            
            # Branch 1: include v
            _bbs(candidates & ~(1 << v) & ~adj_mask[v], size + 1)
            
            # Branch 2: exclude v
            _bbs(candidates & ~(1 << v), size)
        
        _bbs((1 << n) - 1, 0)
        return best
    else:
        # For larger n, use a greedy approximation as upper bound
        # This is conservative - we'll verify with a different method if needed
        # Actually, let's use a different approach for larger n
        # We'll use a simple greedy algorithm to find a large independent set
        # and use that as a lower bound, and use Turan's theorem as upper bound
        
        # Greedy independent set
        remaining = set(range(n))
        ind_set = []
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            v = -1
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            ind_set.append(v)
            remaining -= adj[v]
            remaining.discard(v)
        
        # This gives a lower bound on alpha
        # For our purposes, we need to know if alpha < 10
        # If the greedy finds 10, alpha >= 10
        # If greedy finds < 10, we need to check more carefully
        # For now, return the greedy result as a conservative estimate
        # (if greedy finds 10, definitely alpha >= 10; if < 10, might still be >= 10)
        # Actually this is not safe. Let me use a different approach.
        
        # Use a simple exact algorithm with pruning
        # Since we only care if alpha >= 10, we can do a bounded search
        
        def _has_independent_set_of_size(adj, n, target):
            """Check if there's an independent set of size >= target."""
            if target <= 0:
                return True
            if n < target:
                return False
            
            # If target is small, use a simple backtracking
            # Choose vertices one by one
            def _backtrack(remaining, count):
                if count >= target:
                    return True
                if len(remaining) < target - count:
                    return False
                
                # Pick a vertex to branch on
                # Use the one with minimum degree for better pruning
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                
                # Option 1: include v
                new_remaining = remaining - adj[v] - {v}
                if _backtrack(new_remaining, count + 1):
                    return True
                
                # Option 2: exclude v
                new_remaining = remaining - {v}
                if _backtrack(new_remaining, count):
                    return True
                
                return False
            
            return _backtrack(set(range(n)), 0)
        
        # Check if alpha >= 10
        if _has_independent_set_of_size(adj, n, 10):
            return 10  # At least 10
        else:
            # Alpha < 10, find the exact value
            # Binary search or just compute
            for k in range(9, 0, -1):
                if _has_independent_set_of_size(adj, n, k):
                    return k
            return 0


def _try_add_vertex(adj, n, new_vertex_id):
    """Try to add a new vertex connected to a subset of existing vertices.
    Return (new_adj, new_n) if successful, or None."""
    # The new vertex must not create a triangle.
    # If we connect new vertex to u and v, then u and v must not be adjacent.
    # So the neighbors of the new vertex must form an independent set.
    
    # Try to find a large independent set among existing vertices
    # to connect to (this helps reduce alpha of the whole graph)
    
    # For simplicity, try connecting to a random independent set
    # or try all subsets up to some size
    
    # Actually, let's try a specific strategy:
    # Connect the new vertex to vertices that form an independent set
    # This way we don't create triangles.
    
    # Find an independent set greedily
    remaining = set(range(n))
    ind_set = []
    while remaining:
        # Pick vertex with minimum degree in remaining
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        ind_set.append(v)
        remaining -= adj[v]
        remaining.discard(v)
    
    # Try different sizes of independent sets to connect to
    # Start with smaller sets to be safe
    for size in range(min(5, len(ind_set)), -1, -1):
        # Try random subsets of the independent set
        for _ in range(20):
            if size == 0:
                neighbors = set()
            else:
                neighbors = set(random.sample(ind_set, size))
            
            # Create new adjacency
            new_adj = [set(s) for s in adj]
            new_adj.append(set(neighbors))
            for v in neighbors:
                new_adj[v].add(n)
            
            # Check triangle-free
            # New vertex connected to neighbors; need neighbors to be independent
            # which they are by construction
            # But also need to check that adding edges doesn't create triangles
            # involving existing edges - but we only added edges from new vertex
            # so the only new triangles would involve the new vertex
            # and we ensured neighbors are independent, so no triangles
            
            new_n = n + 1
            return new_adj, new_n
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with the Andrásfai graph And(9) on 26 vertices
    # Or start smaller and grow
    
    # Let's start with a small triangle-free graph and grow it
    # Initial: empty graph on 9 vertices (alpha = 9, triangle-free)
    # But we need alpha < 10, so 9 vertices with no edges has alpha = 9, OK
    
    # Actually, let's start with a known good graph and try to improve
    
    best_n = 0
    best_adj = None
    best_alpha = float('inf')
    
    # Start with the Andrásfai graph
    n = 26
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Verify
    assert _is_triangle_free(adj, n), "Initial graph not triangle-free"
    alpha = _max_independent_set_size(adj, n)
    print(f"Initial: n={n}, alpha={alpha}", file=sys.stderr)
    
    if alpha < 10:
        best_n = n
        best_adj = [set(s) for s in adj]
        best_alpha = alpha
    
    # Try to grow the graph by adding vertices
    current_adj = [set(s) for s in adj]
    current_n = n
    
    # Local search: try to reduce alpha by adding edges, then grow
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        # Try to add a vertex
        result = _try_add_vertex(current_adj, current_n, current_n)
        if result:
            new_adj, new_n = result
            # Check triangle-free
            if _is_triangle_free(new_adj, new_n):
                new_alpha = _max_independent_set_size(new_adj, new_n)
                if new_alpha < 10:
                    current_adj = new_adj
                    current_n = new_n
                    if current_n > best_n:
                        best_n = current_n
                        best_adj = [set(s) for s in current_adj]
                        best_alpha = new_alpha
                        print(f"Improved: n={current_n}, alpha={new_alpha}", file=sys.stderr)
        
        # Try to add edges to reduce alpha (while maintaining triangle-free)
        if time.time() > deadline - 5:
            break
        
        # Find pairs of non-adjacent vertices that can be connected
        # without creating a triangle
        # u and v can be connected if they have no common neighbors
        for _ in range(50):
            u = random.randint(0, current_n - 1)
            v = random.randint(0, current_n - 1)
            if u == v or v in current_adj[u]:
                continue
            
            # Check if u and v have common neighbors
            common = current_adj[u] & current_adj[v]
            if not common:
                # Safe to add edge
                current_adj[u].add(v)
                current_adj[v].add(u)
        
        # Check if alpha decreased
        if time.time() > deadline - 5:
            break
        
        new_alpha = _max_independent_set_size(current_adj, current_n)
        if new_alpha < best_alpha and current_n == best_n:
            best_alpha = new_alpha
            best_adj = [set(s) for s in current_adj]
        elif current_n > best_n and new_alpha < 10:
            best_n = current_n
            best_adj = [set(s) for s in current_adj]
            best_alpha = new_alpha
            print(f"Improved: n={current_n}, alpha={new_alpha}", file=sys.stderr)
    
    # Return the best graph found
    if best_adj is None:
        # Fallback to initial graph
        n = 26
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i].add(j)
                    adj[j].add(i)
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if j in adj[i]]
        return {'n': n, 'edges': edges}
    
    n = best_n
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if j in best_adj[i]]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END

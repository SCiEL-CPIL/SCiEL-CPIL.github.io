"""Find a graph on up to 22 vertices with no regular induced subgraph of size >= 6."""
import time
import random
import itertools
from collections import defaultdict

# Deadline guard: evaluator kills after 150s, leave safety margin
DEADLINE = time.time() + 120  # 120s wall-clock limit, use 115s for safety
SAFETY_MARGIN = 5  # seconds before limit to stop


def _is_regular_induced(adj, subset):
    """Check if the induced subgraph on 'subset' is regular.
    adj is a list of bitmasks (one per vertex) representing adjacency.
    """
    k = len(subset)
    if k < 6:
        return False
    # Compute degree of each vertex within the induced subgraph
    degs = []
    for v in subset:
        d = 0
        for u in subset:
            if u != v and (adj[v] >> u) & 1:
                d += 1
        degs.append(d)
    # Regular means all degrees equal
    if all(d == degs[0] for d in degs):
        return True
    return False


def _count_violations_fast(adj, n):
    """Count number of regular induced subgraphs of size >= 6.
    Returns (count, list_of_violating_subsets) for small counts.
    Uses bitmasks for speed.
    """
    violations = []
    # Check all subsets of size 6 to n
    # For n up to 22, C(22,6)=74613 which is manageable
    # But we need to check all sizes 6..n. Let's be smart.
    
    # Precompute adjacency as list of ints (bitmasks)
    # adj[v] has bit u set if edge (v,u) exists
    
    # For each subset of size exactly 6 (smallest), check regularity
    # Then for larger subsets, we can potentially prune, but let's just check all
    
    # Actually, for n=20, number of subsets of size 6 is C(20,6)=38760
    # For size 7: C(20,7)=77520, size 8: 125970, etc.
    # Total for n=20: sum_{k=6}^{20} C(20,k) which is large (~2^20 - sum_{k=0}^5 C(20,k))
    # That's about 1,048,576 - 21700 = ~1M subsets. Too slow in pure Python.
    
    # Better approach: only check subsets of size 6 first. If none found, 
    # we still need to check larger ones. But actually a regular subgraph of size 
    # > 6 contains regular induced subgraphs? No, that's not true in general.
    # A 7-vertex regular graph (e.g., 3-regular on 7 vertices) does NOT necessarily 
    # contain a 6-vertex regular induced subgraph.
    
    # So we must check all sizes. Let's use a different strategy:
    # For each subset of size k >= 6, check if it's regular.
    # To speed up: iterate over all subsets using bit manipulation.
    
    count = 0
    # Iterate over all subsets of size >= 6
    # For efficiency, iterate by size
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Check if induced subgraph is regular
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and (adj[v] >> u) & 1:
                        d += 1
                degs.append(d)
            if all(d == degs[0] for d in degs):
                count += 1
                if len(violations) < 10:
                    violations.append(subset)
                if count > 100:  # Early exit for very bad graphs
                    return count, violations
    return count, violations


def _count_violations_smart(adj, n):
    """Faster violation counting using bitmask operations."""
    count = 0
    violations = []
    
    # For each subset of size k >= 6, check regularity
    # Use bitmasks to compute degrees faster
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            degs = [0] * k
            for i, v in enumerate(subset):
                # Count neighbors of v within subset
                mask = 0
                for j, u in enumerate(subset):
                    if j != i:
                        mask |= (1 << u)
                # Count bits in (adj[v] & mask)
                degs[i] = bin(adj[v] & mask).count('1')
            if all(d == degs[0] for d in degs):
                count += 1
                if len(violations) < 5:
                    violations.append(subset)
                if count > 50:
                    return count, violations
    return count, violations


def _local_search(adj, n, max_iters=5000):
    """Local search: flip edges to reduce violations.
    Returns (adj, violations_count).
    """
    current_violations, current_subs = _count_violations_smart(adj, n)
    if current_violations == 0:
        return adj, 0
    
    best_adj = adj[:]
    best_count = current_violations
    
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Find a violating subset and try to fix it
        if not current_subs:
            break
        
        subset = current_subs[0]
        # Try flipping edges within or incident to this subset
        # Strategy: for each edge (u,v) where u,v in subset or one in subset,
        # try flipping it and see if violations decrease
        
        improved = False
        # Try all edges among vertices in subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                u, v = subset[i], subset[j]
                # Try flipping edge (u,v)
                old_bit = (adj[u] >> v) & 1
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_count, new_subs = _count_violations_smart(adj, n)
                
                if new_count < best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    improved = True
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                if new_count == 0:
                    return adj, 0
                if time.time() > DEADLINE - SAFETY_MARGIN:
                    return best_adj, best_count
        
        if not improved:
            # Try flipping edges incident to subset vertices
            for u in subset:
                for v in range(n):
                    if v in subset:
                        continue
                    old_bit = (adj[u] >> v) & 1
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_count, new_subs = _count_violations_smart(adj, n)
                    
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj[:]
                        improved = True
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    if new_count == 0:
                        return adj, 0
                    if time.time() > DEADLINE - SAFETY_MARGIN:
                        return best_adj, best_count
            
            if not improved:
                # Random restart within local search
                # Randomly flip some edges
                for _ in range(3):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                current_violations, current_subs = _count_violations_smart(adj, n)
                if current_violations < best_count:
                    best_count = current_violations
                    best_adj = adj[:]
    
    return best_adj, best_count


def construct():
    """Construct a graph with no regular induced subgraph of size >= 6.
    Maximize n.
    """
    start_time = time.time()
    
    # Try to build graphs of increasing size
    # Start with n=10 and work up
    
    best_n = 5
    best_edges = []
    
    # Try random initialization + local search for each n
    for n in range(10, 23):
        if time.time() > DEADLINE - 10:
            break
        
        found = False
        # Multiple random restarts
        max_restarts = 50 if n <= 15 else 30 if n <= 18 else 20
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            # Initialize with random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Check initial violations
            violations, subs = _count_violations_smart(adj, n)
            
            if violations == 0:
                found = True
                break
            
            # Local search
            adj, final_violations = _local_search(adj, n, max_iters=2000)
            
            if final_violations == 0:
                found = True
                break
        
        if found:
            best_n = n
            # Extract edges
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            best_edges = edges
        else:
            # Didn't find a valid graph for this n, try next
            continue
    
    # Verify the result
    if best_n >= 6 and best_edges:
        # Build adj from best_edges and verify
        adj = [0] * best_n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        violations, _ = _count_violations_smart(adj, best_n)
        if violations != 0:
            # Fall back to smaller n
            # Try n=6 with empty graph (always valid since no subset of size 6+... wait, n=6 has exactly one subset of size 6)
            # Empty graph on 6 vertices: all degrees 0, so it IS regular. Invalid.
            # We need to find a valid graph.
            # Let's try a known construction or just return a safe small graph.
            
            # For n=5, any graph works (no subset of size 6+)
            best_n = 5
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}

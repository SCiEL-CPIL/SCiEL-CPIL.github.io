<<<<<<< SEARCH
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
    # Total for n=20: sum_{k=6}^{20} C(20,k) which is large (~2^20 - sum_{k=0}^{5} C(20,k))
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
=======
"""Find a graph on up to 22 vertices with no regular induced subgraph of size >= 6."""
import time
import random
import numpy as np

# Deadline guard: evaluator kills after 150s, leave safety margin
DEADLINE = time.time() + 120
SAFETY_MARGIN = 5


def _count_violations_np(adj_matrix, n):
    """Count regular induced subgraphs of size >= 6 using NumPy vectorization.
    adj_matrix is an n x n numpy array (0/1).
    Returns (count, list of violating subsets as tuples).
    """
    import itertools
    
    count = 0
    violations = []
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            idx = list(subset)
            # Extract induced subgraph: k x k
            sub = adj_matrix[np.ix_(idx, idx)]
            # Degrees: row sums (diagonal is 0 in simple graph)
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                count += 1
                if len(violations) < 20:
                    violations.append(tuple(subset))
                if count > 200:
                    return count, violations
    return count, violations


def _count_violations_np_fast(adj_matrix, n, max_check=50000):
    """Faster: only check size-6 subsets first, then size-7, etc.
    Stop early if too many violations found.
    """
    import itertools
    
    count = 0
    violations = []
    checked = 0
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            checked += 1
            if checked > max_check:
                return count, violations
            
            idx = list(subset)
            sub = adj_matrix[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if np.all(degs == degs[0]):
                count += 1
                if len(violations) < 20:
                    violations.append(tuple(subset))
                if count > 100:
                    return count, violations
    return count, violations


def _local_search_np(adj_matrix, n, max_iters=3000):
    """Local search on numpy adjacency matrix. Flip edges to reduce violations."""
    current_count, current_subs = _count_violations_np_fast(adj_matrix, n)
    if current_count == 0:
        return adj_matrix, 0
    
    best_adj = adj_matrix.copy()
    best_count = current_count
    
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        if not current_subs:
            break
        
        # Pick a random violating subset
        subset = random.choice(current_subs)
        
        improved = False
        
        # Try flipping each edge within the subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                u, v = subset[i], subset[j]
                adj_matrix[u, v] = 1 - adj_matrix[u, v]
                adj_matrix[v, u] = adj_matrix[u, v]
                
                new_count, new_subs = _count_violations_np_fast(adj_matrix, n)
                
                if new_count < best_count:
                    best_count = new_count
                    best_adj = adj_matrix.copy()
                    improved = True
                else:
                    # Revert
                    adj_matrix[u, v] = 1 - adj_matrix[u, v]
                    adj_matrix[v, u] = adj_matrix[u, v]
                
                if new_count == 0:
                    return adj_matrix, 0
                if time.time() > DEADLINE - SAFETY_MARGIN:
                    return best_adj, best_count
        
        if not improved:
            # Try flipping edges from subset vertices to outside
            for u in subset:
                for v in range(n):
                    if v in subset:
                        continue
                    adj_matrix[u, v] = 1 - adj_matrix[u, v]
                    adj_matrix[v, u] = adj_matrix[u, v]
                    
                    new_count, new_subs = _count_violations_np_fast(adj_matrix, n)
                    
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj_matrix.copy()
                        improved = True
                    else:
                        adj_matrix[u, v] = 1 - adj_matrix[u, v]
                        adj_matrix[v, u] = adj_matrix[u, v]
                    
                    if new_count == 0:
                        return adj_matrix, 0
                    if time.time() > DEADLINE - SAFETY_MARGIN:
                        return best_adj, best_count
            
            if not improved:
                # Random perturbation
                for _ in range(5):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        adj_matrix[u, v] = 1 - adj_matrix[u, v]
                        adj_matrix[v, u] = adj_matrix[u, v]
                current_count, current_subs = _count_violations_np_fast(adj_matrix, n)
                if current_count < best_count:
                    best_count = current_count
                    best_adj = adj_matrix.copy()
    
    return best_adj, best_count
>>>>>>> REPLACE


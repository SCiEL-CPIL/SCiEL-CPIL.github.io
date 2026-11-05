import time
import random
from itertools import combinations

def construct():
    """
    Construct a graph on n vertices with no regular induced subgraph of size >= 6.
    Maximizes n.
    """
    deadline = time.time() + 115  # Safety margin
    rng = random.Random(42)
    
    best_n = 0
    best_edges = []
    
    # We know n=20 is possible. We aim for that or higher.
    # Let's try to find a valid graph for n=20 first. If we succeed, we try 21.
    # If we fail for 20, we might fall back to lower n, but 20 is the target.
    
    def _is_valid(adj_mask, n):
        """Check if the graph has no regular induced subgraph of size >= 6."""
        # We only need to check subsets of size 6. 
        # If a subset of size k > 6 is regular, it contains regular subsets of size 6? 
        # No, a regular graph of size k doesn't necessarily have regular induced subgraphs of size 6.
        # However, if a subset S of size k is regular, then every vertex in S has degree k-1 or 0 (if complete/empty) or some d.
        # Actually, the definition is: "No set of 6 or more vertices may induce a regular subgraph".
        # This means for ANY subset S with |S| >= 6, the induced subgraph G[S] must NOT be regular.
        # It is NOT sufficient to only check size 6.
        # BUT, if G[S] is regular for |S|=k, it is a violation.
        # Checking all subsets of size >= 6 is expensive.
        # However, usually, if you break all size 6 regularities, you might still have size 7, etc.
        # Let's optimize: 
        # 1. Check all subsets of size 6. If any is regular, invalid.
        # 2. If no size 6 is regular, check size 7, etc.
        # In practice, random graphs are unlikely to have large regular induced subgraphs if they don't have small ones.
        # But to be safe, we must check all sizes.
        
        # Efficient check for a specific subset mask:
        def _is_regular_mask(mask):
            verts = [i for i in range(n) if (mask >> i) & 1]
            k = len(verts)
            if k < 6:
                return False
            # Calculate degree of first vertex in subset
            v0 = verts[0]
            deg0 = (adj_mask[v0] & mask).bit_count()
            # Check all others
            for v in verts[1:]:
                deg = (adj_mask[v] & mask).bit_count()
                if deg != deg0:
                    return False
            return True

        # Check all subsets of size 6
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(mask):
                return False
        
        # If we passed size 6, check larger sizes?
        # The probability of a random graph having a regular induced subgraph of size 7+ without having one of size 6 is low,
        # but not zero. To be rigorous, we should check.
        # However, for the sake of speed in the search loop, we might rely on size 6 initially,
        # and do a full check only when we think we have a candidate.
        
        # For now, let's just say "valid" if no size 6 is regular, and do a full check in the final verification.
        return True

    def _full_check(adj_mask, n):
        """Full rigorous check for all subsets of size >= 6."""
        def _is_regular_mask(mask):
            verts = [i for i in range(n) if (mask >> i) & 1]
            k = len(verts)
            if k < 6:
                return False
            v0 = verts[0]
            deg0 = (adj_mask[v0] & mask).bit_count()
            for v in verts[1:]:
                deg = (adj_mask[v] & mask).bit_count()
                if deg != deg0:
                    return False
            return True

        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if _is_regular_mask(mask):
                    return False
        return True

    def _find_violation(adj_mask, n):
        """Find one regular induced subgraph of size >= 6. Returns (mask, size) or (0, 0)."""
        # Prioritize size 6
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            verts = [i for i in range(n) if (mask >> i) & 1]
            # Check regularity
            v0 = verts[0]
            deg0 = (adj_mask[v0] & mask).bit_count()
            is_reg = True
            for v in verts[1:]:
                if (adj_mask[v] & mask).bit_count() != deg0:
                    is_reg = False
                    break
            if is_reg:
                return mask, 6
        
        # If no size 6, try size 7
        for comb in combinations(range(n), 7):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            verts = [i for i in range(n) if (mask >> i) & 1]
            v0 = verts[0]
            deg0 = (adj_mask[v0] & mask).bit_count()
            is_reg = True
            for v in verts[1:]:
                if (adj_mask[v] & mask).bit_count() != deg0:
                    is_reg = False
                    break
            if is_reg:
                return mask, 7
        
        # ... up to n? 
        # For efficiency, if we don't find 6 or 7, we assume it's likely valid, 
        # but we will do a full check before accepting.
        return 0, 0

    def _random_init(n, p=0.5):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def _local_search(adj_mask, n, max_steps=5000):
        """
        Local search to eliminate regular induced subgraphs.
        """
        adj_mask = adj_mask[:]
        steps = 0
        while steps < max_steps and time.time() < deadline:
            steps += 1
            mask, size = _find_violation(adj_mask, n)
            if mask == 0:
                # No small violations found. Do a full check.
                if _full_check(adj_mask, n):
                    return adj_mask, True
                # If full check fails, _find_violation missed something larger?
                # Unlikely if we check 6 and 7. Let's break and restart if this happens.
                break
            
            # Break the regularity by flipping an edge inside the subset
            verts = [i for i in range(n) if (mask >> i) & 1]
            # Pick a random pair in verts
            u, v = rng.sample(verts, 2)
            # Flip edge (u, v)
            if adj_mask[u] & (1 << v):
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask, False

    # Try n=20
    target_n = 20
    found = False
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        p = 0.5
        # Maybe vary p slightly
        p = 0.5 + rng.uniform(-0.1, 0.1)
        
        adj_mask = _random_init(target_n, p)
        adj_mask, valid = _local_search(adj_mask, target_n, max_steps=2000)
        
        if valid:
            best_n = target_n
            # Construct edge list
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj_mask[i] & (1 << j):
                        edges.append([i, j])
            found = True
            break
            
        # If we have time, try again with different p or restart
    
    # If we didn't find n=20, try n=19, etc? 
    # The task says "Maximize n". If we fail 20, we should try lower.
    # But 20 is known to exist. If our search is too weak, we might fail.
    # Let's add a fallback to smaller n if 20 fails.
    
    if not found:
        for n in range(19, 5, -1):
            if time.time() > deadline - 5:
                break
            for restart in range(10):
                if time.time() > deadline - 5:
                    break
                adj_mask = _random_init(n, 0.5)
                adj_mask, valid = _local_search(adj_mask, n, max_steps=1000)
                if valid:
                    best_n = n
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj_mask[i] & (1 << j):
                                edges.append([i, j])
                    found = True
                    break
            if found:
                break

    if not found:
        # Last resort: return a small valid graph
        # n=5 is always valid (no subset of size >= 6)
        best_n = 5
        edges = []
        for i in range(5):
            for j in range(i+1, 5):
                 if rng.random() < 0.5:
                     edges.append([i, j])

    return {'n': best_n, 'edges': edges}

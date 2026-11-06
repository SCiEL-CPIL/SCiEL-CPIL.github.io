import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s

    def get_subset_masks(n, k):
        """Precompute masks for subsets of size k."""
        masks = []
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks.append(mask)
        return masks

    def check_mask_regularity(adj, mask, n):
        """Check if the induced subgraph defined by mask is regular."""
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        if not verts:
            return False, []
            
        first_deg = (adj[verts[0]] & mask).bit_count()
        is_reg = True
        for v in verts[1:]:
            d = (adj[v] & mask).bit_count()
            if d != first_deg:
                is_reg = False
                break
        
        if is_reg:
            return True, verts
        return False, verts

    def find_violation_sampled(adj, n, masks_by_size, sample_size=50):
        """
        Find a violation by sampling subsets.
        Returns (mask, k, verts) or (None, None, None).
        """
        # Check size 6 first
        if 6 in masks_by_size and masks_by_size[6]:
            masks = masks_by_size[6]
            # Sample random masks
            for _ in range(sample_size):
                mask = random.choice(masks)
                is_reg, verts = check_mask_regularity(adj, mask, n)
                if is_reg:
                    return mask, 6, verts
        
        # Check size 7
        if 7 in masks_by_size and masks_by_size[7]:
            masks = masks_by_size[7]
            for _ in range(sample_size):
                mask = random.choice(masks)
                is_reg, verts = check_mask_regularity(adj, mask, n)
                if is_reg:
                    return mask, 7, verts
                    
        # Check size 8
        if 8 in masks_by_size and masks_by_size[8]:
            masks = masks_by_size[8]
            for _ in range(sample_size):
                mask = random.choice(masks)
                is_reg, verts = check_mask_regularity(adj, mask, n)
                if is_reg:
                    return mask, 8, verts
                    
        return None, None, None

    def find_violation_exhaustive(adj, n, masks_by_size):
        """
        Find any violation exhaustively.
        Returns (mask, k, verts) or (None, None, None).
        """
        for k in range(6, n + 1):
            if k not in masks_by_size:
                continue
            masks = masks_by_size[k]
            for mask in masks:
                is_reg, verts = check_mask_regularity(adj, mask, n)
                if is_reg:
                    return mask, k, verts
        return None, None, None

    def search(n, time_limit):
        """
        Local search for a valid graph on n vertices.
        """
        t_start = time.time()
        
        # Precompute masks for sizes 6, 7, 8
        # We don't precompute all sizes to save memory/time, we just check 6,7,8 in sampling
        # For exhaustive check, we need all. But we will only do exhaustive check rarely.
        masks_by_size = {}
        for k in [6, 7, 8]:
            if k <= n:
                masks_by_size[k] = get_subset_masks(n, k)
        
        # For exhaustive check, we generate masks on the fly or precompute if needed
        # Let's precompute all for n=21 as it's only ~2M masks total, but storing them takes memory.
        # 2M ints * 28 bytes ~ 56MB. Acceptable.
        # However, generating them takes time.
        # Let's just generate them for the exhaustive check when needed.
        
        while time.time() - t_start < time_limit:
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            for step in range(5000):
                if time.time() - t_start > time_limit:
                    break
                
                # Sampled violation check
                mask, k, verts = find_violation_sampled(adj, n, masks_by_size, sample_size=20)
                
                if mask is not None:
                    # Fix violation
                    if len(verts) >= 2:
                        u, v = random.sample(verts, 2)
                    else:
                        u, v = random.sample(range(n), 2)
                    
                    # Flip edge
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # No sampled violation found.
                    # Do an exhaustive check periodically
                    if step % 10 == 0:
                        # Generate masks for all sizes 6..n for exhaustive check
                        # This is expensive, so we only do it if we've been "clean" for a while
                        # To save time, we can just check size 6 and 7 exhaustively, and sample 8+
                        # But for correctness, we need full check.
                        
                        # Let's try a faster exhaustive check for size 6 only first.
                        # If no size 6 violation, check size 7.
                        
                        # We'll use a helper that does exhaustive check for specific sizes
                        violation_found = False
                        for k_check in range(6, n + 1):
                            if k_check not in masks_by_size:
                                # Generate masks for this size
                                masks_by_size[k_check] = get_subset_masks(n, k_check)
                            
                            masks = masks_by_size[k_check]
                            for m in masks:
                                is_reg, v_list = check_mask_regularity(adj, m, n)
                                if is_reg:
                                    # Found violation
                                    mask, k, verts = m, k_check, v_list
                                    violation_found = True
                                    break
                            if violation_found:
                                break
                        
                        if violation_found:
                            # Fix it
                            if len(verts) >= 2:
                                u, v = random.sample(verts, 2)
                            else:
                                u, v = random.sample(range(n), 2)
                            
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                        else:
                            # No violation found in exhaustive check!
                            # Graph is valid
                            return adj
                        
                        # If we found a violation and fixed it, continue loop
                        # If we didn't find one, we returned.
                        
                # Add noise occasionally
                if step % 50 == 0:
                    u2, v2 = random.sample(range(n), 2)
                    if adj[u2] & (1 << v2):
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                    else:
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)

        return None

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try n=21 first
    if time.time() < deadline - 50:
        adj = search(21, time_limit=100)
        if adj:
            return {'n': 21, 'edges': adj_to_edges(adj, 21)}
            
    # Try n=20
    remaining = deadline - time.time()
    if remaining > 5:
        adj = search(20, time_limit=remaining - 5)
        if adj:
            return {'n': 20, 'edges': adj_to_edges(adj, 20)}

    # Fallback
    return {'n': 5, 'edges': []}

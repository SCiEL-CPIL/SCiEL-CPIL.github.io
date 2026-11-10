import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def check_regular(mask, adj_masks):
        """Check if the induced subgraph on 'mask' is regular."""
        m = mask
        if m == 0:
            return False
        first_deg = None
        count = 0
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj_masks[v] & mask).bit_count()
            if first_deg is None:
                first_deg = deg
            elif deg != first_deg:
                return False
            count += 1
            m ^= lsb
        return count >= 6

    def full_check(adj_masks, n):
        """Check all subsets of size >= 6. Returns (is_valid, violating_mask, violating_size)"""
        vertices = list(range(n))
        for k in range(6, n + 1):
            for subset in itertools.combinations(vertices, k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if check_regular(mask, adj_masks):
                    return False, mask, k
        return True, None, None

    def count_violations_size6(adj_masks, n):
        """Count number of size-6 regular induced subgraphs."""
        count = 0
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if check_regular(mask, adj_masks):
                count += 1
        return count

    def find_violation_size6(adj_masks, n):
        """Find one size-6 regular induced subgraph. Returns mask or None."""
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if check_regular(mask, adj_masks):
                return mask
        return None

    def flip_edge(adj_masks, edges, u, v):
        """Flip edge between u and v, updating adj_masks and edges list."""
        if adj_masks[u] & (1 << v):
            # Remove edge
            adj_masks[u] ^= (1 << v)
            adj_masks[v] ^= (1 << u)
            edges = [e for e in edges if not ((e[0]==u and e[1]==v) or (e[0]==v and e[1]==u))]
        else:
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
        return edges

    best_result = {'n': 5, 'edges': []}
    
    # Strategy: Try to find a valid graph for n=20, then 19, etc.
    # Use local search with multiple restarts and targeted edge flips.
    
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        found = False
        num_restarts = 30 if n <= 16 else 15
        for attempt in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random graph - try different densities
            if attempt < num_restarts // 3:
                p = 0.5
            elif attempt < 2 * num_restarts // 3:
                p = 0.4
            else:
                p = 0.6
                
            adj_masks = [0] * n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        edges.append([i, j])
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local search: iteratively fix violations
            max_iters = 800
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Find a violation of size 6 (cheapest to check)
                viol_mask = find_violation_size6(adj_masks, n)
                
                if viol_mask is None:
                    # No size-6 violations, do full check
                    is_valid, _, _ = full_check(adj_masks, n)
                    if is_valid:
                        best_result = {'n': n, 'edges': edges}
                        found = True
                        break
                    else:
                        # Has larger violation, try to fix by random edge flips
                        # Pick a random pair and flip
                        for _ in range(20):
                            u = random.randint(0, n-1)
                            v = random.randint(0, n-1)
                            if u != v:
                                edges = flip_edge(adj_masks, edges, u, v)
                        break
                else:
                    # Extract vertices from violation mask
                    verts = []
                    m = viol_mask
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        verts.append(v)
                        m ^= lsb
                    
                    # Try flipping several random edges within the violating subgraph
                    # to break the regularity
                    improved = False
                    for _ in range(5):
                        v1, v2 = random.sample(verts, 2)
                        old_edges = edges
                        edges = flip_edge(adj_masks, edges, v1, v2)
                        
                        # Check if this helped (fewer violations or same)
                        # For efficiency, just continue and check at next iteration
                        improved = True
                        break
                    
                    if not improved:
                        # Try flipping an edge between violating and non-violating vertices
                        non_viol = [v for v in range(n) if not (viol_mask & (1 << v))]
                        if non_viol:
                            v1 = random.choice(verts)
                            v2 = random.choice(non_viol)
                            edges = flip_edge(adj_masks, edges, v1, v2)
            
            if found:
                break
        
        if found:
            break
    
    # Fallback: try a smaller n if above failed
    if best_result['n'] < 6 and time.time() < deadline:
        for n in [6, 7, 8, 9, 10]:
            if time.time() > deadline - 2:
                break
            # For small n, brute force is fast
            for attempt in range(50):
                adj_masks = [0] * n
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            edges.append([i, j])
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                is_valid, _, _ = full_check(adj_masks, n)
                if is_valid:
                    best_result = {'n': n, 'edges': edges}
                    break
            if best_result['n'] >= n:
                break
    
    return best_result

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def find_regular_subgraph(adj_masks, n):
        """Returns a mask of a regular induced subgraph of size >= 6, or 0 if none."""
        # Check size 6
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            v0 = subset[0]
            d0 = (adj_masks[v0] & mask).bit_count()
            is_reg = True
            for v in subset[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
            if is_reg:
                return mask
        
        # Check size 7
        for subset in combinations(range(n), 7):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            v0 = subset[0]
            d0 = (adj_masks[v0] & mask).bit_count()
            is_reg = True
            for v in subset[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
            if is_reg:
                return mask

        # Check larger sizes
        for size in range(8, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                v0 = subset[0]
                d0 = (adj_masks[v0] & mask).bit_count()
                is_reg = True
                for v in subset[1:]:
                    if (adj_masks[v] & mask).bit_count() != d0:
                        is_reg = False
                        break
                if is_reg:
                    return mask
        return 0

    def get_edges_from_masks(adj_masks, n):
        edges = []
        for i in range(n):
            m = adj_masks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Try to find graph for n
    def solve_n(n):
        if time.time() > deadline:
            return None
        
        # Random restarts
        for restart in range(100):
            if time.time() > deadline:
                return None
            
            # Init random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    return None
                
                mask = find_regular_subgraph(adj_masks, n)
                if mask == 0:
                    return get_edges_from_masks(adj_masks, n)
                
                # Break the regular subgraph
                # Pick a random vertex in mask and flip an edge to a vertex in mask
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                v = random.choice(verts)
                # Pick another vertex in mask
                u = random.choice(verts)
                if u == v:
                    continue
                    
                # Flip edge (u, v)
                if adj_masks[v] & (1 << u):
                    adj_masks[v] ^= (1 << u)
                    adj_masks[u] ^= (1 << v)
                else:
                    adj_masks[v] |= (1 << u)
                    adj_masks[u] |= (1 << v)
        
        return None

    # Try n=22, 21, 20
    for n in [22, 21, 20, 19]:
        if time.time() > deadline:
            break
        edges = solve_n(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    return {'n': 5, 'edges': []}

import time
import itertools
import random

# EVOLVE-BLOCK-START
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 320  # 320s safety margin before 330s kill
    
    def time_left():
        return deadline - time.time()
    
    # Build Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    n = q
    # adj_mask[i] is a bitmask of neighbors of vertex i
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    current_n = n
    
    def has_clique_4(mask):
        """Check if the subgraph induced by vertices in mask contains a K4."""
        # Get list of vertices in mask
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            verts.append(idx)
            m ^= lb
        
        if len(verts) < 4:
            return False
        
        # Check all 4-combinations
        for i in range(len(verts)):
            vi = verts[i]
            for j in range(i+1, len(verts)):
                vj = verts[j]
                if not (adj_mask[vi] & (1 << vj)):
                    continue
                for k in range(j+1, len(verts)):
                    vk = verts[k]
                    if not (adj_mask[vi] & (1 << vk)) or not (adj_mask[vj] & (1 << vk)):
                        continue
                    for l in range(k+1, len(verts)):
                        vl = verts[l]
                        if (adj_mask[vi] & (1 << vl)) and (adj_mask[vj] & (1 << vl)) and (adj_mask[vk] & (1 << vl)):
                            return True
        return False
    
    def has_independent_4(mask):
        """Check if the subgraph induced by vertices in mask contains an independent set of size 4."""
        verts = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            verts.append(idx)
            m ^= lb
        
        if len(verts) < 4:
            return False
        
        # Check all 4-combinations for independence
        for i in range(len(verts)):
            vi = verts[i]
            for j in range(i+1, len(verts)):
                vj = verts[j]
                if adj_mask[vi] & (1 << vj):
                    continue
                for k in range(j+1, len(verts)):
                    vk = verts[k]
                    if (adj_mask[vi] & (1 << vk)) or (adj_mask[vj] & (1 << vk)):
                        continue
                    for l in range(k+1, len(verts)):
                        vl = verts[l]
                        if not (adj_mask[vi] & (1 << vl)) and not (adj_mask[vj] & (1 << vl)) and not (adj_mask[vk] & (1 << vl)):
                            return True
        return False
    
    def is_valid_neighbor_mask(n, neighbor_mask):
        """Check if adding vertex n with given neighbor connections is valid."""
        # neighbor_mask must not contain K4
        if has_clique_4(neighbor_mask):
            return False
        # non-neighbors must not contain independent 4-set
        all_mask = (1 << n) - 1
        non_neighbor_mask = all_mask & ~neighbor_mask
        if has_independent_4(non_neighbor_mask):
            return False
        return True
    
    def try_add_vertex(n):
        """Try to find a valid neighbor mask for a new vertex n."""
        all_mask = (1 << n) - 1
        
        # Quick checks
        if is_valid_neighbor_mask(n, all_mask):
            return all_mask
        if is_valid_neighbor_mask(n, 0):
            return 0
        
        # Smart random search: start from a random subset and flip bits
        random.seed(42 + n)
        
        # Try multiple random starting points
        for attempt in range(50000):
            if time_left() < 10:
                break
            
            # Random mask
            mask = 0
            m = all_mask
            while m:
                lb = m & (-m)
                if random.random() < 0.5:
                    mask |= lb
                m ^= lb
            
            if is_valid_neighbor_mask(n, mask):
                return mask
            
            # Try local search: flip one bit at a time
            if attempt % 10 == 0:
                current = mask
                for _ in range(20):
                    if time_left() < 10:
                        break
                    # Pick a random vertex to flip
                    v = random.randint(0, n-1)
                    bit = 1 << v
                    new_mask = current ^ bit
                    if is_valid_neighbor_mask(n, new_mask):
                        return new_mask
                    # Only keep if it's "better" (harder to evaluate, just random walk)
                    if random.random() < 0.5:
                        current = new_mask
        
        # Systematic: try masks with specific density
        for target_density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for attempt in range(10000):
                if time_left() < 10:
                    break
                mask = 0
                m = all_mask
                while m:
                    lb = m & (-m)
                    if random.random() < target_density:
                        mask |= lb
                    m ^= lb
                if is_valid_neighbor_mask(n, mask):
                    return mask
        
        return None
    
    # Try to extend the graph one vertex at a time
    max_target = 45
    for target_n in range(current_n + 1, max_target + 1):
        if time_left() < 15:
            break
        
        neighbor_mask = try_add_vertex(current_n)
        if neighbor_mask is not None:
            # Add the new vertex
            new_idx = current_n
            adj_mask.append(neighbor_mask)
            # Update adjacency for existing vertices
            m = neighbor_mask
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                adj_mask[v] |= (1 << new_idx)
                m ^= lb
            current_n += 1
        else:
            break
    
    # Build edge list
    edges = []
    for i in range(current_n):
        m = adj_mask[i]
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            m ^= lb
    
    return current_n, edges

# EVOLVE-BLOCK-END

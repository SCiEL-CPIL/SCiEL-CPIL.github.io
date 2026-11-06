"""Improved program for R(K2,2,2) maximization via Paley construction and local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _paley_mask(n):
    """
    Generate the Paley graph mask for n = q (prime power).
    Vertices 0..n-1. Edge (i,j) if (i-j) is a quadratic residue mod n.
    Note: Paley graphs are defined for q = p^k where p is prime.
    For n=29, 29 is prime.
    """
    if n < 2:
        return [0] * n
    
    # Check if n is a prime power? For simplicity, assume prime for now or handle small cases.
    # We will specifically target n=29, 28, etc.
    # For n=29, we need quadratic residues mod 29.
    
    residues = set()
    for i in range(1, n):
        residues.add((i * i) % n)
        
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i == j:
                continue
            diff = (i - j) % n
            if diff in residues:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _has_k222(masks, n):
    """
    Fast check for K2,2,2 in the graph defined by masks.
    Returns True if a K2,2,2 is found.
    """
    if n < 6:
        return False
        
    # Precompute non-edge masks
    all_ones = (1 << n) - 1
    non_edges = [0] * n
    for i in range(n):
        # Non-edges are bits not in masks[i], excluding i itself
        non_edges[i] = (all_ones ^ masks[i]) & ~(1 << i)
        
    # List of non-edges (i, j) with i < j
    ne_list = []
    for i in range(n):
        # Iterate bits in non_edges[i] that are > i
        m = non_edges[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                ne_list.append((i, j))
            m ^= lsb
            
    if len(ne_list) < 3:
        return False

    len_ne = len(ne_list)
    
    # Iterate over pairs of non-edges
    for i1 in range(len_ne):
        a, b = ne_list[i1]
        # Common neighbors of a and b in G (edges)
        # Vertices connected to both a and b
        common_ab = masks[a] & masks[b]
        common_ab &= ~(1 << a) & ~(1 << b)
        
        for i2 in range(i1 + 1, len_ne):
            c, d = ne_list[i2]
            # Disjoint check
            if c == a or c == b or d == a or d == b:
                continue
            
            # c and d must be in common_ab
            if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                continue
            
            # Candidates for e, f must be in common_ab AND common_cd
            common_cd = masks[c] & masks[d]
            candidates = common_ab & common_cd
            candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
            
            if candidates == 0:
                continue
            
            # Check if there is a non-edge (e, f) within candidates
            # Iterate through bits in candidates
            m = candidates
            while m:
                lsb = m & -m
                e = lsb.bit_length() - 1
                # Check if any other vertex f in candidates is a non-neighbor of e
                # non_edges[e] & (candidates ^ (1<<e))
                if non_edges[e] & (candidates ^ lsb):
                    return True
                m ^= lsb
                
    return False

def _is_valid(masks, n):
    """Check if graph is K2,2,2 free."""
    return not _has_k222(masks, n)

def _adj_to_mask(adj, n):
    """Convert numpy boolean adjacency matrix to list of int bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        masks[i] = mask
    return masks

def _mask_to_adj(masks, n):
    """Convert list of bitmasks to numpy boolean adjacency matrix."""
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if (masks[i] >> j) & 1:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _local_search(masks, n, time_limit):
    """
    Attempt to remove K2,2,2s by flipping edges.
    Returns (success, masks)
    """
    start = time.time()
    current_masks = masks[:]
    
    # We don't have a good cost function that is fast.
    # Strategy: Randomly flip edges that are part of many "almost" K2,2,2s?
    # Or just random flips and check validity?
    # Since _has_k222 is boolean, we can use a "repair" loop:
    # 1. Find a K2,2,2.
    # 2. Flip one of its non-edges to an edge (break the K2,2,2).
    # 3. Repeat.
    
    # However, flipping might create new K2,2,2s.
    # Let's try a simple hill-climbing on the number of K2,2,2s?
    # We need a counter. Let's implement a fast counter for the search phase.
    
    def count_k222(masks):
        if n < 6: return 0
        all_ones = (1 << n) - 1
        non_edges = [0] * n
        for i in range(n):
            non_edges[i] = (all_ones ^ masks[i]) & ~(1 << i)
            
        ne_list = []
        for i in range(n):
            m = non_edges[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    ne_list.append((i, j))
                m ^= lsb
                
        if len(ne_list) < 3:
            return 0

        count = 0
        len_ne = len(ne_list)
        
        for i1 in range(len_ne):
            a, b = ne_list[i1]
            common_ab = masks[a] & masks[b]
            common_ab &= ~(1 << a) & ~(1 << b)
            
            for i2 in range(i1 + 1, len_ne):
                c, d = ne_list[i2]
                if c == a or c == b or d == a or d == b:
                    continue
                if not ((common_ab >> c) & 1) or not ((common_ab >> d) & 1):
                    continue
                common_cd = masks[c] & masks[d]
                candidates = common_ab & common_cd
                candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                
                if candidates == 0:
                    continue
                # Count non-edges in candidates
                # For each e in candidates, check non_edges[e] & (candidates ^ (1<<e))
                # This counts each non-edge twice? No, we just want existence or count.
                # For cost, let's just return 1 if any found to keep it fast?
                # No, count is better for gradient.
                # But O(M^2 * |S|) is slow.
                # Let's stick to boolean for now and use random restarts from Paley.
                return 1 # Early exit for speed in search? 
        return 0

    # Since count is slow/boolean, we rely on the quality of the initial guess (Paley).
    # If Paley 29 is valid, we return it.
    # If not, we try to fix it.
    
    # Let's just check if the input masks are valid. If not, we do a few random flips and re-check.
    # This is a weak search but better than nothing if Paley fails.
    
    for step in range(1000):
        if time.time() > time_limit:
            break
        if _is_valid(current_masks, n):
            return True, current_masks
            
        # Pick random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j: continue
        if i > j: i, j = j, i
        
        bit = 1 << j
        if current_masks[i] & bit:
            current_masks[i] &= ~bit
            current_masks[j] &= ~(1 << i)
        else:
            current_masks[i] |= bit
            current_masks[j] |= (1 << i)
            
    return False, current_masks

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_n = 1
    best_colors = [0]
    
    # Try n=29, 28, ... down to 6
    # We prioritize known good candidates like Paley graphs.
    
    for target_n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
            
        n = target_n
        found = False
        
        # 1. Try Paley construction if n is prime (or prime power, but we only handle prime easily here)
        # Check if n is prime
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
            
        if is_prime(n):
            masks = _paley_mask(n)
            if _is_valid(masks, n):
                found = True
                # Store best
                best_n = n
                best_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        best_colors.append(1 if (masks[i] >> j) & 1 else 0)
                break # Found a valid one for this n, which is the max we tried so far
            
        # 2. If Paley failed or n not prime, try random + local search
        # But random is unlikely to work for n=29.
        # We can try to extend a smaller valid graph?
        # For now, if Paley didn't work, we skip to next n unless we have time for random search.
        
        # Let's add a few random restarts for non-prime n or if Paley failed
        if not found and (n < 20 or time.time() < start_time + 30):
             # Only do random search for smaller n or if we have plenty of time
             for restart in range(10):
                if time.time() > deadline: break
                masks = [0] * n
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            masks[i] |= (1 << j)
                            masks[j] |= (1 << i)
                # Quick check
                if _is_valid(masks, n):
                    found = True
                    best_n = n
                    best_colors = []
                    for i in range(n):
                        for j in range(i+1, n):
                            best_colors.append(1 if (masks[i] >> j) & 1 else 0)
                    break
                # Local search
                success, masks = _local_search(masks, n, deadline)
                if success:
                    found = True
                    best_n = n
                    best_colors = []
                    for i in range(n):
                        for j in range(i+1, n):
                            best_colors.append(1 if (masks[i] >> j) & 1 else 0)
                    break

        if found:
            break

    # If we didn't find anything (unlikely), return a small valid graph
    if best_n == 1:
        best_n = 5
        best_colors = [0] * (5*4//2)
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

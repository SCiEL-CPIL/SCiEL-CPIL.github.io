"""Construct a large (5,5)-free graph via randomized local search with time budget."""

import time
import random

def _bits(mask):
    """Yield indices of set bits in mask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb

def _find_k5(n, adj_masks):
    """Find a K5 in the graph. Returns tuple of 5 vertices or None."""
    for v in range(n):
        nbr_mask = adj_masks[v]
        nodes = list(_bits(nbr_mask))
        k = len(nodes)
        if k < 4:
            continue
        for i in range(k):
            a = nodes[i]
            for j in range(i + 1, k):
                b = nodes[j]
                common = nbr_mask & adj_masks[a] & adj_masks[b]
                c_mask = common
                while c_mask:
                    lsb = c_mask & -c_mask
                    c = lsb.bit_length() - 1
                    c_mask ^= lsb
                    d_mask = common & adj_masks[c]
                    if d_mask:
                        lsb_d = d_mask & -d_mask
                        d = lsb_d.bit_length() - 1
                        return (v, a, b, c, d)
    return None

def _find_empty5(n, adj_masks):
    """Find an independent set of size 5. Returns tuple of 5 vertices or None."""
    all_mask = (1 << n) - 1
    for v in range(n):
        non_nbr_mask = all_mask & ~(adj_masks[v] | (1 << v))
        nodes = list(_bits(non_nbr_mask))
        k = len(nodes)
        if k < 4:
            continue
        for i in range(k):
            a = nodes[i]
            for j in range(i + 1, k):
                b = nodes[j]
                common_non = non_nbr_mask & ~adj_masks[a] & ~adj_masks[b]
                c_mask = common_non
                while c_mask:
                    lsb = c_mask & -c_mask
                    c = lsb.bit_length() - 1
                    c_mask ^= lsb
                    d_mask = common_non & ~adj_masks[c] & ~(1 << c)
                    if d_mask:
                        lsb_d = d_mask & -d_mask
                        d = lsb_d.bit_length() - 1
                        return (v, a, b, c, d)
    return None

def _count_violations(n, adj_masks, cap=50):
    """Count K5s and Empty5s, capped."""
    count = 0
    for v in range(n):
        nbr_mask = adj_masks[v]
        nodes = list(_bits(nbr_mask))
        k = len(nodes)
        if k >= 4:
            for i in range(k):
                a = nodes[i]
                for j in range(i + 1, k):
                    b = nodes[j]
                    common = nbr_mask & adj_masks[a] & adj_masks[b]
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        c = lsb.bit_length() - 1
                        c_mask ^= lsb
                        if common & adj_masks[c]:
                            count += 1
                            if count >= cap:
                                return count
    all_mask = (1 << n) - 1
    for v in range(n):
        non_nbr_mask = all_mask & ~(adj_masks[v] | (1 << v))
        nodes = list(_bits(non_nbr_mask))
        k = len(nodes)
        if k >= 4:
            for i in range(k):
                a = nodes[i]
                for j in range(i + 1, k):
                    b = nodes[j]
                    common_non = non_nbr_mask & ~adj_masks[a] & ~adj_masks[b]
                    c_mask = common_non
                    while c_mask:
                        lsb = c_mask & -c_mask
                        c = lsb.bit_length() - 1
                        c_mask ^= lsb
                        d_mask = common_non & ~adj_masks[c] & ~(1 << c)
                        if d_mask:
                            count += 1
                            if count >= cap:
                                return count
    return count

def construct():
    deadline = time.time() + 290
    
    for n in range(44, 37, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            current_violations = _count_violations(n, adj_masks)
            
            if current_violations == 0:
                edges = []
                for i in range(n):
                    for j in _bits(adj_masks[i]):
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
                
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Find a violation and try to fix it
                k5 = _find_k5(n, adj_masks)
                if k5:
                    # Try to remove one edge from the K5
                    # Pick a random edge from the K5
                    verts = list(k5)
                    random.shuffle(verts)
                    i, j = verts[0], verts[1]
                    
                    # Flip
                    adj_masks[i] ^= (1 << j)
                    adj_masks[j] ^= (1 << i)
                    
                    new_violations = _count_violations(n, adj_masks)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        if current_violations == 0:
                            edges = []
                            for i in range(n):
                                for j in _bits(adj_masks[i]):
                                    if i < j:
                                        edges.append((i, j))
                            return (n, edges)
                    else:
                        # Revert
                        adj_masks[i] ^= (1 << j)
                        adj_masks[j] ^= (1 << i)
                        continue

                e5 = _find_empty5(n, adj_masks)
                if e5:
                    # Try to add one edge from the Empty5
                    verts = list(e5)
                    random.shuffle(verts)
                    i, j = verts[0], verts[1]
                    
                    # Flip
                    adj_masks[i] ^= (1 << j)
                    adj_masks[j] ^= (1 << i)
                    
                    new_violations = _count_violations(n, adj_masks)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        if current_violations == 0:
                            edges = []
                            for i in range(n):
                                for j in _bits(adj_masks[i]):
                                    if i < j:
                                        edges.append((i, j))
                            return (n, edges)
                    else:
                        # Revert
                        adj_masks[i] ^= (1 << j)
                        adj_masks[j] ^= (1 << i)
                        continue
                        
                # If no violation found by _find, but count > 0, something is wrong or we missed it.
                # Break and restart
                break

    # Fallback
    n = 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return n, edges

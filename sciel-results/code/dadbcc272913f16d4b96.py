"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 320.0  # safety margin before 330s kill

def _paley_edges_37():
    """Construct the Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return edges

def _edges_to_masks(n, edges):
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _masks_to_edges(masks, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if masks[i] & (1 << j):
                edges.append((i, j))
    return edges

def _find_violation(n, masks):
    """Find a K5 or independent 5-set. Returns (tuple_of_5, is_clique) or None."""
    for i in range(n - 4):
        mi = masks[i]
        for j in range(i + 1, n - 3):
            mj = masks[j]
            e_ij = (mi >> j) & 1
            for k in range(j + 1, n - 2):
                mk = masks[k]
                e_ik = (mi >> k) & 1
                e_jk = (mj >> k) & 1
                for l in range(k + 1, n - 1):
                    ml = masks[l]
                    e_il = (mi >> l) & 1
                    e_jl = (mj >> l) & 1
                    e_kl = (mk >> l) & 1
                    for m in range(l + 1, n):
                        mm = masks[m]
                        e_im = (mi >> m) & 1
                        e_jm = (mj >> m) & 1
                        e_km = (mk >> m) & 1
                        e_lm = (ml >> m) & 1
                        # Check clique: all 10 edges present
                        if (e_ij and e_ik and e_il and e_im and
                            e_jk and e_jl and e_jm and
                            e_kl and e_km and e_lm):
                            return (i, j, k, l, m), True
                        # Check independent: all 10 edges absent
                        if (not e_ij and not e_ik and not e_il and not e_im and
                            not e_jk and not e_jl and not e_jm and
                            not e_kl and not e_km and not e_lm):
                            return (i, j, k, l, m), False
    return None

def _local_search(n, masks, deadline):
    """Local search: flip edges in violating 5-sets until valid or time runs out."""
    step = 0
    while time.time() < deadline - 5.0:
        viol = _find_violation(n, masks)
        if viol is None:
            return True
        (v0, v1, v2, v3, v4), is_clique = viol
        vertices = [v0, v1, v2, v3, v4]
        # Pick a random edge in this 5-set to flip
        a, b = random.sample(range(5), 2)
        u, v = vertices[a], vertices[b]
        if masks[u] & (1 << v):
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        else:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        step += 1
        if step > 100000:
            break
    return False

def _try_find_graph(n, deadline, max_restarts=50):
    """Try to find a valid (5,5)-free graph on n vertices using local search."""
    paley_edges = _paley_edges_37()
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            return None
        
        # Initialize: start with Paley 37 + random edges for extra vertices
        masks = [0] * n
        for u, v in paley_edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        
        # Add random edges for vertices 37..n-1
        for i in range(37, n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
            # Connect to first 37 vertices randomly
            for j in range(37):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        # Local search to repair
        if _local_search(n, masks, deadline):
            return _masks_to_edges(masks, n)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Paley 37
    best_n = 37
    best_edges = _paley_edges_37()
    
    # Try to find larger graphs, starting from the largest known
    for target_n in [42, 41, 40, 39, 38]:
        if target_n <= best_n:
            break
        if time.time() > deadline - 30.0:
            break
        
        edges = _try_find_graph(target_n, deadline)
        if edges is not None:
            best_n = target_n
            best_edges = edges
            if best_n == 42:
                break
    
    return best_n, best_edges
# EVOLVE-BLOCK-END

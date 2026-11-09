import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin

def _has_K4(n, bits):
    """
    Returns a list of edges involved in a K4, or None if no K4 exists.
    bits: list of ints, bits[i] is adjacency bitset for vertex i.
    """
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = bits[a] & bits[b]
            # We need to find c, d in common such that c < d and edge(c, d)
            # To be fast, we just need to find ONE such pair.
            # Iterate c
            c = (common & ~((1 << (b + 1)) - 1)).bit_length() - 1
            # Actually, let's just iterate properly but break early
            # Mask out a and b and lower vertices
            common &= ~((1 << (b + 1)) - 1)
            while common:
                c_bit = common & -common
                c = c_bit.bit_length() - 1
                common ^= c_bit
                # Check if c has any neighbor d in common with d > c
                d_mask = bits[c] & common
                if d_mask:
                    # Found K4: a, b, c, d
                    # Return one edge to break, e.g., (a, b)
                    return [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
    return None

def _has_Ind6(n, bits):
    """
    Returns a list of pairs in an independent set of size 6, or None.
    """
    all_mask = (1 << n) - 1
    non_adj = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    
    for a in range(n):
        for b in range(a + 1, n):
            if (non_adj[a] >> b) & 1:
                common = non_adj[a] & non_adj[b]
                common &= ~((1 << (b + 1)) - 1)
                while common:
                    c_bit = common & -common
                    c = c_bit.bit_length() - 1
                    common ^= c_bit
                    common2 = common & non_adj[c]
                    common2 &= ~((1 << (c + 1)) - 1)
                    while common2:
                        d_bit = common2 & -common2
                        d = d_bit.bit_length() - 1
                        common2 ^= d_bit
                        common3 = common2 & non_adj[d]
                        common3 &= ~((1 << (d + 1)) - 1)
                        while common3:
                            e_bit = common3 & -common3
                            e = e_bit.bit_length() - 1
                            common3 ^= e_bit
                            common4 = common3 & non_adj[e]
                            common4 &= ~((1 << (e + 1)) - 1)
                            if common4:
                                # Found Ind6: a, b, c, d, e, f
                                f_bit = common4 & -common4
                                f = f_bit.bit_length() - 1
                                # Return pairs in the independent set to potentially add edges
                                # We want to add an edge. Let's return the set of vertices.
                                return [a, b, c, d, e, f]
    return None

def _is_valid(n, bits):
    return _has_K4(n, bits) is None and _has_Ind6(n, bits) is None

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

def _local_search(n, bits, max_steps=5000):
    """
    Iteratively repair the graph by removing K4 edges and adding Ind6 edges.
    """
    rng = random.Random(42)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 2:
            break
            
        k4 = _has_K4(n, bits)
        if k4:
            # Pick a random edge from the K4 to remove
            u, v = rng.choice(k4)
            bits[u] &= ~(1 << v)
            bits[v] &= ~(1 << u)
            continue
            
        ind6 = _has_Ind6(n, bits)
        if ind6:
            # Pick a random pair from the Ind6 to add
            # To avoid creating a K4, we should be careful, but local search will fix it
            pair = rng.sample(ind6, 2)
            u, v = pair
            if u > v:
                u, v = v, u
            # Check if adding this edge creates a K4? 
            # If we add (u,v), it creates a K4 if they have a common neighbor w such that u-w, v-w, and u-v exist.
            # Actually, if u and v are non-adjacent, adding (u,v) creates a triangle. 
            # It creates a K4 if there is a w adjacent to both u and v, AND u-v was the missing edge.
            # Wait, K4 requires 4 vertices. If we add (u,v), we form a K4 if there is a w such that 
            # u-w, v-w, and (u,v) is the 4th edge? No, K4 is {u,v,w,x}.
            # If we add (u,v), we create a K4 if there exist w, x such that 
            # all edges among {u,v,w,x} exist.
            # This is complex to check on the fly. Just add it and let the next K4 check remove something.
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
            continue
            
        # If no K4 and no Ind6, we are valid
        return bits
        
    return bits

def _search_for_n(n):
    """
    Try to find a valid graph on n vertices using local search from multiple random starts.
    """
    if time.time() > _DEADLINE - 5:
        return None
        
    # Try a few random starts
    for attempt in range(5):
        if time.time() > _DEADLINE - 2:
            break
            
        rng = random.Random(attempt * 1000 + n)
        bits = [0] * n
        # Start with a random graph, density ~0.5
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
                    
        # Run local search
        # Limit time per attempt
        start_t = time.time()
        # We don't know how many steps, so just run until valid or time out
        # We'll modify _local_search to take a time limit
        while time.time() - start_t < 5.0:
             k4 = _has_K4(n, bits)
             if k4:
                 u, v = rng.choice(k4)
                 bits[u] &= ~(1 << v)
                 bits[v] &= ~(1 << u)
                 continue
             ind6 = _has_Ind6(n, bits)
             if ind6:
                 u, v = rng.sample(ind6, 2)
                 if u > v:
                     u, v = v, u
                 bits[u] |= (1 << v)
                 bits[v] |= (1 << u)
                 continue
             # Valid!
             return bits
             
    return None

def construct():
    t0 = time.time()
    
    # Baseline: Paley 17 is known to be valid? 
    # Let's verify Paley 17 quickly.
    # Actually, let's just try to find n=35, 34, ... down to 18.
    # If we fail, fall back to 17.
    
    best_n = 17
    best_bits = None
    
    # Verify Paley 17 as a fallback
    def _paley_graph(p):
        if p < 2 or p % 4 != 1:
            return None
        squares = {x * x % p for x in range(1, p)}
        bits = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        return bits
        
    bits17 = _paley_graph(17)
    if bits17 and _is_valid(17, bits17):
        best_bits = bits17
    else:
        # Fallback to a simple valid graph if Paley fails (unlikely)
        # C5 is valid for K3, Ind3? No, task is K4, Ind6.
        # Empty graph on 5 vertices is valid (no K4, no Ind6).
        # Complete graph on 3 vertices is valid.
        # Let's just use n=1 if nothing works.
        best_bits = [0]
        best_n = 1

    # Try to improve
    # We know n=35 is possible. Let's aim high.
    for n in range(35, 17, -1):
        if time.time() > _DEADLINE - 10:
            break
            
        bits = _search_for_n(n)
        if bits is not None:
            if _is_valid(n, bits):
                best_n = n
                best_bits = bits
                break # Found a valid graph for this n, no need to check lower n
                
    # Construct output
    if best_bits is None:
        return {'n': 1, 'edges': []}
        
    edges = _edges_from_bits(best_n, best_bits)
    return {'n': best_n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")

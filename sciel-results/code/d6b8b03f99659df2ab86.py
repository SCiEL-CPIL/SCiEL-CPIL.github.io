<<<<<<< SEARCH
def _try_random_search(n_target, k, g, time_limit):
    """Attempt to find a k-regular graph of girth >= g on n_target vertices
    via random generation + local repair. Returns (n, edges) or None."""
    # This is a heuristic; for n=243+ it's very hard. We'll try small n.
    # Actually, let's just try to build via random 4-regular graphs and check.
    # For the purposes of this task, the LUW construction with d=5 gives
    # a valid graph. Let's check its size.
    return None
=======
def _try_random_search(n_target, k, g, time_limit):
    """Attempt to find a k-regular graph of girth >= g on n_target vertices
    via tensor product + edge switching. Returns (n, edges) or None."""
    # Try different cycle lengths for tensor product
    # Target: n_target vertices, 4-regular
    # Tensor product of C_L1 x C_L2 gives L1*L2 vertices
    # We want L1*L2 close to n_target
    best_result = None
    best_n = float('inf')
    
    for L1 in range(8, 40):
        for L2 in range(8, 40):
            n = L1 * L2
            if n > n_target * 1.2 or n < n_target * 0.8:
                continue
            if time.time() > time_limit - 5.0:
                return best_result
            
            _, edges = _build_tensor_product(L1, L2)
            
            # Verify initial graph
            if not _verify_girth(n, edges, g):
                # Try edge switching to fix
                result = _edge_swap_search(n, edges, g, time_limit)
                if result is not None:
                    rn, re = result
                    if _verify_girth(rn, re, g):
                        if rn < best_n:
                            best_n = rn
                            best_result = result
                        if rn <= n_target:
                            return result
            else:
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                if n <= n_target:
                    return (n, edges)
    
    return best_result
>>>>>>> REPLACE


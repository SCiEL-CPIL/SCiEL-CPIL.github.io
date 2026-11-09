def _has_independent_set_k(adj, adj_bits, n, k):
    """
    Exact check for independent set of size k.
    """
    # 1. Greedy check
    remaining = (1 << n) - 1
    count = 0
    while remaining:
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        count += 1
        if count >= k:
            return True
        # Remove v and its neighbors from remaining
        remaining &= ~adj_bits[v]
        remaining &= ~(1 << v)
        
    if count >= k:
        return True
        
    # 2. Exact check via complement clique
    full_mask = (1 << n) - 1
    comp_adj_bits = [0] * n
    for i in range(n):
        comp_adj_bits[i] = full_mask ^ (1 << i) ^ adj_bits[i]
        
    return _has_clique_k(comp_adj_bits, n, k)

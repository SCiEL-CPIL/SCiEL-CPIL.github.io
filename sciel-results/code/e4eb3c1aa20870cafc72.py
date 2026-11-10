<<<<<<< SEARCH
    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_k33_free(adj_mask):
                break
            # Find a violating triple
            viol = None
            for u, v, w in triples:
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    viol = (u, v, w, common)
                    break
            if viol is None:
                break
            u, v, w, common = viol
            # Remove a random edge incident to a common neighbor
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                # Try removing (u,c), (v,c), or (w,c)
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if is_k33_free(adj_mask):
                            removed = True
                            break
                        else:
                            # Restore
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                # Force remove one
                if common_list:
                    c = common_list[0]
                    a, b = u, c
                    adj_mask[a] &= ~bit(b)
                    adj_mask[b] &= ~bit(a)
=======
    def find_violation(adj_mask):
        """Find a violating triple using vectorized check. Returns (u,v,w,common_mask) or None."""
        adj_arr = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    adj_arr[i, j] = 1
                    adj_arr[j, i] = 1
        
        triples_arr = np.array(triples, dtype=np.int32)
        u_idx = triples_arr[:, 0]
        v_idx = triples_arr[:, 1]
        w_idx = triples_arr[:, 2]
        
        adj_u = adj_arr[u_idx]
        adj_v = adj_arr[v_idx]
        adj_w = adj_arr[w_idx]
        
        common = adj_u & adj_v & adj_w
        common_counts = common.sum(axis=1)
        
        violating = np.where(common_counts > 2)[0]
        if len(violating) == 0:
            return None
        
        # Pick a random violating triple
        idx = violating[random.randint(0, len(violating) - 1)]
        u, v, w = triples[idx]
        # Reconstruct common mask from adj_mask
        common_mask = adj_mask[u] & adj_mask[v] & adj_mask[w]
        return (u, v, w, common_mask)

    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(200):
            if time.time() > deadline:
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, common = viol
            # Remove a random edge incident to a common neighbor
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                # Try removing (u,c), (v,c), or (w,c)
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if is_k33_free(adj_mask):
                            removed = True
                            break
                        else:
                            # Restore
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                # Force remove one
                if common_list:
                    c = common_list[0]
                    a, b = u, c
                    adj_mask[a] &= ~bit(b)
                    adj_mask[b] &= ~bit(a)
>>>>>>> REPLACE


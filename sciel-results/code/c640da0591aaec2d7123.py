    def _calc_alpha(triples, n=16):
        # Precompute masks for triples
        triple_masks = [0] * len(triples)
        for i, t in enumerate(triples):
            m = 0
            for v in t:
                m |= (1 << v)
            triple_masks[i] = m
        
        best = 0
        # Iterate masks. To find max independent set, we can just check all masks.
        # Optimization: If a mask is independent, all its subsets are independent.
        # We want the largest.
        # We can iterate from (1<<n)-1 down to 0? No, that's slow.
        # Just iterate all, keep track of max size.
        # To speed up, if bin(mask).count('1') <= best: continue
        
        # Even faster: Use a recursive backtracking or just the simple loop with bit tricks.
        # For N=16, 65536 iterations is trivial. The overhead is in Python loops.
        # Let's use a list comprehension or similar? No, simple loop is fine.
        
        for mask in range(1 << n):
            size = bin(mask).count('1')
            if size <= best:
                continue
            
            # Check if mask contains any triple
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                best = size
                # If best == n, break (unlikely for hypergraphs)
                if best == n:
                    break
        return best
    
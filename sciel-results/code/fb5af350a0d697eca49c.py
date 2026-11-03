    def _local_search(v, k, t, blocks, tsubs, imap, deadline, rng):
        # Build coverage count or boolean array
        n = len(tsubs)
        covered = [False] * n
        for b in blocks:
            for c in itertools.combinations(b, t):
                covered[imap[c]] = True
        
        # Check full coverage
        if not all(covered):
             return blocks # Should not happen if greedy finished, but safe
            
        # Pass 1: Remove redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            i = 0
            while i < len(blocks):
                b = blocks[i]
                # Check if any t-subset of b is ONLY covered by b
                # Simpler: temporarily remove b, check if any uncovered appears
                # We need to know counts, not just bools, to check "only covered by b"
                # So we need counts.
                
                # Let's switch to counts for local search.
                break # Abort this simple version, need counts.
                
        # Re-implement with counts
        counts = np.zeros(n, dtype=np.int32)
        for b in blocks:
            for c in itertools.combinations(b, t):
                counts[imap[c]] += 1
        
        def is_covered():
            return np.all(counts > 0)
            
        # Pass 1: Remove redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            i = 0
            while i < len(blocks):
                b = blocks[i]
                # Decrement counts
                for c in itertools.combinations(b, t):
                    counts[imap[c]] -= 1
                if is_covered():
                    blocks.pop(i)
                    changed = True
                else:
                    # Restore
                    for c in itertools.combinations(b, t):
                        counts[imap[c]] += 1
                    i += 1

        # Pass 2: Add/Remove
        # ... (similar to before but using counts)
        
        return blocks
    
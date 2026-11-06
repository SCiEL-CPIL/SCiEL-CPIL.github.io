    def local_search(S, S_bits, counts, C2_bits, max_swaps=50):
        """Try swap-based local search."""
        S = list(S)
        S_bits = S_bits
        counts = counts.copy()
        C2_bits = C2_bits
        
        S_set = set(S)
        improved = True
        swap_count = 0
        
        while improved and swap_count < max_swaps:
            improved = False
            # Try swapping each element with a random candidate
            # Iterate over a copy of the list to avoid modification during iteration
            for x in list(S):
                if swap_count >= max_swaps:
                    break
                
                # Try removing x
                S.remove(x)
                S_set.discard(x)
                S_bits &= ~(1 << x)
                
                # Decrement counts for all pairs involving x
                for y in S:
                    s = x + y
                    counts[s] -= 1
                counts[2 * x] -= 1
                
                # Update C2_bits: clear bits where count dropped from 2 to 1
                # This is expensive to do incrementally. 
                # However, since we are doing local search, maybe we should just 
                # rebuild C2_bits if we want to be 100% safe, or accept conservative.
                # But if we accept conservative, we might miss improvements.
                # For now, let's just try to add. If we fail, we restore.
                
                # Try adding random candidates
                candidates = [c for c in range(1, N + 1) if c not in S_set]
                random.shuffle(candidates)
                
                added = False
                for c in candidates:
                    # Check if we can add c
                    # Note: C2_bits might be stale (conservative). 
                    # If we want to be more aggressive, we should check actual counts.
                    # Let's check actual counts to be safe and accurate.
                    valid = True
                    for y in S:
                        s = c + y
                        if counts[s] >= 2:
                            valid = False
                            break
                    if valid and counts[2 * c] >= 2:
                        valid = False
                    
                    if not valid:
                        continue
                        
                    # Add c
                    S.append(c)
                    S_set.add(c)
                    S_bits |= (1 << c)
                    for y in S:
                        s = c + y
                        counts[s] += 1
                        if counts[s] == 2:
                            C2_bits |= (1 << s)
                    
                    added = True
                    improved = True
                    swap_count += 1
                    break
                
                if not added:
                    # Restore x
                    S.append(x)
                    S_set.add(x)
                    S_bits |= (1 << x)
                    for y in S:
                        s = x + y
                        counts[s] += 1
                        if counts[s] == 2:
                            C2_bits |= (1 << s)
                    counts[2 * x] += 1
                    # Re-calculate C2_bits for correctness? 
                    # If we restored x, the state should be back to original.
                    # But C2_bits might have gained bits from failed attempts? 
                    # No, we only update C2_bits on successful add.
                    # On restore, we update counts. We should probably rebuild C2_bits 
                    # to ensure it matches counts, or just leave it conservative.
                    # Leaving it conservative is safe.
        
        return S, S_bits, counts, C2_bits

    def try_find_mate(A):
        row_masks = [0] * n
        col_masks = [0] * n
        used_pairs = set()
        B = [[0 for _ in range(n)] for _ in range(n)]
        
        # Randomize value order
        val_order = list(range(n))
        random.shuffle(val_order)
        
        def backtrack(r, c):
            if time.time() > deadline:
                return False
            
            if r == n:
                return True
            
            if c == n:
                return backtrack(r + 1, 0)
            
            for val in val_order:
                if row_masks[r] & (1 << val):
                    continue
                if col_masks[c] & (1 << val):
                    continue
                pair = (A[r][c], val)
                if pair in used_pairs:
                    continue
                
                row_masks[r] |= (1 << val)
                col_masks[c] |= (1 << val)
                used_pairs.add(pair)
                B[r][c] = val
                
                if backtrack(r, c + 1):
                    return True
                
                row_masks[r] ^= (1 << val)
                col_masks[c] ^= (1 << val)
                used_pairs.remove(pair)
                B[r][c] = 0
            
            return False
        
        if backtrack(0, 0):
            return B
        return None

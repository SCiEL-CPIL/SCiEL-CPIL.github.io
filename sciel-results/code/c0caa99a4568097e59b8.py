import time

def construct():
    n = 10
    # Standard Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin for 60s limit
    
    def find_next_square(existing_squares):
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        row_masks = [0] * n
        col_masks = [0] * n
        # pair_masks[i] corresponds to existing_squares[i]
        pair_masks = [0] * len(existing_squares)
        
        def solve():
            if time.time() - start_time > time_limit:
                return False
            
            # MRV Heuristic
            min_count = n + 1
            min_i = -1
            min_j = -1
            min_candidates = None
            
            for i in range(n):
                for j in range(n):
                    if L_new[i][j] != -1:
                        continue
                    
                    candidates = []
                    for v in range(n):
                        # Check row and col constraints
                        if (row_masks[i] & (1 << v)) or (col_masks[j] & (1 << v)):
                            continue
                        
                        # Check orthogonality constraints
                        valid = True
                        for idx in range(len(existing_squares)):
                            k = existing_squares[idx][i][j]
                            pair_bit = k * n + v
                            if pair_masks[idx] & (1 << pair_bit):
                                valid = False
                                break
                        
                        if valid:
                            candidates.append(v)
                    
                    if len(candidates) < min_count:
                        min_count = len(candidates)
                        min_i = i
                        min_j = j
                        min_candidates = candidates
                        
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            # Can't get better than 1, but ensure no 0s in remaining?
                            # No, if we find 1, we break out of j-loop, then i-loop.
                            # But there might be a 0 later.
                            # However, MRV usually picks the most constrained.
                            # If we break early, we might miss a 0.
                            # To be safe, don't break on 1, just continue to find 0s?
                            # Or break if we are confident?
                            # Let's just break if min_count == 1 to save time, 
                            # but if there is a 0 later, we will fail at that cell.
                            # Actually, if there is a 0 later, the current cell with 1 is not the most constrained.
                            # So we should NOT break on 1. We should only break if we find 0.
                            # But finding 0 returns False immediately.
                            # So we can break if min_count == 1?
                            # If there is a 0 elsewhere, we will find it.
                            # But if we break, we won't check elsewhere.
                            # So do NOT break on 1.
                            pass
                
                # Removed break on 1 to ensure we find 0s if they exist.
                # If min_count is 1, we continue to see if any is 0.
            
            if min_i == -1:
                return True # All cells filled
            
            # Try candidates in min_candidates
            # Shuffle? No, deterministic is fine.
            for v in min_candidates:
                L_new[min_i][min_j] = v
                row_masks[min_i] |= (1 << v)
                col_masks[min_j] |= (1 << v)
                
                for idx in range(len(existing_squares)):
                    k = existing_squares[idx][min_i][min_j]
                    pair_masks[idx] |= (1 << (k * n + v))
                
                if solve():
                    return True
                
                # Backtrack
                L_new[min_i][min_j] = -1
                row_masks[min_i] ^= (1 << v)
                col_masks[min_j] ^= (1 << v)
                for idx in range(len(existing_squares)):
                    k = existing_squares[idx][min_i][min_j]
                    pair_masks[idx] ^= (1 << (k * n + v))
            
            return False

        if solve():
            return L_new
        else:
            return None

    # Iteratively find more squares
    while True:
        if time.time() - start_time > time_limit:
            break
        next_sq = find_next_square(squares)
        if next_sq is None:
            break
        squares.append(next_sq)

    return squares

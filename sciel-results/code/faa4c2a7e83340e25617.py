import time

def construct():
    n = 10
    max_squares = 9
    start_time = time.time()
    time_limit = 85.0
    
    squares = []
    sq0 = [[(i+j) % n for j in range(n)] for i in range(n)]
    squares.append(sq0)
    
    def generate_square(existing_sqs):
        new_sq = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Symmetry breaking
        for j in range(n): new_sq[0][j] = j
        for i in range(n): new_sq[i][0] = i
        
        row_symbols = [set() for _ in range(n)]
        col_symbols = [set() for _ in range(n)]
        
        for i in range(n):
            row_symbols[i].add(i)
            for j in range(1, n):
                row_symbols[i].add(new_sq[i][j])
            for j in range(n):
                col_symbols[j].add(new_sq[i][j])
                
        seen_pairs = [set() for _ in range(len(existing_sqs))]
        for k, esq in enumerate(existing_sqs):
            for i in range(n):
                for j in range(n):
                    if new_sq[i][j] != -1:
                        seen_pairs[k].add((new_sq[i][j], esq[i][j]))
                        
        # MRV Backtracking
        def backtrack():
            if time.time() - start_time > time_limit:
                return False
                
            # Find empty cell with minimum remaining values
            min_count = n + 1
            min_i = -1
            min_j = -1
            min_candidates = []
            
            for i in range(1, n):
                for j in range(1, n):
                    if new_sq[i][j] != -1:
                        continue
                    # Calculate candidates
                    candidates = []
                    for s in range(n):
                        if s in row_symbols[i] or s in col_symbols[j]:
                            continue
                        # Check orthogonality
                        valid = True
                        for k in range(len(existing_sqs)):
                            pair = (s, existing_sqs[k][i][j])
                            if pair in seen_pairs[k]:
                                valid = False
                                break
                        if valid:
                            candidates.append(s)
                            
                    if len(candidates) == 0:
                        return False # Dead end
                    if len(candidates) < min_count:
                        min_count = len(candidates)
                        min_i, min_j = i, j
                        min_candidates = candidates
                        if min_count == 1:
                            break # Can't get better
                if min_count == 1:
                    break
            
            if min_i == -1:
                return True # All filled
                
            # Try candidates
            for s in min_candidates:
                new_sq[min_i][min_j] = s
                row_symbols[min_i].add(s)
                col_symbols[min_j].add(s)
                pairs_to_add = []
                for k in range(len(existing_sqs)):
                    pair = (s, existing_sqs[k][min_i][min_j])
                    seen_pairs[k].add(pair)
                    pairs_to_add.append((k, pair))
                    
                if backtrack():
                    return True
                    
                # Undo
                new_sq[min_i][min_j] = -1
                row_symbols[min_i].remove(s)
                col_symbols[min_j].remove(s)
                for k, pair in pairs_to_add:
                    seen_pairs[k].remove(pair)
                    
            return False
            
        if backtrack():
            return new_sq
        return None

    for _ in range(max_squares - 1):
        if time.time() - start_time > time_limit:
            break
        new_sq = generate_square(squares)
        if new_sq is None:
            break
        squares.append(new_sq)
        
    return squares

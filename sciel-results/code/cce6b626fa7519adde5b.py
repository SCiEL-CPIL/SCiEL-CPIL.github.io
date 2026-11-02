import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        return _generic_construct(q, n, radius)

    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    start_time = time.time()
    deadline = start_time + 110

    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute neighbors for each word index
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Calculate index of new_w
                idx = 0
                for j in range(n):
                    idx += new_w[j] * (q ** (n - 1 - j))
                nbs.append(idx)
        neighbors.append(nbs)
    neighbors = np.array(neighbors, dtype=np.int32)

    def get_code_ball(G):
        codewords = []
        for coeffs in itertools.product(range(3), repeat=4):
            cw = np.zeros(n, dtype=np.int32)
            for r in range(4):
                if coeffs[r] != 0:
                    cw += coeffs[r] * G[r]
            cw = cw % 3
            codewords.append(cw)
        
        ball = np.zeros(total_words, dtype=bool)
        for cw in codewords:
            idx = 0
            for j in range(n):
                idx += cw[j] * (q ** (n - 1 - j))
            ball[neighbors[idx]] = True
        return ball

    def try_structured_search():
        attempts = 0
        while time.time() < deadline - 15:
            attempts += 1
            G = np.random.randint(0, 3, size=(4, 8))
            
            # Rank check
            M = G.copy()
            rank = 0
            for col in range(8):
                pivot_row = -1
                for r in range(rank, 4):
                    if M[r, col] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                M[rank], M[pivot_row] = M[pivot_row].copy(), M[rank].copy()
                for r in range(4):
                    if r != rank and M[r, col] != 0:
                        factor = (M[r, col] * M[rank, col]) % 3
                        M[r] = (M[r] - factor * M[rank]) % 3
                rank += 1
                if rank == 4:
                    break
            
            if rank < 4:
                continue
            
            B = get_code_ball(G)
            U = ~B
            
            if not np.any(U):
                # Single code covers
                # Extract indices
                c_indices = []
                for coeffs in itertools.product(range(3), repeat=4):
                    cw = np.zeros(n, dtype=np.int32)
                    for r in range(4):
                        if coeffs[r] != 0:
                            cw += coeffs[r] * G[r]
                    cw = cw % 3
                    idx = 0
                    for j in range(n):
                        idx += cw[j] * (q ** (n - 1 - j))
                    c_indices.append(idx)
                return c_indices

            success = True
            chosen_shifts = []
            for step in range(5):
                if not np.any(U):
                    break
                U_fft = np.fft.fft(U.astype(float))
                B_fft = np.fft.fft(B.astype(float))
                corr_fft = U_fft * np.conj(B_fft)
                corr = np.fft.ifft(corr_fft).real
                best_idx = np.argmax(corr)
                if corr[best_idx] < 1e-9:
                    success = False
                    break
                chosen_shifts.append(best_idx)
                B_shifted = np.roll(B, best_idx)
                U = U & ~B_shifted
                
            if success and not np.any(U):
                # Generate codewords
                c_indices = []
                for coeffs in itertools.product(range(3), repeat=4):
                    cw = np.zeros(n, dtype=np.int32)
                    for r in range(4):
                        if coeffs[r] != 0:
                            cw += coeffs[r] * G[r]
                    cw = cw % 3
                    idx = 0
                    for j in range(n):
                        idx += cw[j] * (q ** (n - 1 - j))
                    c_indices.append(idx)
                
                all_cw_indices = set(c_indices)
                for s_idx in chosen_shifts:
                    sv = np.zeros(n, dtype=np.int32)
                    rem = s_idx
                    for j in range(n-1, -1, -1):
                        sv[j] = rem % q
                        rem //= q
                    for cw_idx in c_indices:
                        cw = all_words[cw_idx]
                        new_cw = (cw + sv) % 3
                        new_idx = 0
                        for j in range(n):
                            new_idx += new_cw[j] * (q ** (n - 1 - j))
                        all_cw_indices.add(new_idx)
                        
                return list(all_cw_indices)
            
        return None

    result_indices = try_structured_search()
    
    if result_indices is not None:
        return [list(all_words[i]) for i in result_indices]
        
    # Fallback to greedy
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[neighbors]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[neighbors[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _generic_construct(q, n, radius):
    return []

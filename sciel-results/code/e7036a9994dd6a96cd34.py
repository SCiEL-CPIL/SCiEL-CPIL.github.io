def _build_ball_matrix(q, n, radius):
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    B = np.zeros((total_words, total_words), dtype=bool)
    
    if radius == 1:
        for i in range(total_words):
            word_i = [(i // p) % q for p in powers]
            ball_indices = [i]
            for pos in range(n):
                base_val = word_i[pos]
                for new_val in range(q):
                    if new_val == base_val:
                        continue
                    neighbor_idx = i + (new_val - base_val) * powers[pos]
                    ball_indices.append(neighbor_idx)
            B[i, ball_indices] = True
    else:
        # General case: iterate all pairs? Or generate combinations?
        # For small n, q, this is fine.
        for i in range(total_words):
            word_i = [(i // p) % q for p in powers]
            for j in range(total_words):
                word_j = [(j // p) % q for p in powers]
                dist = sum(1 for k in range(n) if word_i[k] != word_j[k])
                if dist <= radius:
                    B[i, j] = True
                    
    return B

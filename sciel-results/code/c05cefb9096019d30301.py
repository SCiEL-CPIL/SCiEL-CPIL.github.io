import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    
    # 1. Generate the [17,8,6] linear code
    # Using the hardcoded generator matrix for the [17,9,5] QR code
    G_QR = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int)
    
    def gf2_null_space(A):
        m, n = A.shape
        A = A.copy()
        pivot_cols = []
        row = 0
        for col in range(n):
            if row >= m:
                break
            pivot = -1
            for i in range(row, m):
                if A[i, col] == 1:
                    pivot = i
                    break
            if pivot == -1:
                continue
            A[[row, pivot]] = A[[pivot, row]]
            for i in range(m):
                if i != row and A[i, col] == 1:
                    A[i] = (A[i] + A[row]) % 2
            pivot_cols.append(col)
            row += 1
        
        free_cols = [c for c in range(n) if c not in pivot_cols]
        num_free = len(free_cols)
        if num_free == 0:
            return np.array([]).reshape(0, n)
        
        basis = []
        for j, fc in enumerate(free_cols):
            v = np.zeros(n, dtype=int)
            v[fc] = 1
            for i, pc in enumerate(pivot_cols):
                val = 0
                for fj in free_cols:
                    if A[i, fj] == 1:
                        val ^= v[fj]
                v[pc] = val
            basis.append(v)
        
        return np.array(basis)

    G_dual = gf2_null_space(G_QR)
    
    words = []
    for mask in range(256):
        c = np.array([(mask >> i) & 1 for i in range(8)], dtype=int)
        w = c @ G_dual
        # Convert to int
        val = 0
        for i in range(17):
            if w[i] == 1:
                val |= (1 << (16 - i))
        words.append(val)
        
    words = list(set(words))
    
    def int_to_vec(w):
        return np.array([(w >> i) & 1 for i in range(16, -1, -1)], dtype=int)
        
    current_vecs = np.array([int_to_vec(w) for w in words])
    current_set = set(words)
    best_size = len(words)
    best_vecs = current_vecs.copy()
    best_set = current_set.copy()
    
    # Candidates: all other words
    all_words = list(range(2**17))
    others = [w for w in all_words if w not in current_set]
    
    # Try multiple random shuffles
    for _ in range(10):
        np.random.shuffle(others)
        curr_vecs = best_vecs.copy()
        curr_set = best_set.copy()
        
        for w in others:
            if w in curr_set:
                continue
            vec = int_to_vec(w)
            # Vectorized distance check
            dists = np.sum(curr_vecs != vec, axis=1)
            if np.min(dists) >= MIN_DIST:
                curr_set.add(w)
                # Append to curr_vecs
                curr_vecs = np.vstack([curr_vecs, vec])
        
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_vecs = curr_vecs
            best_set = curr_set
            if best_size >= 258:
                break
                
    return list(best_set)

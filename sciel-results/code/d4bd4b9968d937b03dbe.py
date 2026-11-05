import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: greedy for other parameters
        return _greedy_cover(q, n, radius, deadline)
    
    # Ternary Hamming code [7,4,3] - generator matrix
    # H is 3x7 over GF(3)
    # Columns of H are all non-zero vectors in GF(3)^3 (up to scalar), but for [7,4,3] we need 7 columns
    # Actually, the ternary [7,4,3] Hamming code has parity check matrix with columns being all non-zero vectors of GF(3)^3 / ~
    # GF(3)^3 has 27 vectors, 1 is zero, 26 non-zero. Up to scalar (×1, ×2), we have 13 equivalence classes.
    # But we need exactly 7 columns for [7,4,3].
    
    # Let me use the standard construction:
    # The ternary [7,4,3] Hamming code can be defined with parity check matrix:
    H = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 0, 1, 2, 0],
        [0, 0, 0, 1, 1, 1, 2]
    ], dtype=int)
    
    # Actually, let me use a known good parity check matrix for ternary [7,4,3]
    # The columns should be all distinct non-zero vectors in GF(3)^3 up to scalar multiplication
    # We need 7 columns. Let's pick 7 from the 13 equivalence classes.
    
    # Better approach: Use the known fact that K_3(7,1) = 186 can be achieved
    # by a specific construction involving the tetracode and its extensions.
    
    # The van Laarhoven et al. construction:
    # 1. Start with the tetracode (12 words of length 4 over GF(3))
    # 2. Extend to length 7 using specific coset structure
    
    # Tetracode: all words (a, b, a+b, a+2b) for a,b in GF(3)
    tetracode = set()
    for a in range(3):
        for b in range(3):
            w = (a, b, (a + b) % 3, (a + 2 * b) % 3)
            tetracode.add(w)
    
    # The 186-word code is constructed as follows:
    # It's a union of cosets of a subcode. Let me use a different approach:
    # Use the fact that the optimal code is a covering code that can be found
    # by starting from a good initial code and applying local search.
    
    # For speed, let me use a well-known initial construction:
    # The code from van Laarhoven et al. has 186 words.
    # A good starting point: take all words where the first 4 positions form a tetracode word
    # and the last 3 positions are chosen from a specific set.
    
    # Actually, let me use a more systematic approach:
    # 1. Generate an initial code using greedy set cover
    # 2. Apply aggressive local search to reach 186
    
    # Precompute neighbors for all 2187 words
    all_indices = list(range(2187))
    
    def word_to_idx(w):
        idx = 0
        for d in range(7):
            idx = idx * 3 + w[d]
        return idx
    
    def idx_to_word(idx):
        w = [0] * 7
        for d in range(6, -1, -1):
            w[d] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute neighbor lists (Hamming distance <= 1)
    # Each word has 1 + 7*2 = 15 neighbors
    neighbors = np.zeros((2187, 15), dtype=np.int32)
    for idx in range(2187):
        w = idx_to_word(idx)
        neighbors[idx, 0] = idx
        pos = 1
        for d in range(7):
            for val in range(3):
                if val == w[d]:
                    continue
                new_w = list(w)
                new_w[d] = val
                neighbors[idx, pos] = word_to_idx(tuple(new_w))
                pos += 1
    
    # Initial code: greedy set cover
    # Use a fast greedy approach
    uncovered = np.ones(2187, dtype=bool)
    current_code = []
    coverage = np.zeros(2187, dtype=np.int32)
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    for _ in range(250):  # Upper bound on iterations
        if not uncovered.any():
            break
        if time.time() > deadline - 10:
            break
        
        # Find word that covers most uncovered
        best_idx = -1
        best_count = 0
        for idx in range(2187):
            if coverage[idx] > 0:
                # Skip if already in code (approximation)
                pass
            cnt = uncovered[neighbors[idx]].sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        current_code.append(best_idx)
        coverage[neighbors[best_idx]] += 1
        uncovered[coverage == 0] = False
    
    best_code = current_code.copy()
    best_size = len(current_code)
    
    # Local search: deletion and swap
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Phase 1: Deletion
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                if np.min(coverage[neighbors[c]]) > 1:
                    coverage[neighbors[c]] -= 1
                    current_code.pop(i)
                    changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code.copy()
        
        # Phase 2: Swap - try replacing one codeword with another
        if len(current_code) > 186:
            # Try to find a swap that reduces size or maintains size
            for i in range(len(current_code)):
                if time.time() > deadline - 5:
                    break
                c_old = current_code[i]
                # Temporarily remove c_old
                coverage[neighbors[c_old]] -= 1
                uncovered_now = coverage == 0
                
                # Find best replacement
                best_new = -1
                best_gain = 0
                for j in range(2187):
                    if j == c_old:
                        continue
                    if coverage[j] > 0 and j in current_code:
                        continue
                    gain = uncovered_now[neighbors[j]].sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_new = j
                
                if best_new != -1 and best_gain > 0:
                    # Perform swap
                    current_code[i] = best_new
                    coverage[neighbors[best_new]] += 1
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code.copy()
                else:
                    # Revert
                    coverage[neighbors[c_old]] += 1
        
        # Phase 3: Kick if stuck
        if len(current_code) > 186 and random.random() < 0.3:
            # Remove 3-5 random codewords and repair
            k = random.randint(3, 5)
            if len(current_code) > k:
                remove_set = set(random.sample(range(len(current_code)), k))
                new_code = []
                for i, c in enumerate(current_code):
                    if i not in remove_set:
                        new_code.append(c)
                    else:
                        coverage[neighbors[c]] -= 1
                current_code = new_code
                
                # Repair
                while time.time() < deadline - 5:
                    uncov = np.where(coverage == 0)[0]
                    if len(uncov) == 0:
                        break
                    
                    best_idx = -1
                    best_count = 0
                    uncov_set = set(uncov.tolist())
                    for idx in range(2187):
                        if idx in current_code:
                            continue
                        cnt = sum(1 for nb in neighbors[idx] if nb in uncov_set)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    
                    if best_idx == -1 or best_count == 0:
                        break
                    
                    current_code.append(best_idx)
                    coverage[neighbors[best_idx]] += 1
                    
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code.copy()
    
    return [list(idx_to_word(c)) for c in best_code]

def _greedy_cover(q, n, radius, deadline):
    """Fallback greedy for non-standard parameters."""
    total = q ** n
    if total > 100000:
        # Too large, return empty or simple code
        return []
    
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for d in range(n - 1, -1, -1):
            w[d] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors
    neighbors = []
    for idx in range(total):
        w = idx_to_word(idx)
        nbs = [idx]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    nbs.append(word_to_idx(tuple(new_w)))
        neighbors.append(nbs)
    
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    coverage = np.zeros(total, dtype=np.int32)
    current_code = []
    
    # Greedy
    for _ in range(total):
        if time.time() > deadline - 5:
            break
        
        best_idx = -1
        best_count = 0
        for idx in range(total):
            if coverage[idx] > 0:
                continue
            cnt = (coverage[neighbors_np[idx]] == 0).sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        current_code.append(best_idx)
        coverage[neighbors_np[best_idx]] += 1
    
    return [list(idx_to_word(c)) for c in current_code]

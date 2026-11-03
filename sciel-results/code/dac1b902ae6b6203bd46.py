import numpy as np
import itertools
import time
import random

Q = 3
N = 6
NUM_WORDS = Q ** N

all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute ball masks (boolean) and ball index lists
ball_masks = []
ball_idx_lists = []
for i in range(NUM_WORDS):
    c = all_words[i]
    mask = np.zeros(NUM_WORDS, dtype=bool)
    mask[i] = True
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            nb = c.copy()
            nb[pos] = val
            idx = word_to_idx[tuple(nb)]
            mask[idx] = True
    ball_masks.append(mask)
    ball_idx_lists.append(np.where(mask)[0])

# Precompute ball_masks as a 2D array for vectorized operations
ball_matrix = np.array(ball_masks, dtype=np.int8)  # shape (729, 729)

def get_tetracode_code():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 118

    best_code = get_tetracode_code()
    best_size = len(best_code)

    def greedy_cover():
        """Vectorized greedy: pick codeword covering most uncovered words."""
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            # Vectorized: score[i] = sum of uncovered words in ball[i]
            scores = ball_matrix.dot(uncovered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def greedy_randomized():
        """Randomized greedy: pick from top candidates."""
        uncovered = np.ones(NUM_WORDS, dtype=np.int32)
        code = []
        for _ in range(NUM_WORDS):
            if uncovered.sum() == 0:
                break
            scores = ball_matrix.dot(uncovered)
            max_s = scores.max()
            if max_s == 0:
                break
            # Pick randomly among top 20% candidates
            threshold = max_s * 0.7
            top = np.where(scores >= threshold)[0]
            c = top[np.random.randint(len(top))]
            code.append(c)
            uncovered[ball_masks[c]] = 0
        return code

    def reduction(code):
        """Remove redundant codewords (those whose ball is fully covered by others)."""
        code = list(code)
        if not code:
            return code
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            np.random.shuffle(indices)
            for idx in indices:
                c = code[idx]
                if np.all(cov_counts[ball_masks[c]] > 1):
                    cov_counts[ball_masks[c]] -= 1
                    code.pop(idx)
                    changed = True
                    break
        return code

    def local_search(code, max_iters=20000):
        """Local search with remove, add, swap, and pair-exchange moves."""
        code = list(code)
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            cov_counts[ball_masks[c]] += 1
        
        best_code = list(code)
        best_unc = (cov_counts == 0).sum()
        best_size = len(code)
        
        T = 15.0
        cool = (0.05 / 15.0) ** (1.0 / max_iters)
        
        # Precompute for faster swap evaluation
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            mt = np.random.randint(4)
            
            if mt == 0 and len(code) > 0:
                # Remove a codeword
                idx = np.random.randint(len(code))
                r = code[idx]
                would_lose = (cov_counts[ball_masks[r]] == 1).sum()
                delta_unc = would_lose
                delta_size = -1
                if delta_unc < 0 or (delta_unc == 0 and delta_size < 0) or np.random.rand() < np.exp(-(1000.0 * delta_unc + delta_size) / T):
                    code.pop(idx)
                    cov_counts[ball_masks[r]] -= 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 1:
                # Add a codeword
                a = np.random.randint(NUM_WORDS)
                if a in code:
                    continue
                new_cov = (cov_counts[ball_masks[a]] == 0).sum()
                delta_unc = -new_cov
                delta_size = 1
                if delta_unc < 0 or (delta_unc == 0 and delta_size < 0) or np.random.rand() < np.exp(-(1000.0 * delta_unc + delta_size) / T):
                    code.append(a)
                    cov_counts[ball_masks[a]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 2 and len(code) > 0:
                # Swap: remove r, add a
                idx = np.random.randint(len(code))
                r = code[idx]
                a = np.random.randint(NUM_WORDS)
                if a == r or a in code:
                    continue
                
                unique_r = ball_masks[r] & ~ball_masks[a]
                unique_a = ball_masks[a] & ~ball_masks[r]
                
                L = (cov_counts[unique_r] == 1).sum()
                Nv = (cov_counts[unique_a] == 0).sum()
                
                delta_unc = L - Nv
                if delta_unc < 0 or np.random.rand() < np.exp(-1000.0 * delta_unc / T):
                    code[idx] = a
                    cov_counts[ball_masks[r]] -= 1
                    cov_counts[ball_masks[a]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            elif mt == 3 and len(code) >= 2:
                # Pair exchange: remove r1, r2; add a1, a2
                idx1 = np.random.randint(len(code))
                idx2 = np.random.randint(len(code) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                r1 = code[idx1]
                r2 = code[idx2]
                
                a1 = np.random.randint(NUM_WORDS)
                a2 = np.random.randint(NUM_WORDS)
                if a1 == r1 or a1 == r2 or a2 == r1 or a2 == r2 or a1 in code or a2 in code or a1 == a2:
                    continue
                
                # Compute delta for pair exchange
                # Words that lose coverage: in (ball[r1] U ball[r2]) but not in (ball[a1] U ball[a2]) with count==1
                # Words that gain coverage: in (ball[a1] U ball[a2]) but not in (ball[r1] U ball[r2]) with count==0
                
                ball_r1_r2 = ball_masks[r1] | ball_masks[r2]
                ball_a1_a2 = ball_masks[a1] | ball_masks[a2]
                
                lose_mask = ball_r1_r2 & ~ball_a1_a2
                gain_mask = ball_a1_a2 & ~ball_r1_r2
                
                L = (cov_counts[lose_mask] == 1).sum()
                Nv = (cov_counts[gain_mask] == 0).sum()
                
                delta_unc = L - Nv
                if delta_unc < 0 or np.random.rand() < np.exp(-1000.0 * delta_unc / T):
                    code[idx1] = a1
                    code[idx2] = a2
                    cov_counts[ball_masks[r1]] -= 1
                    cov_counts[ball_masks[r2]] -= 1
                    cov_counts[ball_masks[a1]] += 1
                    cov_counts[ball_masks[a2]] += 1
                    unc = (cov_counts == 0).sum()
                    if unc < best_unc or (unc == best_unc and len(code) < best_size):
                        best_unc = unc
                        best_size = len(code)
                        best_code = list(code)
            
            T *= cool
        
        return best_code

    # Phase 1: Generate diverse initial solutions
    inits = []
    for i in range(20):
        if time.time() > deadline:
            break
        if i < 8:
            sol = greedy_cover()
        else:
            sol = greedy_randomized()
        sol = reduction(sol)
        inits.append(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_code = [list(all_words[c]) for c in sol]
    
    # Phase 2: Local search from best initial solutions
    inits.sort(key=len)
    for sol in inits[:12]:
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            res = local_search(sol, max_iters=10000)
            res = reduction(res)
            if len(res) < best_size:
                best_size = len(res)
                best_code = [list(all_words[c]) for c in res]
    
    # Phase 3: If we found something good, do more refinement
    if best_size <= 73 and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            res = local_search([all_words.index(np.array(c)) if c in [tuple(w) for w in all_words] else -1 for c in best_code], max_iters=15000)
            # Convert back to indices
            code_idx = []
            for c in res:
                if c >= 0:
                    code_idx.append(c)
            if code_idx:
                res = reduction(code_idx)
                if len(res) < best_size:
                    best_size = len(res)
                    best_code = [list(all_words[c]) for c in res]
    
    return best_code

import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We handle the general case but optimize for the specific task.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other parameters if they ever appear
        # though the contract implies specific task parameters.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = 3 ** n
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precompute neighbors for all words in the space
    # word is an integer in [0, N-1]
    # neighbors[word] is a list of integers (the word itself + 16 neighbors)
    neighbors = [[] for _ in range(N)]
    
    # To speed up precomputation, we can use base-3 arithmetic properties
    # Or simply iterate. 6561 * 17 ops is tiny.
    for i in range(N):
        # Convert i to base 3
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % 3)
            tmp //= 3
        # digits is reversed (LSB first)
        
        ball = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                # Calculate new integer
                diff = new_val - orig
                new_int = i + diff * (3 ** pos)
                ball.append(new_int)
        neighbors[i] = ball

    # Convert neighbors to numpy arrays for faster indexing
    neighbors_np = [np.array(ball, dtype=np.int32) for ball in neighbors]

    # Dual Greedy Set Cover with Shadow Prices
    # 1. Initialize cover_count to 0 for all words.
    # 2. Maintain 'uncovered' mask.
    # 3. Maintain 'shadow_price' for each word: inverse of number of available codewords that cover it.
    #    Initially, every word is covered by 17 codewords (its neighbors). So price = 1/17.
    #    As we pick codewords, the "availability" of other codewords to cover a word decreases
    #    if those codewords are "used up" or if we prioritize scarcity.
    #    Actually, standard dual greedy for set cover:
    #    While uncovered exists:
    #      For each candidate c, score = sum(shadow_price[u] for u in ball(c) if u is uncovered)
    #      Pick c with max score.
    #      Add c.
    #      Update shadow prices: for each u in ball(c), decrease the "supply" count?
    #      In primal-dual, we increase price of u until some set is tight.
    #      Here, a simpler heuristic:
    #      price[u] = 1 / (number of remaining candidate codewords that cover u)
    #      But "remaining candidates" is the whole space minus picked? No, any word can be a codeword.
    #      So the set of candidates is always all N words.
    #      However, if we pick c, we "consume" the coverage capacity? No.
    #      
    #      Let's use a different structural change: **Weighted Greedy with Decay**.
    #      Weight of a word u starts at 1.
    #      When we pick a codeword c, for all u in ball(c), we mark them covered.
    #      To escape local optima, we can re-weight uncovered words based on how "isolated" they are.
    #      A word u is "hard" if few codewords cover it. But all words have exactly 17 codewords covering them.
    #      So uniform weight is optimal for standard greedy.
    #      
    #      The prompt suggests: "prioritizes codewords covering words with the highest 'shadow prices' (inverse frequency of remaining candidate coverers)".
    #      If we assume that once a codeword is picked, it's "used", does that reduce the candidates for other words?
    #      No, a word u can be covered by ANY of its 17 neighbors. Picking c doesn't prevent picking another neighbor of u.
    #      So the "frequency of remaining candidate coverers" for u is always 17, unless we restrict the search space.
    #      
    #      Perhaps the idea is to restrict the search space to a subset?
    #      Or maybe it means: if we pick c, the "value" of covering u is reduced if u is also covered by many other picked codewords?
    #      That's just standard marginal gain.
    #      
    #      Let's try a **Simulated Annealing** approach on the set of codewords, starting from the greedy solution.
    #      Moves: Remove one, Add one, Swap one.
    #      Cost function: Number of codewords + Penalty * (Number of uncovered words).
    #      We want to minimize codewords while keeping uncovered = 0.
    #      If uncovered > 0, we must add or swap to cover them.
    #      If uncovered == 0, we can remove or swap to reduce size.

    # Phase 1: Greedy Initialization
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    
    # Standard greedy: pick candidate covering most uncovered
    # To speed this up, we can use a random restart or a smarter heuristic.
    # Let's stick to a fast greedy first.
    # We can vectorize the "count uncovered in ball" if we precompute a matrix? 6561x6561 is 43M bools, ~43MB. Feasible.
    # But building it takes time.
    # Let's just do the O(N^2) loop with NumPy sums.
    
    # Optimization: Instead of checking all N candidates, we can pick a random uncovered word,
    # and pick the best neighbor. This is a heuristic that is O(N * 17 * 17) roughly?
    # No, it's O(Steps * 17 * 17). Steps ~ N/17. So O(N * 17) ~ 100k ops. Very fast.
    # But it might be suboptimal.
    # Let's try the full greedy but with a time limit.
    
    while True:
        if time.time() > deadline:
            break
        if not np.any(uncovered_mask):
            break
        
        # Pick a random uncovered word to anchor the search
        unc_idx = np.where(uncovered_mask)[0]
        if len(unc_idx) == 0:
            break
        target = unc_idx[random.randint(0, len(unc_idx)-1)]
        
        # Find the best neighbor of target
        best_cand = -1
        max_cov = -1
        for cand in neighbors_np[target]:
            cov = np.sum(uncovered_mask[cand]) # Wait, neighbors_np[cand] is the ball of cand
            # Correction: neighbors_np[cand] contains the indices of the ball.
            # So we need sum of uncovered_mask at those indices.
            # neighbors_np[cand] is an array of ints.
            cov = np.sum(uncovered_mask[neighbors_np[cand]])
            if cov > max_cov:
                max_cov = cov
                best_cand = cand
        
        if best_cand == -1:
            break
            
        code.append(best_cand)
        
        # Update uncovered_mask
        for w in neighbors_np[best_cand]:
            uncovered_mask[w] = False

    # Rebuild cover_count
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for w in neighbors_np[c]:
            cover_count[w] += 1

    # Phase 2: Local Search (Simulated Annealing)
    # State: code (list of ints)
    # Cost: len(code) * 1000 + (N - sum(cover_count > 0))
    # We want to minimize cost.
    
    def get_cost(code_list):
        # Rebuild cover count is O(len(code) * 17)
        # Check uncovered
        # This is expensive if done every step.
        # Let's maintain cover_count incrementally.
        pass

    # Incremental Local Search
    # We maintain cover_count and uncovered_count.
    uncovered_count = int(np.sum(uncovered_mask))
    
    # Ensure full coverage first
    while uncovered_count > 0 and time.time() < deadline:
        # Find an uncovered word
        unc_idx = np.where(uncovered_mask)[0]
        if len(unc_idx) == 0:
            break
        target = unc_idx[random.randint(0, len(unc_idx)-1)]
        
        # Find a neighbor that covers the most uncovered
        best_cand = -1
        max_cov = -1
        for cand in neighbors_np[target]:
            cov = np.sum(uncovered_mask[neighbors_np[cand]])
            if cov > max_cov:
                max_cov = cov
                best_cand = cand
        
        if best_cand == -1:
            break
            
        # Add best_cand
        code.append(best_cand)
        for w in neighbors_np[best_cand]:
            if uncovered_mask[w]:
                uncovered_mask[w] = False
                uncovered_count -= 1
            cover_count[w] += 1

    # Now minimize size
    # Moves:
    # 1. Remove: If min(cover_count[ball(c)]) > 1, remove c.
    # 2. Swap: Replace c with w.
    #    Valid if min(cover_count[v] for v in ball(c) \ ball(w)) > 1.
    #    If valid, perform swap.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Removal
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = code[i]
            counts_in_ball = cover_count[neighbors_np[c]]
            if np.min(counts_in_ball) > 1:
                for w in neighbors_np[c]:
                    cover_count[w] -= 1
                code.pop(i)
                improved = True
        
        # Pass 2: Swap
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            # Try swapping with neighbors of c
            for w in neighbors_np[c]:
                if w == c:
                    continue
                # Check validity
                valid = True
                for v in neighbors_np[c]:
                    if v not in neighbors_np[w]:
                        if cover_count[v] == 1:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    for v in neighbors_np[c]:
                        cover_count[v] -= 1
                    for v in neighbors_np[w]:
                        cover_count[v] += 1
                    code[i] = w
                    improved = True
                    break
            
            if improved:
                break

    # Convert indices back to lists of integers
    result = []
    for c in code:
        word = []
        tmp = c
        for _ in range(n):
            word.append(tmp % 3)
            tmp //= 3
        word.reverse()
        result.append(word)
        
    return result

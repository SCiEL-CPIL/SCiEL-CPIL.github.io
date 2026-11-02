import itertools
import time
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341214312142312"
        
    # 1. Generate the standard superpermutation for n-1 to get the initial order
    # We need the order of permutations of 1..n-1 as they appear in the standard construction.
    def standard_superperm(k):
        if k == 1:
            return [1]
        word = standard_superperm(k-1)
        perms, seen = [], set()
        # Extract permutations of length k-1 from the current word
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Find maximum overlap with the end of new_word
            overlap = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        return word

    # Get the standard word for n-1
    word_prev = standard_superperm(n-1)
    
    # Extract the order of permutations of 1..n-1
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)
            
    N = len(perms_order)
    # N should be (n-1)!
    
    # 2. Precompute the overlap matrix
    # M[i][j] = max k such that perms_order[i][-k:] == perms_order[j][:k]
    # This overlap is used when transitioning from permutation i to permutation j in the superpermutation of size n.
    # The block for perm p is p + [n] + p.
    # If we go from p_i to p_j, the suffix of p_i + [n] + p_i is ... p_i.
    # The prefix of p_j + [n] + p_j is p_j ...
    # Actually, the standard construction overlaps the suffix of the previous block with the prefix of the next block.
    # Block i: p_i, n, p_i
    # Block j: p_j, n, p_j
    # We want to overlap the end of Block i with the start of Block j.
    # The end of Block i is p_i. The start of Block j is p_j.
    # So we look for the longest suffix of p_i that is a prefix of p_j.
    # Let this length be k. Then we can append Block j starting from index k.
    # The length added is len(Block j) - k = (2*(n-1) + 1) - k.
    # To minimize total length, we maximize sum of k (overlaps).
    
    # Note: The first block has no overlap.
    # Total length = len(Block 0) + sum_{i=1}^{N-1} (len(Block i) - M[path[i-1]][path[i]])
    # len(Block) is constant. So we maximize sum of M.
    
    # Check time
    if time.time() > deadline - 30:
        # Not enough time for matrix calc and search, return standard
        # Reconstruct standard word for n
        # We can just run the standard construction for n directly if we haven't done so?
        # We have word_prev for n-1. We can extend it.
        # But we already have perms_order.
        # Let's just return the standard construction result for n if we are in a hurry.
        # However, we can quickly construct it from perms_order with identity mapping.
        word = list(perms_order[0]) + [n] + list(perms_order[0])
        for idx in range(1, N):
            p = perms_order[idx]
            block = list(p) + [n] + list(p)
            ov = 0
            prev = perms_order[idx-1]
            for k in range(n-1, 0, -1):
                if prev[-k:] == p[:k]:
                    ov = k
                    break
            word += block[ov:]
        return "".join(map(str, word))

    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        p_i = perms_order[i]
        for j in range(N):
            p_j = perms_order[j]
            max_k = n - 1
            for k in range(max_k, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    M[i][j] = k
                    break
    
    # Initial path is 0, 1, ..., N-1
    path = np.arange(N)
    
    # Calculate initial score (sum of overlaps)
    def calc_score(path):
        # path is an array of indices
        # Sum M[path[i]][path[i+1]] for i in 0..N-2
        return np.sum(M[path[:-1], path[1:]])
    
    current_score = calc_score(path)
    
    # 3. Local Search: 2-opt
    # We want to maximize score.
    # A 2-opt move reverses a segment of the path.
    # Path: ... u, v, ..., x, y ...
    # Becomes: ... u, x, ..., v, y ...
    # Edges removed: (u, v) and (x, y)
    # Edges added: (u, x) and (v, y)
    # Delta = M[u][x] + M[v][y] - M[u][v] - M[x][y]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # To speed up, we can randomize or just iterate. 
        # For N=5040, O(N^2) is 25 million operations. In Python, this is too slow for a full pass if done naively.
        # We need to be efficient.
        # However, NumPy vectorization can help, but 2-opt logic is tricky to vectorize fully due to the break condition.
        # Let's try a few passes. If it's too slow, we'll rely on fewer passes.
        
        # Optimization: Only check a subset of moves or use a faster heuristic?
        # For now, let's implement a simple 2-opt loop but with time checks.
        # Since N is 5040, a full O(N^2) pass is ~25M iterations. Python does ~10M simple ops/sec. 
        # So one full pass takes ~2.5 seconds. We have ~100s. So ~40 passes are possible.
        
        for i in range(N - 1):
            if time.time() > deadline - 5:
                break
            u = path[i]
            v = path[i+1]
            # We can vectorize the inner loop calculation for delta
            # delta[j] = M[u, path[j]] + M[path[i+1], path[j+1]] - M[u, v] - M[path[j], path[j+1]]
            # for j in i+1..N-1
            
            # j ranges from i+1 to N-1
            # path[j] is the element at index j
            # path[j+1] is the element at index j+1
            
            # If j == N-1, the term M[path[i+1], path[j+1]] and M[path[j], path[j+1]] doesn't exist in the same way?
            # If j == N-1, we reverse path[i+1..N-1].
            # Old edges: (u, v) and (path[N-2], path[N-1])? No.
            # Path segment: u, v, ..., path[N-1]
            # Reverse v..path[N-1]: u, path[N-1], ..., v
            # Edges removed: (u, v) and (path[N-2], path[N-1]) is NOT removed if N-2 is inside the reversal?
            # Let's trace carefully.
            # Path indices: 0, 1, ..., i, i+1, ..., j, j+1, ..., N-1
            # Reverse i+1..j.
            # New path: 0, ..., i, j, j-1, ..., i+1, j+1, ..., N-1
            # Edges changed:
            # 1. (path[i], path[i+1]) becomes (path[i], path[j])
            # 2. (path[j], path[j+1]) becomes (path[i+1], path[j+1])
            # If j == N-1, there is no path[j+1].
            # So edge 2 is not present.
            # Removed: (u, v) and (path[N-2], path[N-1])? No, (path[N-2], path[N-1]) becomes an internal edge of the reversed segment?
            # If j=N-1, the segment is i+1..N-1.
            # Old: ... u, v, ..., path[N-2], path[N-1]
            # New: ... u, path[N-1], ..., path[N-2], v
            # Edges removed: (u, v) and (path[N-2], path[N-1])?
            # Wait, the edge (path[N-2], path[N-1]) is between two nodes in the reversed segment. Its orientation flips, but it's still an edge in the path.
            # However, the score function sums M[a][b] for consecutive (a,b).
            # M is not symmetric. M[a][b] != M[b][a].
            # So reversing an edge (a,b) to (b,a) changes the score from M[a][b] to M[b][a].
            # My delta formula assumed the internal edges don't change contribution?
            # No, 2-opt on a directed graph (asymmetric M) is more complex.
            # Reversing a segment reverses all internal edges.
            # This makes 2-opt harder.
            
            # Let's stick to the simpler 1-swap or just accept that 2-opt might be less effective if not handled carefully.
            # Alternatively, we can just use the standard 2-opt logic which assumes we are maximizing sum of weights.
            # If M is asymmetric, standard 2-opt still works for finding local optima in the "sum of edges" metric, 
            # but the delta calculation must account for all changed edges.
            
            # Changed edges:
            # 1. (u, v) -> (u, path[j])
            # 2. (path[j], path[j+1]) -> (path[i+1], path[j+1])  (if j < N-1)
            # 3. Internal edges (path[k], path[k+1]) for k in i+1..j-1 become (path[k+1], path[k]).
            
            # This is O(N) per move to calculate delta if we sum internal edges.
            # Total O(N^3) for full pass? No, O(N^2) moves * O(N) delta = O(N^3). Too slow.
            
            # Let's try a simpler local search: Hill Climbing on swaps.
            # Swap path[i] and path[j].
            # This is also complex with asymmetric M.
            
            # Given the complexity and time, let's try a different approach:
            # The standard construction is already quite good.
            # The main gain from Inspiration 2 was the *idea* of maximizing overlap.
            # If 2-opt is too slow/complex, we can try a simpler greedy improvement:
            # Iterate through the path, and for each position, try to swap with the next element if it improves the local score.
            # Or, simply return the standard construction if the search is too risky/slow.
            
            # However, the prompt asks to combine strategies.
            # Let's try a limited 2-opt where we only check j = i+1 (adjacent swap) or j = i+2.
            # Adjacent swap (i, i+1):
            # Old: u, v, w
            # New: u, w, v
            # Removed: (u,v), (v,w)
            # Added: (u,w), (w,v)
            # Delta = M[u][w] + M[w][v] - M[u][v] - M[v][w]
            # This is O(1) per swap. O(N) per pass. Very fast.
            
            # Let's implement adjacent swap hill climbing first.
            
            for i in range(N - 1):
                if time.time() > deadline - 5:
                    break
                u = path[i]
                v = path[i+1]
                if i < N - 2:
                    w = path[i+2]
                    delta = M[u, w] + M[w, v] - M[u, v] - M[v, w]
                else:
                    # i = N-2, swap last two
                    # Old: u, v
                    # New: v, u
                    # Removed: (u,v)
                    # Added: (v,u)
                    # But wait, is there an edge before u? Yes, path[i-1] to u.
                    # Let x = path[i-1] if i > 0 else None.
                    # If i > 0:
                    # Old: x, u, v
                    # New: x, v, u
                    # Removed: (x,u), (u,v)
                    # Added: (x,v), (v,u)
                    # Delta = M[x][v] + M[v][u] - M[x][u] - M[u][v]
                    # If i == 0:
                    # Old: u, v
                    # New: v, u
                    # Delta = M[v][u] - M[u][v]
                    
                    if i > 0:
                        x = path[i-1]
                        delta = M[x, v] + M[v, u] - M[x, u] - M[u, v]
                    else:
                        delta = M[v, u] - M[u, v]
                
                if delta > 0:
                    # Swap path[i] and path[i+1]
                    path[i], path[i+1] = path[i+1], path[i]
                    current_score += delta
                    improved = True
                    # Restart or continue? Continue is fine for hill climbing.
        
        if not improved:
            break

    # 4. Construct the final word from the optimized path
    word = list(perms_order[path[0]]) + [n] + list(perms_order[path[0]])
    
    for idx in range(1, N):
        p = perms_order[path[idx]]
        block = list(p) + [n] + list(p)
        # Overlap is M[path[idx-1]][path[idx]]
        ov = M[path[idx-1], path[idx]]
        word += block[ov:]
        
    # 5. Optional: Greedy Deletion Polish
    # If we have time, try to remove characters.
    # This is the strategy from the current program.
    if time.time() < deadline - 10:
        # Map permutations to IDs for fast counting
        all_perms = list(itertools.permutations(range(1, n + 1)))
        perm_to_id = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        def get_counts(w):
            counts = [0] * num_perms
            L = len(w)
            for i in range(L - n + 1):
                p = tuple(w[i:i+n])
                if p in perm_to_id:
                    counts[perm_to_id[p]] += 1
            return counts

        counts = get_counts(word)
        
        # Verify initial validity
        if any(c == 0 for c in counts):
            return "".join(map(str, word))

        # Greedy deletion loop
        # We iterate until no single character can be removed or time runs out.
        while time.time() < deadline - 2:
            improved = False
            L = len(word)
            # Check each index
            for i in range(L):
                if time.time() > deadline - 2:
                    break
                    
                # Check if deleting word[i] is valid
                # Windows affected are those starting at s where s <= i < s+n
                # i.e., s in [i-n+1, i]
                
                affected = {}
                for s in range(max(0, i - n + 1), min(i, L - n) + 1):
                    p = tuple(word[s:s+n])
                    if p in perm_to_id:
                        pid = perm_to_id[p]
                        affected[pid] = affected.get(pid, 0) + 1
                
                valid = True
                for pid, cnt_in_window in affected.items():
                    if counts[pid] - cnt_in_window < 1:
                        valid = False
                        break
                
                if valid:
                    word.pop(i)
                    counts = get_counts(word)
                    improved = True
                    break
        
            if not improved:
                break

    return "".join(map(str, word))

import time
import sys

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Max gap is LIMIT
    G = LIMIT
    
    # Fenwick Tree implementation
    # Each node stores (max_length, prev_prime_idx, prev_gap)
    # We use -1 for null prev
    # Comparison: higher length is better.
    
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            # Store tuples (length, prev_idx, prev_gap)
            # Initialize with (0, -1, -1)
            self.tree = [(0, -1, -1) for _ in range(size + 1)]
            
        def update(self, idx, val):
            # val is (length, prev_idx, prev_gap)
            while idx <= self.size:
                # If new val has greater length, replace
                if val[0] > self.tree[idx][0]:
                    self.tree[idx] = val
                idx += idx & -idx
                
        def query(self, idx):
            # Get max val in [1, idx]
            res = (0, -1, -1)
            while idx > 0:
                if self.tree[idx][0] > res[0]:
                    res = self.tree[idx]
                idx -= idx & -idx
            return res

    # Initialize Fenwick Trees for each prime
    # This might be memory heavy if done naively, but N is small.
    # However, creating 1229 objects with lists of size 10000 is ~12M tuples.
    # Python tuples are heavy.
    # Let's try to optimize memory by using parallel lists or just accepting the overhead.
    # 12M tuples * ~72 bytes/tuple = ~800MB. This might be too much.
    
    # Alternative: Store only the best path info in a 2D array, and use Fenwick tree only for lengths?
    # No, we need the path.
    
    # Let's check memory constraints. Usually 1-2GB is available.
    # But 800MB is risky.
    
    # Optimization: We don't need to store the full history in the Fenwick tree nodes if we can reconstruct.
    # Actually, the Fenwick tree node MUST store the argmax to allow reconstruction.
    
    # Can we reduce the size of G?
    # Max gap between consecutive primes is small, but we are looking for subsequences.
    # The gap $g$ in the sequence can be large.
    # However, if the gap is large, the sequence length is short.
    # We only care about sequences longer than the current best.
    
    # Let's stick to the standard DP but be careful with memory.
    # We can use arrays of ints instead of tuples.
    # tree_len[i], tree_prev_idx[i], tree_prev_gap[i]
    
    trees_len = [[0] * (G + 1) for _ in range(n)]
    trees_prev_idx = [[-1] * (G + 1) for _ in range(n)]
    trees_prev_gap = [[-1] * (G + 1) for _ in range(n)]
    
    # This is 3 * 1229 * 10001 * 4 bytes (if int32) ~ 150MB.
    # Using Python lists of ints is much larger.
    # Let's use `array` module or `numpy`.
    
    import numpy as np
    
    # Initialize numpy arrays
    # Note: Fenwick tree updates are random access, so numpy might not be faster for the loop itself,
    # but memory is tighter.
    
    # Actually, pure Python lists are faster for access than numpy in tight loops if not vectorized.
    # But memory is the issue.
    
    # Let's try a hybrid:
    # We only update/query for valid (i, g) pairs where p_i - g is prime.
    # We don't need to initialize the whole tree.
    
    # Let's use a dictionary based Fenwick tree? No, indices are dense.
    
    # Let's assume Python lists are okay and monitor memory.
    # If it fails, we'll switch to a different strategy.
    # Actually, 12M Python ints is ~300MB per list. 3 lists = 900MB.
    # This is likely to MLE (Memory Limit Exceeded) or be slow.
    
    # Better approach:
    # We don't need to store the Fenwick tree for ALL primes at once if we process sequentially?
    # No, we need `tree[j]` when processing `i`.
    
    # Let's reduce G.
    # If we have a sequence of length L, the average gap is $(p_k - p_1)/L$.
    # For L=100, gap is ~100.
    # For L=20, gap is ~500.
    # We can cap G dynamically?
    
    # Let's try to implement with standard lists and hope for the best, or use `array('i')`.
    
    from array import array
    
    trees_len = [array('i', [0]) * (G + 1) for _ in range(n)]
    trees_prev_idx = [array('i', [-1]) * (G + 1) for _ in range(n)]
    trees_prev_gap = [array('i', [-1]) * (G + 1) for _ in range(n)]
    
    # array('i') uses 4 bytes per element.
    # 3 * 1229 * 10001 * 4 = 147 MB. This is safe.
    
    def fenwick_update(i, g, length, prev_idx, prev_gap):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        while g <= G:
            if length > tree_l[g]:
                tree_l[g] = length
                tree_pi[g] = prev_idx
                tree_pg[g] = prev_gap
            g += g & -g
            
    def fenwick_query(i, g):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        max_len = 0
        best_pi = -1
        best_pg = -1
        while g > 0:
            if tree_l[g] > max_len:
                max_len = tree_l[g]
                best_pi = tree_pi[g]
                best_pg = tree_pg[g]
            g -= g & -g
        return max_len, best_pi, best_pg

    best_len = 1
    best_end_idx = 0
    best_end_gap = -1
    
    # Initialize base case: sequences of length 1
    # dp[i][0] = 1? No, gap is defined between two primes.
    # A sequence of length 1 has no gap.
    # We can treat it as gap 0.
    for i in range(n):
        # Update tree[i] at index 0?
        # Fenwick trees usually start at 1.
        # Let's shift indices by 1.
        # gap g -> index g+1.
        # Max gap G -> index G+1.
        # Array size G+2.
        
        pass # Will re-initialize with shift
    
    # Re-initialize with shift
    G_idx = G + 1
    trees_len = [array('i', [0]) * (G_idx + 1) for _ in range(n)]
    trees_prev_idx = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    trees_prev_gap = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    
    def fenwick_update(i, g_idx, length, prev_idx, prev_gap):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        while g_idx <= G_idx:
            if length > tree_l[g_idx]:
                tree_l[g_idx] = length
                tree_pi[g_idx] = prev_idx
                tree_pg[g_idx] = prev_gap
            g_idx += g_idx & -g_idx
            
    def fenwick_query(i, g_idx):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        max_len = 0
        best_pi = -1
        best_pg = -1
        while g_idx > 0:
            if tree_l[g_idx] > max_len:
                max_len = tree_l[g_idx]
                best_pi = tree_pi[g_idx]
                best_pg = tree_pg[g_idx]
            g_idx -= g_idx & -g_idx
        return max_len, best_pi, best_pg

    # Base case: length 1 sequences
    # Represented as ending at i with "last gap" 0 (index 1)
    for i in range(n):
        fenwick_update(i, 1, 1, -1, -1)
        
    # DP
    for i in range(1, n):
        p_i = primes[i]
        # Iterate over all possible previous primes j
        # We need p_j < p_i
        # Gap g = p_i - p_j
        # g_idx = g + 1
        
        # Optimization: Only iterate j such that time is safe
        for j in range(i):
            p_j = primes[j]
            g = p_i - p_j
            g_idx = g + 1
            
            # Query tree[j] for max length with last gap <= g
            # Note: The previous gap in the sequence ending at j must be <= g
            prev_len, prev_prev_idx, prev_prev_gap = fenwick_query(j, g_idx)
            
            # New length
            new_len = prev_len + 1
            
            # Update tree[i] at g_idx
            # The previous prime in the new sequence is j
            # The previous gap is g
            fenwick_update(i, g_idx, new_len, j, g)
            
            if new_len > best_len:
                best_len = new_len
                best_end_idx = i
                best_end_gap = g # This is the gap used to arrive at i from j?
                # Wait, fenwick_update stores (length, prev_idx, prev_gap) in the tree node.
                # To reconstruct, we need to know which node in tree[i] gave the best result.
                # The best result for tree[i] is not necessarily at index g_idx.
                # We need to find the max over all g_idx in tree[i].
                
        # Check time
        if time.time() - start_time > 80:
            break
            
    # Find the best end state
    # We need to find the max length in any tree[i] and the corresponding path.
    # Actually, we tracked best_len, but we didn't track the specific (i, g) that achieved it globally.
    # We can just scan all trees at the end, or update global best during DP.
    
    # Let's update global best during DP properly.
    # In the loop above, `new_len` is the length of a sequence ending at i with last gap g.
    # We update tree[i] with this.
    # We should check if this `new_len` is the global best.
    
    # Re-run DP with proper tracking
    
    best_len = 1
    best_i = 0
    best_g = 0 # gap used to enter best_i
    
    for i in range(1, n):
        p_i = primes[i]
        for j in range(i):
            p_j = primes[j]
            g = p_i - p_j
            g_idx = g + 1
            
            prev_len, prev_prev_idx, prev_prev_gap = fenwick_query(j, g_idx)
            
            new_len = prev_len + 1
            
            fenwick_update(i, g_idx, new_len, j, g)
            
            if new_len > best_len:
                best_len = new_len
                best_i = i
                best_g = g
                
        if time.time() - start_time > 80:
            break

    # Reconstruct
    seq = []
    curr_i = best_i
    curr_g = best_g
    
    # The state in tree[best_i] at index curr_g+1 should have (best_len, prev_i, prev_g)
    # But fenwick_update might have been overwritten by a better value at the same index?
    # No, we only update if new_len > old_len.
    # So the value stored is the max length for that specific gap.
    # However, the global best might be stored in a Fenwick node that aggregates multiple gaps.
    # Wait, `fenwick_update` updates the tree nodes.
    # To retrieve the specific path for the best_len, we need to know the exact (i, g) pair.
    # We stored `best_i` and `best_g`.
    # We can look up `trees_len[best_i][best_g+1]`?
    # No, the Fenwick tree structure means `tree[i][g_idx]` stores the max for a range.
    # The actual value for gap `g` is not directly accessible from the Fenwick tree array `tree[i]` in a simple way unless we stored it separately.
    
    # Correction:
    # We should store the DP table separately for reconstruction, or use a different structure.
    # Let's store `dp_len[i][g]`, `dp_prev_i[i][g]`, `dp_prev_g[i][g]`.
    # And use Fenwick Tree only for the query part.
    
    # This requires 3 more arrays of size N x G.
    # Memory: 3 * 147MB = 441MB. Total ~600MB. Still okay.
    
    # Let's restart the implementation with this cleaner separation.

    # Re-initialize
    G_idx = G + 1
    trees_len = [array('i', [0]) * (G_idx + 1) for _ in range(n)]
    trees_prev_idx = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    trees_prev_gap = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    
    dp_len = [array('i', [0]) * (G_idx + 1) for _ in range(n)]
    dp_prev_i = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    dp_prev_g = [array('i', [-1]) * (G_idx + 1) for _ in range(n)]
    
    def fenwick_update(i, g_idx, length, prev_idx, prev_gap):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        while g_idx <= G_idx:
            if length > tree_l[g_idx]:
                tree_l[g_idx] = length
                tree_pi[g_idx] = prev_idx
                tree_pg[g_idx] = prev_gap
            g_idx += g_idx & -g_idx
            
    def fenwick_query(i, g_idx):
        tree_l = trees_len[i]
        tree_pi = trees_prev_idx[i]
        tree_pg = trees_prev_gap[i]
        max_len = 0
        best_pi = -1
        best_pg = -1
        while g_idx > 0:
            if tree_l[g_idx] > max_len:
                max_len = tree_l[g_idx]
                best_pi = tree_pi[g_idx]
                best_pg = tree_pg[g_idx]
            g_idx -= g_idx & -g_idx
        return max_len, best_pi, best_pg

    # Base case
    for i in range(n):
        # gap 0, index 1
        dp_len[i][1] = 1
        dp_prev_i[i][1] = -1
        dp_prev_g[i][1] = -1
        fenwick_update(i, 1, 1, -1, -1)
        
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(1, n):
        p_i = primes[i]
        for j in range(i):
            p_j = primes[j]
            g = p_i - p_j
            g_idx = g + 1
            
            prev_len, prev_prev_i, prev_prev_g = fenwick_query(j, g_idx)
            
            new_len = prev_len + 1
            
            # Update DP table
            dp_len[i][g_idx] = new_len
            dp_prev_i[i][g_idx] = j
            dp_prev_g[i][g_idx] = g
            
            # Update Fenwick Tree
            fenwick_update(i, g_idx, new_len, j, g)
            
            if new_len > best_len:
                best_len = new_len
                best_i = i
                best_g = g
                
        if time.time() - start_time > 80:
            break

    # Reconstruct
    seq = []
    curr_i = best_i
    curr_g = best_g
    
    # If best_len is 1, best_g is 0, curr_i is some prime.
    # Handle base case
    if best_len == 1:
        return [primes[best_i]]
        
    while curr_i != -1:
        seq.append(primes[curr_i])
        if curr_g == 0: # Base case marker?
             # Actually, base case has prev_i = -1.
             # Let's check dp_prev_i
             pass
             
        next_g = dp_prev_g[curr_i][curr_g + 1]
        next_i = dp_prev_i[curr_i][curr_g + 1]
        
        if next_i == -1:
            break
            
        curr_i = next_i
        curr_g = next_g
        
    seq.reverse()
    return seq

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res[:10])

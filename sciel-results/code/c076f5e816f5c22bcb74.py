import random
import time
import sys

def construct(q, n, radius):
    # This function is specialized for q=3, n=6, radius=1
    # Approach: Backtracking with Bitsets and Heuristic Pruning
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy fallback to ensure validity if parameters change
        total_words = q ** n
        neighbors = [[] for _ in range(total_words)]
        for w in range(total_words):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            nbrs = [w]
            for pos in range(n):
                div = q ** (n - 1 - pos)
                curr_digit = (w // div) % q
                for new_digit in range(q):
                    if new_digit != curr_digit:
                        new_w = w + (new_digit - curr_digit) * div
                        nbrs.append(new_w)
            neighbors[w] = nbrs
        
        covered = [False] * total_words
        code = []
        while any(not c for c in covered):
            best_count = -1
            best_w = -1
            for w in range(total_words):
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_w = w
            code.append(best_w)
            for nbr in neighbors[best_w]:
                covered[nbr] = True
        return [[(w // q**(n-1-i)) % q for i in range(n)] for w in code]

    # Optimized path for q=3, n=6, radius=1
    total_words = 729
    target_size = 73 # Known best upper bound
    
    # Precompute masks for each center
    # mask[c] is an integer with bits set for c and its neighbors
    masks = [0] * total_words
    # Helper to convert integer to digits
    def to_digits(w):
        return [(w // (3 ** (n - 1 - i))) % 3 for i in range(n)]
        
    for c in range(total_words):
        digits = to_digits(c)
        mask = 0
        # Center itself
        mask |= (1 << c)
        # Neighbors
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    # Create new word
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    # Convert back to int
                    new_w = 0
                    for i, d in enumerate(new_digits):
                        new_w += d * (3 ** (n - 1 - i))
                    mask |= (1 << new_w)
        masks[c] = mask
    
    # Precompute popcount for small integers? No, Python int.bit_count() is fast.
    
    best_code = None
    
    # Backtracking search
    # We want to find a set of centers of size <= target_size that covers all points.
    # uncovered: bitset of uncovered points. Initially all 1s.
    # full_mask: (1 << total_words) - 1
    
    full_mask = (1 << total_words) - 1
    current_uncovered = full_mask
    current_code = []
    
    # To avoid recursion depth issues and allow timeout handling, we use a manual stack or careful recursion.
    # Python recursion limit is usually 1000, depth 73 is fine.
    
    start_time = time.time()
    timeout = 110 # Leave 10s for cleanup
    
    def backtrack(depth, uncovered, code):
        nonlocal best_code, start_time, timeout
        
        if time.time() - start_time > timeout:
            return False
        
        if uncovered == 0:
            if len(code) < len(best_code) if best_code is not None else True:
                best_code = code[:]
            return True
            
        # Pruning: if current depth + lower bound > target_size, fail
        # Lower bound: ceil(popcount(uncovered) / 13)
        # Actually, max coverage is 13.
        remaining = uncovered.bit_count()
        lb = (remaining + 12) // 13
        if depth + lb > target_size:
            return False
            
        # Pick a point to cover
        # Heuristic: Pick the point with the fewest "available" centers?
        # All points have 13 centers that can cover them.
        # However, some centers might be "better" because they cover more *other* uncovered points.
        # Let's pick the first uncovered point (lowest index).
        # To improve, we can pick the point that is "hardest" to cover?
        # In this uniform graph, they are all symmetric initially.
        # As the search progresses, some points might be covered by fewer *remaining* choices?
        # No, we are adding centers.
        # Let's just pick the lowest set bit.
        
        # Find lowest set bit
        # lsb = uncovered & -uncovered
        # p = lsb.bit_length() - 1
        
        # Better heuristic: Pick an uncovered point p.
        # Generate all centers c that cover p.
        # Sort them by how many *currently uncovered* points they cover (descending).
        # Try the best one first.
        
        # To speed up, let's pick the point p that has the minimum number of centers c such that
        # mask[c] covers p AND c is not yet in code?
        # Actually, we can reuse centers? No, codewords must be distinct.
        # But we are building a set, so we don't pick the same center twice.
        
        # Let's pick the first uncovered point p.
        lsb = uncovered & -uncovered
        p = lsb.bit_length() - 1
        
        # Find all centers c such that p is in mask[c]
        # This is equivalent to c being in N(p)
        # We can precompute reverse neighbors: rev_neighbors[p] = list of c such that p in mask[c]
        # Since the graph is regular and undirected (distance 1), N(p) is the same set.
        
        # Get candidates
        # We need to iterate through all 729 centers and check if mask[c] has bit p set?
        # That's 729 checks.
        # Better: Precompute for each p, the list of centers that cover it.
        # Let's do that outside the loop.
        
        return True # Placeholder, we will implement the loop properly below

    # Precompute centers_for_point[p]
    centers_for_point = [[] for _ in range(total_words)]
    for c in range(total_words):
        m = masks[c]
        # Iterate through set bits in m
        # Or simply, for each p in 0..728, if mask[c] has bit p, add c to centers_for_point[p]
        # This is 729*729 = 531k operations. Fast enough.
        for p in range(total_words):
            if (masks[c] >> p) & 1:
                centers_for_point[p].append(c)
                
    # Now implement the recursive search with this precomputation
    # We need to be careful with the "distinct" constraint.
    # We maintain a set of used centers? Or just ensure we don't pick the same one again?
    # Since we are picking a center to cover p, and p is uncovered, the center must not have been picked before?
    # Actually, if a center was picked before, its neighbors are covered. So p would be covered.
    # So we never need to pick a center that is already in the code.
    
    def search(depth, uncovered, code):
        nonlocal best_code
        
        if time.time() - start_time > timeout:
            return False
            
        if uncovered == 0:
            if best_code is None or len(code) < len(best_code):
                best_code = code[:]
            return True
            
        # Pruning
        remaining = uncovered.bit_count()
        lb = (remaining + 12) // 13
        if depth + lb > target_size:
            return False
            
        # Pick point p
        lsb = uncovered & -uncovered
        p = lsb.bit_length() - 1
        
        # Get candidates: centers in centers_for_point[p]
        # Filter out centers already in code?
        # We can maintain a boolean array `used` of size 729.
        # But passing it recursively is expensive.
        # Instead, we can just check `if c in code_set`.
        # Let's maintain `code_set` as a set or a bitset of used centers.
        
        # To optimize, we can sort candidates by marginal gain (how many uncovered points they cover)
        # marginal_gain(c) = (uncovered & masks[c]).bit_count()
        
        candidates = []
        for c in centers_for_point[p]:
            # Check if c is already used
            # We can store used centers in a bitset `used_centers_mask`
            # But we are passing `code` list.
            # Let's use a global `used_mask` that we modify and backtrack.
            pass
        
        # Let's restructure to use a global `used_mask` for speed.
        return True

    # Revised Search with Global State for Speed
    used_mask = 0
    code_list = []
    
    def dfs(depth, uncovered):
        nonlocal used_mask, code_list, best_code
        
        if time.time() - start_time > timeout:
            return False
            
        if uncovered == 0:
            if best_code is None or len(code_list) < len(best_code):
                best_code = code_list[:]
            return True
            
        remaining = uncovered.bit_count()
        lb = (remaining + 12) // 13
        if depth + lb > target_size:
            return False
            
        lsb = uncovered & -uncovered
        p = lsb.bit_length() - 1
        
        # Get candidates
        # We want to try centers that cover the most uncovered points first.
        cands = []
        for c in centers_for_point[p]:
            if (used_mask >> c) & 1:
                continue
            gain = (uncovered & masks[c]).bit_count()
            cands.append((gain, c))
            
        # Sort by gain descending
        cands.sort(reverse=True)
        
        for gain, c in cands:
            # Try center c
            used_mask |= (1 << c)
            code_list.append(c)
            
            new_uncovered = uncovered & ~masks[c]
            
            if dfs(depth + 1, new_uncovered):
                return True
                
            # Backtrack
            code_list.pop()
            used_mask &= ~(1 << c)
            
        return False

    # Initialize
    best_code = None
    used_mask = 0
    code_list = []
    
    # Run search
    # We might need to restart with different tie-breaking if the first fails?
    # The heuristic (greedy choice within backtracking) is deterministic.
    # If it fails to find 73, it might be stuck in a bad local minimum of the search tree.
    # We can randomize the order of candidates with equal gain.
    
    found = dfs(0, full_mask)
    
    if not found:
        # If backtracking didn't find a solution within time or limit, 
        # fall back to a high-quality greedy solution to ensure validity.
        # The greedy solution from the seed was ~73-75.
        # We can run a fast greedy to get a valid code.
        
        covered = [False] * total_words
        code = []
        # Use the precomputed masks for fast coverage check?
        # Greedy: pick c with max (uncovered & mask[c]).bit_count()
        
        uncovered_mask = full_mask
        while uncovered_mask != 0:
            best_c = -1
            best_gain = -1
            for c in range(total_words):
                gain = (uncovered_mask & masks[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            code.append(best_c)
            uncovered_mask &= ~masks[best_c]
            
        best_code = code
        
    # Convert best_code to list of lists
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code

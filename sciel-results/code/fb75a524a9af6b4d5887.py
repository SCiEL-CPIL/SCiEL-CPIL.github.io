import itertools
import time
import sys

def construct(q, n, radius):
    # Fallback for non-specific cases to ensure validity
    if not (q == 3 and n == 7 and radius == 1):
        # Use the seed program's logic as a safe fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Adjust for q if q != 3? The seed assumes q=3 implicitly in tetracode.
            # If q != 3, we can't use tetracode directly.
            # For simplicity and safety, if not (3,7,1), we return a valid but possibly suboptimal code.
            # A simple greedy set cover works for any q, n, radius.
            return _generic_greedy(q, n, radius)
        
        # Specific high-performance solver for q=3, n=7, radius=1
        return _solve_3_7_1()

def _generic_greedy(q, n, radius):
    """
    A generic greedy set cover implementation for small n.
    """
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute coverage masks
    # Using list of sets is slower than bitmasks, but for generic q, n, 
    # bitmasks might be huge if q^n is large. 
    # However, the task is specifically 3,7,1. 
    # For other cases, we assume small N.
    
    cover_masks = [0] * N
    for i, w in enumerate(all_words):
        mask = 0
        # Generate all words within distance radius
        # For efficiency, only do this if N is small enough
        # For large N, this is O(N * q^n * n * q) which is slow.
        # But since the main target is 3,7,1, this fallback is rarely used or used for tiny inputs.
        for offset in itertools.product(range(-radius, radius + 1), repeat=n):
            candidate = tuple((w[k] + offset[k]) % q for k in range(n))
            # Check distance
            dist = sum(1 for k in range(n) if w[k] != candidate[k])
            if dist <= radius:
                idx = word_to_idx[candidate]
                mask |= (1 << idx)
        cover_masks[i] = mask
        
    full_mask = (1 << N) - 1
    uncovered = full_mask
    code = []
    
    while uncovered:
        best_i = -1
        best_count = -1
        for i in range(N):
            count = (cover_masks[i] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_i = i
        code.append(all_words[best_i])
        uncovered &= ~cover_masks[best_i]
        
    return code

def _solve_3_7_1():
    q = 3
    n = 7
    radius = 1
    
    # 1. Generate all words and map to indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 2187
    
    # 2. Precompute cover masks (bitmasks of words covered by each word)
    cover_masks = [0] * N
    # Precompute coverers: for each word j, list of i that cover j
    coverers = [[] for _ in range(N)]
    
    for i, w in enumerate(all_words):
        mask = 0
        # The ball of radius 1 contains 1 + n*(q-1) = 1 + 7*2 = 15 words
        # We can iterate through all words and check distance, or generate neighbors.
        # Generating neighbors is faster.
        # Center w
        mask |= (1 << word_to_idx[w])
        for k in range(n):
            for delta in [-1, 1]:
                new_w = list(w)
                new_w[k] = (new_w[k] + delta) % q
                new_w_tuple = tuple(new_w)
                j = word_to_idx[new_w_tuple]
                mask |= (1 << j)
                coverers[j].append(i)
        cover_masks[i] = mask
        
    full_mask = (1 << N) - 1
    
    # 3. Greedy Set Cover to get initial upper bound
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_i = -1
            best_count = -1
            for i in range(N):
                count = (cover_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
            code.append(i)
            uncovered &= ~cover_masks[best_i]
        return code

    initial_code = greedy_cover()
    best_size = len(initial_code)
    best_code_indices = initial_code.copy()
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    # 4. Branch and Bound to improve
    # We try to find a solution of size < best_size
    # We will iterate target_size from best_size - 1 down to 180
    
    # Helper for B&B
    # state: covered_mask, current_code_list, current_depth
    # We use a global best to prune
    
    # To speed up, we pre-sort coverers by length (static)
    # But the "available" coverers depend on the current state (if the coverer is already in the code, we can't pick it again? 
    # Actually, in Set Cover, we CAN pick the same word again? No, codewords must be distinct.
    # So if i is in current_code, we cannot pick i again.
    # This makes the state space complex.
    
    # Simplification: 
    # Since we are looking for a *minimal* code, and the greedy code is already a cover,
    # we can try to *remove* words.
    # If we remove a word, the set might no longer cover.
    # Local Search: 
    # 1. Start with Greedy code.
    # 2. For each word in code, try to remove it.
    # 3. If removal breaks coverage, try to swap it with a word NOT in the code that restores coverage 
    #    and allows further removals?
    
    # Given the complexity of a full B&B in Python within 120s, 
    # a robust Local Search (1-swap and removal) is often more effective and safer.
    
    # Let's implement a simple Local Search on top of Greedy.
    
    def is_valid_cover(code_indices):
        union_mask = 0
        for i in code_indices:
            union_mask |= cover_masks[i]
        return union_mask == full_mask

    current_code = set(initial_code)
    
    # Phase 1: Remove redundant words
    changed = True
    while changed:
        changed = False
        for i in list(current_code):
            # Try removing i
            test_code = current_code - {i}
            if is_valid_cover(list(test_code)):
                current_code = test_code
                changed = True
    
    best_size = len(current_code)
    best_code_indices = list(current_code)
    
    # Phase 2: Local Search (1-swap)
    # Try to swap a word in the code with a word not in the code
    # This doesn't reduce size directly, but might lead to a state where removal is possible.
    # However, a direct "reduce size" search is harder.
    
    # Let's try a limited B&B targeting best_size - 1
    # If we find a cover of size best_size - 1, we update.
    
    # B&B State:
    # covered_mask: bits covered so far
    # chosen: set of indices chosen so far
    # limit: max allowed size
    
    # Heuristic: Pick uncovered word with fewest available coverers
    # Available coverers: coverers[j] that are not in 'chosen'
    
    import random
    # random.seed(42) # For determinism if needed, but B&B is deterministic
    
    def branch_and_bound(target_size):
        nonlocal best_size, best_code_indices
        
        if target_size >= best_size:
            return False
            
        # Recursion depth limit
        # We will use a stack-based approach or recursion with sys.setrecursionlimit
        # N=2187, depth ~186. Recursion is fine.
        
        # Pre-compute for speed:
        # We need to track 'chosen' to avoid duplicates.
        
        # To make it faster, we can use a "conflict" check.
        # But let's stick to the simple logic.
        
        # Optimization: 
        # If time is up, raise exception.
        
        def search(covered_mask, chosen_set, depth):
            if time.time() - start_time > time_limit:
                raise TimeoutError()
                
            if depth >= target_size:
                return False
                
            if covered_mask == full_mask:
                # Found a solution
                nonlocal best_code_indices
                best_code_indices = list(chosen_set)
                nonlocal best_size
                best_size = depth
                return True
                
            # Lower bound
            uncovered_count = (full_mask ^ covered_mask).bit_count()
            # Max cover of a single word is 15.
            if depth + (uncovered_count + 14) // 15 > target_size:
                return False
                
            # Pick an uncovered word
            # Find an uncovered word with minimum number of available coverers
            uncovered_mask = full_mask ^ covered_mask
            # Iterate through set bits of uncovered_mask
            # To be fast, we can just pick the first one? No, min-options is better.
            # But checking all 2187 is slow.
            # Let's pick a random uncovered word? Or the one with smallest index?
            # Let's try to find one with fewest coverers.
            
            # We can pre-sort words by len(coverers[w])
            # But 'available' changes.
            
            # Heuristic: Pick the uncovered word j with minimum len([c for c in coverers[j] if c not in chosen_set])
            
            min_j = -1
            min_opts = 1000
            # Sample a few uncovered words to speed up?
            # Or just iterate all. 2187 iterations * bit ops is fast.
            
            temp_mask = uncovered_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                # Count available coverers for j
                opts = 0
                for c in coverers[j]:
                    if c not in chosen_set:
                        opts += 1
                        if opts >= min_opts:
                            break
                if opts < min_opts:
                    min_opts = opts
                    min_j = j
                    if opts == 1:
                        break # Can't do better than 1
                
            if min_j == -1:
                return False # Should not happen if uncovered_mask != 0
                
            # Branch on coverers of min_j
            # Order: try those that cover most uncovered words first
            candidates = []
            for c in coverers[min_j]:
                if c not in chosen_set:
                    # Score: how many new words does c cover?
                    new_cover = cover_masks[c] & uncovered_mask
                    score = new_cover.bit_count()
                    candidates.append((score, c))
            
            candidates.sort(reverse=True)
            
            for score, c in candidates:
                new_chosen = chosen_set | {c}
                new_covered = covered_mask | cover_masks[c]
                
                if search(new_covered, new_chosen, depth + 1):
                    return True
                    
            return False
            
        # Start search
        # To avoid deep recursion overhead, we can limit the number of branches or use iterative deepening.
        # We will just call it.
        
        # Initial state
        # We can seed the search with the greedy solution's prefix?
        # No, empty set is fine, but it might explore too much.
        # Let's try to force the first few picks from the greedy solution?
        # That's complex.
        
        try:
            if search(0, set(), 0):
                return True
        except TimeoutError:
            pass
            
        return False

    # Try to improve
    # We know greedy gave us ~195.
    # We try 194, 193...
    
    # To save time, we only try a few steps.
    # If best_size is e.g. 195, we try 194.
    # If it finds one, we update and try 193.
    
    # However, B&B from scratch (empty set) is very slow.
    # A better approach: Start from the Greedy Solution and try to *remove* one.
    # This is "Local Search" again.
    
    # Given the constraints, the Greedy + Removal is the most reliable "strong" algorithm I can write in Python here.
    # The B&B from scratch is likely to time out without finding a solution better than Greedy.
    
    # So, I will return the result of Greedy + Removal.
    
    return [all_words[i] for i in best_code_indices]

# Ensure the main entry point is construct
# The code above defines construct.

if __name__ == "__main__":
    # Quick test
    code = construct(3, 7, 1)
    print(f"Found {len(code)} codewords")

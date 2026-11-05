"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

# Increase recursion limit just in case, though we use iterative loops
sys.setrecursionlimit(10000)

_start_time = time.time()
_DEADLINE = 115.0  # Safety margin before 120s kill (evaluator limit)

def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode (9 words len 4, perfect radius 1) with full tails."""
    if q != 3 or n < 4:
        return None
    # The ternary tetracode is a perfect code of length 4 and radius 1 with 9 codewords.
    # Words are (a, b, a+b, a+2b) mod 3.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = []
    for h in tetracode:
        # The remaining n-4 positions can be anything (full space)
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words

def _idx_to_word(idx, q, n):
    """Convert integer index to word list."""
    word = [0] * n
    temp = idx
    for i in range(n - 1, -1, -1):
        word[i] = temp % q
        temp //= q
    return word

def _word_to_idx(word, q, n):
    """Convert word list to integer index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx

def _compute_neighbors_bitmasks(q, n, radius):
    """
    Precompute the set of neighbors (including self) for every word in the space.
    Returns a list of integers where each integer is a bitmask of the neighbors.
    """
    space_size = q ** n
    neighbors = [0] * space_size
    
    # Precompute powers of q for index calculation
    powers = [1] * (n + 1)
    for i in range(1, n + 1):
        powers[i] = powers[i - 1] * q

    for idx in range(space_size):
        # Get digits
        digits = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
            
        mask = 1 << idx
        
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                p = powers[n - 1 - pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_idx = idx + (v - orig) * p
                    mask |= 1 << new_idx
        
        neighbors[idx] = mask
        
    return neighbors

def _greedy_cover_bitset(q, n, radius, deadline, random_seed=None):
    """
    Greedy set cover using bitmasks.
    """
    if random_seed is not None:
        random.seed(random_seed)
    else:
        random.seed(int(time.time() * 1000) % 10000)

    space_size = q ** n
    neighbors = _compute_neighbors_bitmasks(q, n, radius)
    
    # uncovered: bitmask of all words not yet covered
    all_mask = (1 << space_size) - 1
    uncovered = all_mask
    
    # To speed up greedy, we can maintain a list of candidate indices.
    # A simple greedy: pick the codeword that covers the most uncovered words.
    # Since calculating "most uncovered" from scratch is O(N^2/word_size), 
    # and N=6561, N^2 is ~43M bits. 43M/64 ~ 675k 64-bit ops. This is fast in Python big ints.
    
    selected = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
            
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = 0
        
        # Optimization: We can iterate through all words, but that's 6561 iterations.
        # Each iteration does a bitwise AND and bit count.
        # Python's int.bit_count() is very fast (C implementation).
        for idx in range(space_size):
            # Count how many uncovered words are covered by idx
            count = (neighbors[idx] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
                # Early exit if we found a perfect cover (radius 1, q=3, n=8, max cover is 1+8*2=17)
                # Actually max cover is 1 + n*(q-1) = 1 + 8*2 = 17.
                if best_count == 1 + n * (q - 1):
                    break
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        uncovered &= ~neighbors[best_idx]
        
    return selected

def _prune_bitset(q, n, radius, selected, deadline):
    """
    Prune redundant codewords.
    A codeword is removable if its neighbors are all covered by other codewords.
    """
    if not selected:
        return selected
        
    space_size = q ** n
    neighbors = _compute_neighbors_bitmasks(q, n, radius)
    
    # Map from index to its position in selected list for quick removal?
    # Or just keep a list and remove from end?
    # Removing from a list is O(N). N is ~500. 500*500 = 250k ops. Fine.
    
    # We will iterate and check removability.
    # To be safe, we do multiple passes or just one pass from end to start?
    # One pass from end to start is sufficient for "any" removable, but might miss
    # cascades. However, greedy usually produces a minimal or near-minimal set.
    # Let's do a simple pass.
    
    current_selected = selected[:]
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Iterate backwards to allow safe removal
        for i in range(len(current_selected) - 1, -1, -1):
            idx = current_selected[i]
            # Check if neighbors[idx] is subset of union of neighbors of others
            # Union of others = All covered by current_selected minus idx?
            # No, we need to check if every word in neighbors[idx] is covered by some j != i.
            
            # Construct mask of coverage by all OTHERS
            # This is O(N) to construct.
            # Alternative: 
            # covered_by_others = 0
            # for j in current_selected:
            #    if j != i: covered_by_others |= neighbors[j]
            # if (neighbors[idx] & ~covered_by_others) == 0: remove
            
            # Optimization: 
            # If we remove i, do we lose any coverage?
            # We lose coverage for words w in neighbors[idx] such that no other j covers w.
            
            # Let's just compute the union of others.
            # Since N is small (~500), this is fast.
            others_mask = 0
            for j in current_selected:
                if j == i:
                    continue
                others_mask |= neighbors[j]
                
            # If neighbors[idx] is fully covered by others_mask, i is redundant.
            if (neighbors[idx] & ~others_mask) == 0:
                current_selected.pop(i)
                changed = True
                # Restart loop? Or continue? 
                # Removing one might make another removable.
                # Breaking and restarting is safer and often faster if many are removed.
                break
                
    return current_selected

def _local_search_swap(q, n, radius, selected, deadline):
    """
    Try to improve the solution by swapping one codeword with another.
    Or removing one if possible (handled by prune).
    Here we focus on 1-exchange: replace c_i with c_j to reduce size? No, size is fixed in 1-exchange.
    We want to find a solution with FEWER codewords.
    So we look for a set of k codewords that can be replaced by k-1 codewords.
    This is hard.
    
    Standard approach for covering codes:
    1. Start with greedy.
    2. Prune.
    3. Local Search: 
       - Try to remove a codeword and fix the holes with 1 new codeword? (Net 0 change, might enable further removals)
       - Try to find a "hole" (uncovered word) and see if we can cover it by moving a codeword?
       
    Actually, a common heuristic is:
    - Identify a codeword c.
    - Remove it.
    - Identify the set of uncovered words U.
    - If U can be covered by a single new codeword c', replace c with c'. (Size same, but might help prune?)
    - If U is empty, remove c. (Size -1)
    
    Better: 
    - Try to find a pair (c_i, c_j) that can be replaced by a single c_k? (Size -1)
    - This is "2-to-1" replacement.
    
    Let's implement a simple "1-exchange" that tries to find a better codeword for a position,
    potentially enabling a prune.
    
    Actually, the most effective simple local search for set cover is:
    1. Prune.
    2. For each codeword c:
       a. Remove c.
       b. Let U be the uncovered set.
       c. Find the best codeword c' that covers the most of U.
       d. Add c'.
       e. Prune.
       f. If size decreased, keep. Else revert.
       
    This is O(N^2) per step. N~500. 250k * (cost of prune).
    Prune is O(N^2). So O(N^4). 500^4 = 62.5B. Too slow.
    
    We need a faster local search.
    
    Let's try "2-exchange":
    Remove c_i, c_j.
    Uncovered set U.
    If U can be covered by 1 codeword c_k, replace {c_i, c_j} with {c_k}.
    
    This is also expensive to check for all pairs.
    
    Alternative: 
    Use the "greedy" logic again but on the "holes" created by removing a codeword.
    
    Let's stick to a simpler strategy for this time budget:
    Iterated Local Search with random restarts of the Greedy algorithm.
    The greedy algorithm is deterministic given the tie-breaking.
    If we randomize the tie-breaking in greedy, we might find different local minima.
    
    Let's modify _greedy_cover_bitset to accept a random tie-breaker.
    """
    # We will implement a "randomized greedy" and run it multiple times.
    # Then pick the best.
    
    best_selected = selected
    best_size = len(selected)
    
    # We have limited time. Let's do a few random restarts.
    # Each greedy run takes ~0.1-0.5s?
    # 6561 words. 500 iterations. 500 * 6561 * bit_count.
    # bit_count is fast. 500 * 6561 ~ 3M ops. Should be < 0.1s.
    # So we can do ~50-100 restarts in 10s.
    
    start_time_ls = time.time()
    time_limit_ls = deadline - 5
    attempts = 0
    
    while time.time() < time_limit_ls:
        attempts += 1
        # Randomize the order of candidates in greedy
        # We can do this by adding a small random noise to the count?
        # Or just shuffling the range?
        
        # Let's implement a randomized greedy directly here for clarity
        space_size = q ** n
        neighbors = _compute_neighbors_bitmasks(q, n, radius)
        all_mask = (1 << space_size) - 1
        uncovered = all_mask
        sel = []
        
        while uncovered:
            if time.time() > time_limit_ls:
                break
            
            best_idx = -1
            best_score = -1
            
            # Randomized selection:
            # Pick a random subset of candidates? No, we must pick the best.
            # To break ties randomly, or to explore, we can pick a random word from the top K?
            # Or just standard greedy with random tie-breaking.
            
            # Standard greedy with random tie-breaking:
            # 1. Find max count.
            # 2. If multiple have max count, pick one randomly.
            
            # Optimization: 
            # Instead of scanning all 6561, we can just scan all.
            
            max_count = 0
            candidates = []
            
            for idx in range(space_size):
                count = (neighbors[idx] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [idx]
                elif count == max_count and count > 0:
                    candidates.append(idx)
            
            if not candidates:
                break
                
            # Pick randomly from candidates
            best_idx = random.choice(candidates)
            sel.append(best_idx)
            uncovered &= ~neighbors[best_idx]
            
        # Prune the result
        sel = _prune_bitset(q, n, radius, sel, time_limit_ls)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = sel
            # If we hit the theoretical lower bound or known optimal, we can stop
            if best_size <= 486:
                break
                
    return best_selected

def _verify_coverage(q, n, radius, selected):
    """Verify that all words are covered by the selected codewords."""
    space_size = q ** n
    neighbors = _compute_neighbors_bitmasks(q, n, radius)
    covered = 0
    for idx in selected:
        covered |= neighbors[idx]
    all_mask = (1 << space_size) - 1
    return covered == all_mask

def construct(q, n, radius):
    global _start_time
    _start_time = time.time()
    deadline = _start_time + 110.0

    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        # Generic fallback: return empty or tetracode-like
        return []

    # 1. Initial Greedy Cover
    selected = _greedy_cover_bitset(q, n, radius, deadline)
    
    # 2. Prune
    selected = _prune_bitset(q, n, radius, selected, deadline)
    
    # 3. Verify
    if not _verify_coverage(q, n, radius, selected):
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        return []
        
    # 4. Local Search / Iterated Local Search
    if time.time() < deadline - 10:
        selected = _local_search_swap(q, n, radius, selected, deadline)
        
    # 5. Final Prune and Verify
    selected = _prune_bitset(q, n, radius, selected, deadline)
    
    if not _verify_coverage(q, n, radius, selected):
        # If something went wrong, fallback to the previous valid solution?
        # We didn't save it. Let's just return tetracode.
        tc = _tetracode_solution(q, n, radius)
        if tc:
            return tc
        return []

    # Convert indices to words
    result = [_idx_to_word(idx, q, n) for idx in selected]
    return result

# EVOLVE-BLOCK-END

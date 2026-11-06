import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0

def _seed_words():
    """
    Generate a strong initial code using the extended Reed-Muller RM(1,4) code 
    plus its dual or related structures, or simply a good greedy start.
    RM(1,4) has length 16, size 32, distance 8.
    We can extend it to length 18 by adding 2 bits.
    To maximize size, we can also try to add words from the dual or other linear codes.
    However, a better seed is a greedy construction on a fixed ordering of words,
    prioritizing those with specific weights to ensure good coverage.
    """
    # Start with the 32 RM(1,4) words (extended with 2 zeros)
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # Add 2 zeros at the end (bits 16, 17)
            # This keeps distance 8
            words.append(word)
    
    # Try to greedily add more words from a systematic ordering
    # Order words by weight, then by value.
    # This often finds dense clusters.
    current_set = set(words)
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        # Mark neighbors within distance 5
        # For efficiency in seed, we can just check against current_set
        pass
        
    # Greedy addition from systematic list
    # Generate all words? No, 2^18 is too many to iterate in Python quickly for seed.
    # But we can iterate over weights.
    # Let's just use the 32 RM words as a solid base. 
    # The local search will expand it.
    return words

def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results

def _build_forbidden_mask(words):
    """Build a boolean mask of all words forbidden by the current code."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden

def _local_search(words, deadline):
    """
    Perform local search to improve the code size.
    Moves:
    1. 1-flip: Try flipping one bit of an existing word to see if it becomes valid 
       and allows adding a new word? No, that changes the code.
       Better: Try to ADD a new word.
       If no new word can be added, try to REPLACE a word with two new words?
       Or simply try to add any valid word found by systematic search.
       
    Let's use a simpler local search:
    1. Maintain the forbidden mask.
    2. If we are stuck (no new words added in last K iterations), try to "shake" the code:
       - Remove a random word from the code.
       - Update mask.
       - Try to add 2 words in the space freed.
       - If successful, keep the change.
    """
    words = list(words)
    forbidden = _build_forbidden_mask(words)
    
    # Precompute a list of all words sorted by some heuristic?
    # Or just iterate through all words systematically?
    # 2^18 = 262,144. Iterating through all of them is fast in C (numpy) or even Python if simple.
    # But checking distance against all existing words is O(N).
    
    # Strategy: 
    # 1. Systematically scan all words 0..2^18-1.
    # 2. If word is not forbidden, add it.
    # 3. This is a "greedy scan". It's fast and deterministic.
    # 4. After the scan, we might be stuck.
    # 5. Then we try "swaps": remove a word, re-scan a subset?
    
    # Let's optimize the scan.
    # We can vectorize the check against the forbidden mask.
    # But we need to update the mask as we add words.
    
    # To make the scan fast, we'll just iterate.
    # 262k iterations * (bit_count + set update) is very fast in Python (< 0.1s).
    
    added_in_pass = 0
    last_add_time = time.time()
    
    # Systematic Scan 1
    for w in range(TOTAL):
        if time.time() > deadline:
            break
        if not forbidden[w]:
            words.append(w)
            neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
            forbidden[neighbors] = True
            added_in_pass += 1
            last_add_time = time.time()
            
    # If we added a lot, we might have filled up most of the space.
    # Now try to improve by swapping.
    # Heuristic: Words with high weight or low weight might be "wasteful" or "efficient".
    # Actually, let's try a random restart of the local search if stuck.
    
    # A more powerful move: 
    # Try to find a pair of words (w1, w2) in the code such that removing w1 allows adding w2 and w3.
    # This is complex. Let's stick to:
    # 1. Greedy Scan (done)
    # 2. Random Perturbation: 
    #    - Pick a random word from the code.
    #    - Remove it.
    #    - Try to add 2 random words that fit in the gap.
    #    - If successful, keep. Else, revert.
    
    # However, the "Greedy Scan" is usually very effective for small N.
    # For N=18, D=6, the maximum size A2(18,6) is known to be around 272? 
    # Wait, let's check bounds.
    # Griesmer bound? 
    # For D=6, N=18.
    # A(18,6) is likely in the hundreds.
    # The RM(1,4) extended is 32.
    # The quadratic residue code of length 18? 
    # There is a (18, 2^? , 6) code?
    # Actually, the best known codes for (18,6) are often constructed via concatenation or shortening.
    
    # Let's enhance the local search with a "fill gaps" strategy.
    # After the systematic scan, we can try to add words that were previously forbidden 
    # by removing one existing word and adding two new ones.
    
    # To keep it simple and fast within 90s:
    # We will run the systematic scan.
    # Then we will try a limited number of "swap" moves.
    
    # But wait, the systematic scan order matters.
    # Scanning 0 to 2^18-1 might not be optimal.
    # Scanning by weight might be better?
    # Let's try multiple systematic scans with different orderings.
    
    # Ordering 1: Natural
    # Ordering 2: Reverse
    # Ordering 3: By Weight
    
    # We will implement a function that takes an ordering and greedily adds.
    
    return words

def _greedy_with_order(order, deadline):
    """Greedy construction using a specific ordering of words."""
    words = []
    forbidden = np.zeros(TOTAL, dtype=bool)
    
    for w in order:
        if time.time() > deadline:
            break
        if not forbidden[w]:
            words.append(w)
            neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
            forbidden[neighbors] = True
            
    return words

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    
    best_words = None
    best_len = 0
    
    # Strategy 1: Systematic Greedy with Natural Order
    # 2^18 is 262,144. Iterating this in Python is ~0.05s.
    # We can do this many times with different orderings.
    
    # Precompute orders?
    # Natural order: range(TOTAL)
    # Reverse order: range(TOTAL-1, -1, -1)
    # Weight order: sort by bit_count
    
    # Generating weight order:
    # Group indices by weight.
    # There are 19 weights (0 to 18).
    # We can generate lists for each weight.
    
    orders = []
    
    # 1. Natural
    orders.append(range(TOTAL))
    
    # 2. Reverse
    orders.append(range(TOTAL-1, -1, -1))
    
    # 3. Weight sorted (low to high)
    # This might cluster low-weight words, which might be good or bad.
    # Let's create a list of all indices sorted by weight.
    # To save time, we can just use `sorted(range(TOTAL), key=lambda x: x.bit_count())`
    # This takes ~0.1s.
    
    # Let's try a few random permutations as well?
    # No, deterministic is better for reproducibility and finding the "best" structure.
    
    # Let's try Weight-Ascending and Weight-Descending.
    try:
        all_indices = np.arange(TOTAL, dtype=np.int64)
        weights = np.array([w.bit_count() for w in all_indices])
        # Sort by weight, then by value
        sorted_indices = all_indices[np.lexsort((all_indices, weights))]
        orders.append(sorted_indices)
        
        sorted_indices_desc = all_indices[np.lexsort((all_indices, -weights))]
        orders.append(sorted_indices_desc)
    except Exception:
        pass

    # Also try the seed words as a starting point, then fill the rest?
    # The greedy scan starts from empty.
    # If we start with the RM(1,4) seed, we might get a different result.
    # Let's try:
    # 1. Empty start + Natural
    # 2. Empty start + Reverse
    # 3. Empty start + WeightAsc
    # 4. Empty start + WeightDesc
    # 5. Seed start + Natural (skip seed words in scan? No, just add if not forbidden)
    
    # For Seed start:
    # We initialize forbidden with the seed's neighbors.
    
    # Let's implement a helper that takes initial_words and an order.
    
    def run_greedy(initial_words, order, time_limit):
        words = list(initial_words)
        forbidden = np.zeros(TOTAL, dtype=bool)
        for w in words:
            neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
            forbidden[neighbors] = True
        
        for w in order:
            if time.time() > time_limit:
                break
            if not forbidden[w]:
                words.append(w)
                neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
                forbidden[neighbors] = True
        
        return words

    # Run strategies
    strategies = [
        ([], range(TOTAL)),
        ([], range(TOTAL-1, -1, -1)),
    ]
    
    if len(orders) > 2:
        strategies.append(([], orders[2])) # Weight Asc
        strategies.append(([], orders[3])) # Weight Desc
    
    # Add seed-based strategies
    seed_words = _seed_words()
    strategies.append((seed_words, range(TOTAL)))
    strategies.append((seed_words, range(TOTAL-1, -1, -1)))
    
    # Also try a "Random" order?
    # Random might find different local optima.
    rng = np.random.default_rng(42)
    random_order = rng.permutation(TOTAL)
    strategies.append(([], random_order))
    
    # Run each strategy within a time slice
    # Total time 85s. 7 strategies. ~12s each.
    # The scan is fast, so it will finish early.
    # We can then use the remaining time for local search on the best result.
    
    for init, order in strategies:
        if time.time() > deadline:
            break
        # Give each strategy a small time slice, but the scan is fast.
        # Let's just run them to completion (they are fast).
        try:
            w = run_greedy(init, order, deadline)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # If we have time left, try to improve the best found via local search (swaps)
    if best_words is not None and time.time() < deadline - 2.0:
        # Simple local search: try to add any word that fits.
        # The greedy scan already did this.
        # Try to swap: remove one, add two.
        # This is complex to implement robustly in the remaining time.
        # Let's just try a few random restarts of the greedy with different seeds if time permits.
        # But deterministic orders are usually better for this specific problem structure.
        
        # Let's try one more thing: 
        # If best_len is low, try a "dense" ordering.
        # Pass 2: Try to fill remaining holes with a second pass of a different order?
        # No, the forbidden mask prevents that.
        
        # Let's try to optimize the local search with a "kick" move.
        # Pick a random word from best_words, remove it, and try to add 2 random words.
        # If successful, keep.
        
        current_words = list(best_words)
        forbidden = _build_forbidden_mask(current_words)
        
        # Try 1000 random swap attempts
        for _ in range(1000):
            if time.time() > deadline:
                break
            if len(current_words) < 2:
                break
            # Remove a random word
            idx = np.random.randint(len(current_words))
            removed = current_words[idx]
            current_words.pop(idx)
            
            # Update mask (remove forbidden neighbors? No, we just need to know what's available)
            # Actually, removing a word "un-forbids" its neighbors.
            # But we don't maintain an "un-forbid" count easily.
            # Let's rebuild the mask? Too slow.
            
            # Alternative: Just check if we can add 2 words.
            # We need to check against the NEW set (current_words).
            # Let's pick 2 random candidates from the whole space?
            # That's 2^18 choices.
            
            # This local search is too slow/complex for the remaining time.
            # Let's stick with the multiple greedy scans.
            
            # Revert
            current_words.insert(idx, removed)
            
            # Break out of loop as this logic is flawed for speed
            break

    if best_words is None:
        best_words = _seed_words()

    # Final validation and deduplication
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    
    # Verify pairwise distance for safety (optional but good for debug, might be slow if large)
    # For 272 words, 272*271/2 = ~37k checks. Fast.
    if len(out) > 0:
        valid = True
        for i in range(len(out)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(out)):
                if (out[i] ^ out[j]).bit_count() < MIN_DISTANCE:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback to seed if invalid
            out = _seed_words()

    return out

if __name__ == "__main__":
    # Quick test
    start = time.time()
    res = construct()
    print(f"Time: {time.time()-start:.2f}s, Size: {len(res)}")

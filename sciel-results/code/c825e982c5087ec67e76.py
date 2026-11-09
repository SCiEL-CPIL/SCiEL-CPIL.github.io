"""
Construct a large binary code of length 17 with minimum distance 6.
Strategy: Start from the known 256-word linear code, then use a structured
local search to replace codewords with non-linear ones, aiming for 258 words.
"""
import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _generate_linear_code_256():
    """
    Generate the 256-word [17,8,6] linear code.
    We construct it by finding 8 basis vectors such that all non-zero combinations have weight >= 6.
    We use a greedy approach to find such a basis, then generate all 2^8 combinations.
    """
    # We need 8 vectors v1...v8 in F_2^17 such that any non-zero linear combination has weight >= 6.
    # This is equivalent to finding a [17,8,6] code.
    # Let's try to construct it using a systematic approach.
    # We can start with the RM(1,4) code extended, but that's only dim 5.
    # Actually, the problem statement says the 256-word linear reference is the [18,9,6] QR code shortened.
    # Wait, [18,9,6] has 2^9 = 512 words. Shortened by one coordinate gives [17,8,6] if the removed coordinate is not always 0?
    # No, shortening a [n,k,d] code by removing one coordinate and keeping only words with 0 in that coordinate gives [n-1, k-1, d] or [n-1, k, d-1].
    # If we remove a coordinate and keep words with 0 there, we get a subcode of dimension k-1 (if the coordinate is not zero for all words).
    # So [18,9,6] -> remove coord 17, keep words with bit 17 = 0 -> [17,8,6].
    # So we need to construct the [18,9,6] QR code first.
    
    # Quadratic Residue code of length 18.
    # QR(18) is a [18,8,6] code? Or [18,9,6]?
    # For n=18, the QR code is [18,8,6] (non-extended) and [19,9,7] (extended).
    # The problem says [18,9,6]. This might be a different code or I'm misremembering.
    # Let's just construct a [17,8,6] code directly using greedy basis search.
    
    basis = []
    # We want 8 vectors.
    # To make it easier, let's try to find them.
    # A known [17,8,6] code is the shortened QR code.
    # Let's try to build it.
    
    # Heuristic: Start with vectors that are spread out.
    # We will use a random restart greedy search to find 8 vectors.
    
    def try_find_basis(rng):
        b = []
        # First vector: weight 8 or 9? Min weight 6.
        # Let's pick a random vector of weight 8.
        while len(b) < 8:
            # Generate random vector
            v = 0
            # Try weight 8
            positions = list(range(N))
            rng.shuffle(positions)
            for p in positions[:8]:
                v |= (1 << p)
            
            # Check if v is independent of current basis
            # And if v + any subset of b has weight >= 6
            # Actually, it's easier to just generate the code from the basis found so far
            # and check if adding v maintains min distance 6.
            
            # Current code C from basis b
            # New code C' = C union {c + v for c in C}
            # We need min weight in C' to be >= 6.
            # Since C already has min weight 6, we just need to check the new words c+v.
            # Also v itself must have weight >= 6 (it does, weight 8).
            
            current_words = [0]
            for bv in b:
                new_words = [w ^ bv for w in current_words]
                current_words.extend(new_words)
            
            ok = True
            for w in current_words:
                nw = w ^ v
                if nw.bit_count() < 6:
                    ok = False
                    break
            
            if ok:
                b.append(v)
        
        return b
    
    # Try multiple random restarts
    for attempt in range(1000):
        rng = random.Random(attempt)
        b = try_find_basis(rng)
        if len(b) == 8:
            # Generate all 256 words
            words = [0]
            for bv in b:
                new_words = [w ^ bv for w in words]
                words.extend(new_words)
            return words
    
    # Fallback: Return RM(1,4) extended (32 words) if we fail
    # This shouldn't happen
    return [0]

def _precompute_deltas():
    """Precompute all vectors with weight < MIN_DISTANCE."""
    deltas = []
    for d in range(1 << N):
        if d.bit_count() < MIN_DISTANCE:
            deltas.append(d)
    return np.array(deltas, dtype=np.uint32)

def _construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    # Get the 256-word linear code
    code = _generate_linear_code_256()
    if not code:
        return [0]
    
    code_set = set(code)
    best_code = list(code)
    best_size = len(best_code)
    
    # Precompute deltas for fast forbidden zone checks
    deltas = _precompute_deltas()
    
    # Forbidden array: forbidden[w] is True if w is within distance < 6 of any word in code
    forbidden = np.zeros(1 << N, dtype=bool)
    for w in code:
        forbidden[w ^ deltas] = True
    
    # Local Search: Try to replace words to increase density or maintain size with better structure
    # Actually, the goal is to ADD words. The linear code is maximal in its linear subcode,
    # but we can add non-linear words if we remove some linear words.
    
    # Strategy: 
    # 1. Try to add a word w not in code. If it conflicts, remove a conflicting word.
    #    This is a "swap" or "replace" strategy.
    # 2. We want to end up with > 256 words.
    
    # Let's try a greedy replacement strategy.
    # Iterate through all possible words w (sorted by weight or random).
    # If w is not forbidden, add it.
    # If w is forbidden, it conflicts with some words in code.
    # If we remove one conflicting word, does w become valid?
    # This is complex.
    
    # Alternative: 
    # The known bound is 258. The linear code is 256.
    # Milshtein's 258-word code is nonlinear.
    # It is likely obtained by taking the 256-word linear code and swapping 2 words for 4 new words?
    # Or swapping 1 word for 3? Net gain +2.
    
    # Let's try to find a set of words that can be removed to allow adding more words.
    
    # We will perform a local search on the set of codewords.
    # Move: Remove a word w1, add a word w2. If w2 was forbidden by w1 and no one else, this is valid.
    # But we want to INCREASE the count.
    # Move: Remove w1, add w2, add w3. If w2 and w3 are compatible with each other and with code\{w1}, this is valid.
    # This increases count by 1.
    
    # Let's implement a "1-for-2" swap search.
    # For each word w1 in code:
    #   Temporarily remove w1.
    #   Find all words w2 that are now available (not forbidden by code\{w1}).
    #   Among these w2, find pairs (w2, w3) that are compatible.
    #   If found, replace w1 with w2, w3.
    
    # This is expensive to do naively.
    # Optimization: 
    # When we remove w1, the forbidden set shrinks.
    # New available words are those in (w1 ^ deltas) that were previously forbidden ONLY by w1.
    # This is hard to track incrementally without a reference count array.
    
    # Let's use a reference count array: ref_count[x] = number of words in code that are within distance < 6 of x.
    # A word x is available if ref_count[x] == 0.
    
    ref_count = np.zeros(1 << N, dtype=np.int32)
    for w in code:
        ref_count[w ^ deltas] += 1
    
    # Available words are those with ref_count == 0.
    # Initially, for the 256-word linear code, are there any available words?
    # The linear code is a [17,8,6] code. It is not necessarily perfect or maximal.
    # If there are available words, we can just add them!
    
    available = np.nonzero(ref_count == 0)[0]
    if len(available) > 0:
        # Add all available words greedily
        # But adding one might make others unavailable.
        # We should add them one by one, updating ref_count.
        # To maximize gain, we might want to add words that "block" the least others?
        # Or just add them all if they don't conflict?
        # If they are all available, they are pairwise distance >= 6?
        # No. Available means distance >= 6 from ALL words in current code.
        # But two available words might be distance < 6 from each other.
        # So we must add them one by one.
        
        # Sort available words by some heuristic?
        # Let's just pick them randomly or in order.
        rng = random.Random(42)
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        for w in avail_list:
            if time.time() > time_deadline - 5:
                break
            if ref_count[w] == 0:
                # Add w
                code.append(w)
                best_size += 1
                # Update ref_count
                ref_count[w ^ deltas] += 1
    
    # Now try 1-for-2 swaps to improve further.
    # We need to be careful with time.
    
    # We will iterate through words in the code.
    # For each word w1, check if removing it allows adding 2 new words.
    
    # To speed this up, we can precompute for each w1, the set of words that become available.
    # This is (w1 ^ deltas) intersect (words with ref_count == 1).
    # If ref_count[x] == 1, then x is forbidden by exactly one word. If that word is w1, then x becomes available.
    
    # Let's implement this.
    
    improved = True
    while improved and time.time() < time_deadline - 10:
        improved = False
        # Shuffle code to avoid bias?
        # We iterate over a snapshot of the code
        code_snapshot = list(code)
        
        for i, w1 in enumerate(code_snapshot):
            if time.time() > time_deadline - 5:
                break
            
            # Check if w1 is still in code (it might have been removed in a previous swap in this loop? 
            # No, we are modifying 'code' in place. If we remove w1, it's gone.
            # We should check if w1 is still there.
            if w1 not in code_set:
                continue
            
            # Find words that become available if we remove w1.
            # These are x in (w1 ^ deltas) such that ref_count[x] == 1.
            # And the only word forbidding x is w1.
            
            # We can check this by:
            # 1. Get candidates = w1 ^ deltas
            # 2. Filter candidates where ref_count[c] == 1
            # 3. For each such c, verify that w1 is indeed the one forbidding it.
            #    How? We can't easily know which word forbids it without storing it.
            #    Alternative: Just try to add them.
            
            # Let's get the list of potential new words.
            candidates = []
            for d in deltas:
                x = w1 ^ d
                if ref_count[x] == 1:
                    # It's forbidden by exactly 1 word.
                    # Is it w1?
                    # We can check if x is in (w1 ^ deltas). Yes, by construction.
                    # But is w1 the ONLY one?
                    # If ref_count[x] == 1, and x is in w1's neighborhood, then w1 is a candidate.
                    # Could it be forbidden by another word w2?
                    # If w2 forbids x, then x is in w2 ^ deltas.
                    # If ref_count[x] == 1, then exactly one word forbids it.
                    # If w1 forbids it, then w1 is that word.
                    # So yes, if ref_count[x] == 1 and x is in w1's neighborhood, then w1 is the unique forbider.
                    candidates.append(x)
            
            if len(candidates) < 2:
                continue
            
            # We need to find 2 words in candidates that are mutually compatible (distance >= 6).
            # And also compatible with the rest of the code (which they are, since ref_count becomes 0).
            
            # Sort candidates to find pairs faster?
            # Just check all pairs.
            found_pair = False
            cand_list = list(set(candidates)) # Remove duplicates
            
            # Heuristic: try pairs with higher weight? Or just first two compatible?
            for j in range(len(cand_list)):
                for k in range(j+1, len(cand_list)):
                    w2 = cand_list[j]
                    w3 = cand_list[k]
                    
                    if _hamming_dist(w2, w3) >= 6:
                        # We can swap w1 with w2, w3
                        # Remove w1
                        code.remove(w1)
                        code_set.remove(w1)
                        ref_count[w1 ^ deltas] -= 1
                        
                        # Add w2
                        code.append(w2)
                        code_set.add(w2)
                        ref_count[w2 ^ deltas] += 1
                        
                        # Add w3
                        code.append(w3)
                        code_set.add(w3)
                        ref_count[w3 ^ deltas] += 1
                        
                        best_size += 1
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break
            
            if found_pair:
                # Restart loop? Or continue?
                # Since ref_counts changed, the snapshot is stale.
                # Break inner loop and restart outer while loop.
                break
    
    # Final verification
    valid_words = []
    seen = set()
    for w in code:
        if w in seen:
            continue
        seen.add(w)
        ok = True
        for v in valid_words:
            if _hamming_dist(w, v) < 6:
                ok = False
                break
        if ok:
            valid_words.append(w)
    
    return valid_words

def construct():
    return _construct()

if __name__ == "__main__":
    words = construct()
    print(f"Found {len(words)} words")

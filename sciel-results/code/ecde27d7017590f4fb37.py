"""
Strategy: Stochastic Local Search (Simulated Annealing) for Code Packing
Instead of a one-pass greedy algorithm which gets stuck in local optima,
we use a hill-climbing approach with random restarts and "moves" that
remove a random codeword and attempt to insert a new one (or a different one)
to improve the packing density. This allows the code to escape local maxima
by temporarily sacrificing a codeword to free up space for a better arrangement,
or simply by finding a new codeword that fits in the "gaps" left by the
removal of another, which greedy cannot do once the initial fill is complete.

Key structural change:
1. Seed with a high-quality algebraic code (Golay shortening).
2. Use a "forbidden zone" bitmap for O(1) validity checks.
3. Perform iterations of: 
   a. Pick a random existing word.
   b. Temporarily remove it (clear its forbidden zone).
   c. Try to insert a new random word.
   d. If successful, keep the new word (size increases or stays same, but structure changes).
   e. If not, try to re-insert the original word.
   This "swap" mechanism allows the code to explore the state space of codeword
   combinations, not just the monotonic addition of new words.
"""
import time
import random
import numpy as np
from itertools import combinations

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

N = 18
MIN_DIST = 6
MASK_SPACE = 1 << N

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _generate_golay_code():
    """
    Generate the [24,12,8] binary Golay code.
    """
    qr_set = set((i * i) % 23 for i in range(1, 23))
    qr_set.discard(0)
    
    M = np.zeros((23, 23), dtype=int)
    for i in range(23):
        for j in range(23):
            if (j - i) % 23 in qr_set:
                M[i, j] = 1
    
    rows = [list(row) for row in M]
    basis = []
    for r in rows:
        pivot = -1
        for j in range(23):
            if r[j] == 1:
                pivot = j
                break
        if pivot == -1:
            continue
        for b in basis:
            if b[pivot] == 1:
                for k in range(23):
                    r[k] ^= b[k]
        if any(r):
            basis.append(r)
            if len(basis) == 12:
                break
    
    code_23 = [0]
    for b in basis:
        new_words = []
        for w in code_23:
            bw = 0
            for k in range(23):
                if b[k]:
                    bw |= (1 << k)
            new_words.append(w ^ bw)
        code_23.extend(new_words)
    
    code_24 = []
    for w in code_23:
        parity = bin(w).count('1') % 2
        w_24 = (w << 1) | parity
        code_24.append(w_24)
        
    return code_24

def _shorten_golay_to_18():
    """
    Shorten the [24,12,8] Golay code to length 18.
    """
    golay_code = _generate_golay_code()
    
    # Fix positions 0..5 to 0
    mask = 0
    for p in range(6):
        mask |= (1 << p)
        
    shortened_code = []
    for w in golay_code:
        if (w & mask) == 0:
            w_18 = w >> 6
            shortened_code.append(w_18)
            
    if len(set(shortened_code)) != len(shortened_code):
        raise ValueError("Shortened code has duplicates")
        
    return shortened_code

def _reed_muller_extended():
    """First-order Reed-Muller on 16 coords extended by 2 zero bits -> 32 words, dist 8."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # 1. Construct initial seed
    try:
        base_words = _shorten_golay_to_18()
    except Exception:
        base_words = _reed_muller_extended()
        
    if not _check_min_distance(base_words, MIN_DIST):
        base_words = _reed_muller_extended()
        
    best_words = list(base_words)
    
    # Precompute masks for weight <= 5 (forbidden zone of radius 5)
    # This is used to quickly check if a candidate is valid.
    masks_w5 = []
    for w in range(6):
        for combo in combinations(range(N), w):
            m = 0
            for p in combo:
                m |= (1 << p)
            masks_w5.append(m)
            
    # 2. Local Search / Simulated Annealing Loop
    # We maintain a set of current words and a "forbidden" numpy array.
    # A move consists of:
    #   a. Pick a random word W_old from the current set.
    #   b. Remove W_old from the set and clear its forbidden zone.
    #   c. Try to insert a new random word W_new.
    #   d. If W_new is valid (not forbidden), add it. 
    #      If the set size increased (impossible in this specific move logic unless we allow 
    #      "add without remove" first) or just maintained, we keep the new configuration.
    #      Actually, a better move for *growth* is:
    #      1. Try to add a random word. If it fits, add it.
    #      2. If it doesn't fit, pick a random existing word, remove it, and try to add the random word.
    #      This "swap" allows us to change the topology of the code.
    
    current_words = list(base_words)
    forbidden = np.zeros(MASK_SPACE, dtype=bool)
    
    def add_word_to_state(w):
        current_words.append(w)
        for m in masks_w5:
            forbidden[w ^ m] = True
            
    def remove_word_from_state(w):
        current_words.remove(w)
        for m in masks_w5:
            forbidden[w ^ m] = False
            
    # Initialize state
    for w in current_words:
        for m in masks_w5:
            forbidden[w ^ m] = True
            
    best_size = len(current_words)
    best_snapshot = list(current_words)
    stagnation_count = 0
    max_stagnation = 5000 # Reset or restart if stuck
    
    while time.time() - start_time < DEADLINE:
        # Phase A: Try to add a new random word (Growth)
        found_new = False
        for _ in range(100): # Small batch of random tries
            cand = rng.randint(0, MASK_SPACE - 1)
            if not forbidden[cand]:
                add_word_to_state(cand)
                found_new = True
                if len(current_words) > best_size:
                    best_size = len(current_words)
                    best_snapshot = list(current_words)
                    stagnation_count = 0
                break
        
        if not found_new:
            # Phase B: Try a swap (Reconfiguration)
            # Pick a random existing word to remove
            if current_words:
                idx = rng.randint(0, len(current_words) - 1)
                w_old = current_words[idx]
                
                # Temporarily remove it
                remove_word_from_state(w_old)
                
                # Try to insert a new random word
                # We try a few candidates to see if any fit in the space freed
                inserted = False
                for _ in range(50):
                    cand = rng.randint(0, MASK_SPACE - 1)
                    if not forbidden[cand]:
                        add_word_to_state(cand)
                        inserted = True
                        # Size is same, but structure changed. 
                        # This doesn't update best_size, but might enable future additions.
                        break
                
                if not inserted:
                    # Re-insert the old word to restore state
                    # Note: We must ensure w_old is still valid (it was valid before, 
                    # and we only removed it, so space is more available, so it's valid)
                    add_word_to_state(w_old)
                else:
                    # We swapped w_old for cand.
                    # Check if this leads to a better solution later? 
                    # For now, we just accept the swap to explore.
                    # If we are stuck for a long time, we might want to restart from best_snapshot.
                    pass
            
            stagnation_count += 1
            if stagnation_count > max_stagnation:
                # Restart from best known configuration
                current_words = list(best_snapshot)
                forbidden = np.zeros(MASK_SPACE, dtype=bool)
                for w in current_words:
                    for m in masks_w5:
                        forbidden[w ^ m] = True
                stagnation_count = 0
                max_stagnation = int(max_stagnation * 1.5) # Increase patience slightly? Or keep constant.
                
    # Final check and return
    # Ensure the returned list is the best one found
    if len(best_snapshot) > len(current_words):
        final_words = best_snapshot
    else:
        final_words = current_words
        
    # Verify final validity
    if not _check_min_distance(final_words, MIN_DIST):
        # Fallback to seed if something went wrong
        return base_words
        
    return final_words

# Keep the helper functions accessible if needed by other parts of the system, 
# though construct() is the main entry point.
# The code above is self-contained.

if __name__ == "__main__":
    # Simple test
    words = construct()
    print(f"Found {len(words)} words")
    if _check_min_distance(words, 6):
        print("Valid")
    else:
        print("Invalid")

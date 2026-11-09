"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
import numpy as np
import heapq

N = 17
MIN_DISTANCE = 6
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _generate_reed_muller_seed():
    """Generate the 32-word first-order Reed-Muller code on 16 coords, extended by 0 bit."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit is 0
            words.append(word)
    return words

def _get_linear_code_seed():
    """
    Attempt to construct or retrieve a larger linear code seed.
    The [17,9,6] code is a shortened [18,9,6] QR code.
    Since constructing the specific QR code is complex and error-prone without a known generator matrix,
    we will use the RM(1,4) seed (32 words) but try to expand it aggressively.
    However, to improve the seed, we can try to find a [17, k, 6] code with k > 5.
    A simple approach: Use the RM seed and immediately run a fast greedy expansion to get a larger seed.
    """
    return _generate_reed_muller_seed()

def _precompute_deltas():
    """Precompute all vectors with weight <= MIN_DISTANCE - 1."""
    deltas = []
    for d in range(1 << N):
        if d.bit_count() < MIN_DISTANCE:
            deltas.append(d)
    return np.array(deltas, dtype=np.uint32)

def _greedy_expansion_priority(seed_words, deltas, rng, time_deadline):
    """
    Greedy expansion using a priority queue.
    Instead of scanning a random permutation, we maintain a set of 'candidate' words.
    Initially, candidates are all words.
    We pick the word that is 'furthest' from the current code? 
    Actually, a simpler heuristic: 
    1. Maintain a `forbidden` array.
    2. Instead of random scan, we can scan in a structured way or use multiple random scans.
    The current program uses random scans. 
    Let's try a different structural change: 
    **Simulated Annealing / Local Search** starting from a greedy solution.
    But first, let's make the greedy part faster and more effective by using 
    a 'best-first' search on the candidates? 
    No, the standard greedy is usually good enough if the seed is good.
    
    Let's stick to the fast vectorized greedy but run it multiple times with different seeds,
    and then apply a local search phase.
    """
    words = list(seed_words)
    universe_size = 1 << N
    forbidden = np.zeros(universe_size, dtype=bool)
    
    # Initialize forbidden for seed words
    for w in words:
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
    
    # We will use a random permutation but we will try to add as many as possible.
    # To avoid the "local minimum" trap of a single permutation, we will do this:
    # 1. Get a base greedy solution.
    # 2. Then try to improve it.
    
    indices = np.arange(universe_size)
    rng.shuffle(indices)
    
    for idx in indices:
        if time.time() > time_deadline - 10:
            break
            
        if forbidden[idx]:
            continue
            
        w = int(idx)
        words.append(w)
        
        # Update forbidden zone
        neighbors = (w ^ deltas).astype(np.uint32)
        forbidden[neighbors] = True
        
    return words

def _local_search_improve(words, deltas, rng, time_deadline):
    """
    Local search to improve the code size.
    Moves:
    1. Add a word if possible (shouldn't exist if greedy was exhaustive, but maybe after swaps).
    2. Remove one word, add two (1-for-2 swap).
    3. Remove two words, add three (2-for-3 swap).
    
    We will focus on 1-for-2 swaps as they are the most likely to increase size by 1.
    """
    words = list(words)
    universe_size = 1 << N
    
    def rebuild_forbidden(word_list):
        forbidden = np.zeros(universe_size, dtype=bool)
        for w in word_list:
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
        return forbidden
    
    best_words = words
    best_size = len(words)
    
    # Precompute a random order for candidate searches to be consistent but random
    candidate_order = np.arange(universe_size)
    
    # Limit the number of iterations to ensure we finish in time
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > time_deadline - 5:
            break
        if len(best_words) < 2:
            break
            
        # Pick a random word to remove
        remove_idx = rng.randint(0, len(best_words) - 1)
        removed_word = best_words[remove_idx]
        
        # Create temporary list without the removed word
        temp_words = best_words[:remove_idx] + best_words[remove_idx+1:]
        
        # Rebuild forbidden zone for temp_words
        forbidden = rebuild_forbidden(temp_words)
        
        # Try to find 2 new words
        added_words = []
        
        # Shuffle candidate order for this iteration
        rng.shuffle(candidate_order)
        
        for idx_cand in candidate_order:
            if len(added_words) == 2:
                break
            if forbidden[idx_cand]:
                continue
            
            w = int(idx_cand)
            # Check if w is already in temp_words? 
            # forbidden includes temp_words neighbors, so if w is in temp_words, it's forbidden.
            # So we are safe.
            
            # Add w to temp_words and update forbidden
            temp_words.append(w)
            added_words.append(w)
            neighbors = (w ^ deltas).astype(np.uint32)
            forbidden[neighbors] = True
            
        if len(added_words) == 2:
            # We successfully swapped 1 for 2
            new_size = len(temp_words)
            if new_size > best_size:
                best_words = temp_words
                best_size = new_size
        # Else, revert (best_words remains unchanged)
        
    return best_words

def construct():
    start_time = time.time()
    time_deadline = start_time + TIME_LIMIT
    
    rng = random.Random(42)
    
    # Precompute deltas
    deltas = _precompute_deltas()
    
    # Get initial seed
    seed = _generate_reed_muller_seed()
    
    best_words = seed
    best_size = len(seed)
    
    # Phase 1: Multiple Greedy Expansions
    # Run greedy expansion several times with different random seeds/orderings
    num_greedy_trials = 20
    
    for trial in range(num_greedy_trials):
        if time.time() > time_deadline - 15:
            break
        
        rng_trial = random.Random(1000 + trial)
        words_trial = _greedy_expansion_priority(seed, deltas, rng_trial, time_deadline - 5)
        
        if len(words_trial) > best_size:
            best_words = words_trial
            best_size = len(words_trial)
            
        # If we have a very large code, we might spend less time on further greedy trials
        if best_size > 200:
            # Reduce remaining trials or break? 
            # Let's just continue but break if time is tight.
            pass

    # Phase 2: Local Search Improvement
    if time.time() < time_deadline - 10 and len(best_words) > 50:
        rng_ls = random.Random(2000)
        improved_words = _local_search_improve(best_words, deltas, rng_ls, time_deadline - 5)
        if len(improved_words) > best_size:
            best_words = improved_words
            best_size = len(improved_words)
            
    # Phase 3: Final check and cleanup
    # Ensure all words are distinct and valid
    seen = set()
    valid_words = []
    for w in best_words:
        if w not in seen:
            seen.add(w)
            valid_words.append(w)
            
    # Verify distance constraint (optional, for safety, but might be slow if large)
    # Given the construction method, it should be valid.
    
    if len(valid_words) < 2:
        return valid_words if valid_words else [0]
        
    return valid_words

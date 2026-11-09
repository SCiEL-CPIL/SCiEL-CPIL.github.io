import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    MAX_TIME = 50  # Leave some buffer for overhead
    
    def greedy_search(seed):
        rng = np.random.default_rng(seed)
        code = set()
        # Start with a random word
        first_word = rng.integers(0, 2**N)
        code.add(first_word)
        
        # List for fast iteration, set for O(1) lookup
        code_list = [first_word]
        
        fail_count = 0
        max_fails = 500  # Stop if we fail 500 times in a row
        
        start_time = time.time()
        
        while time.time() - start_time < MAX_TIME * 0.9:
            # Generate a batch of candidates to reduce Python overhead?
            # Or just one by one. One by one is simpler.
            c = rng.integers(0, 2**N)
            
            if c in code:
                continue
                
            # Check distance against all existing words
            valid = True
            for w in code_list:
                if (c ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                code.add(c)
                code_list.append(c)
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > max_fails:
                    break
                    
        return len(code_list)
        
    best_size = 0
    best_code = []
    
    # Run multiple trials with different seeds
    # We don't know the exact time available, so we'll run trials until time runs out
    start_total = time.time()
    trial = 0
    
    while time.time() - start_total < 55:
        seed = 12345 + trial
        # We need to actually return the code, not just the size.
        # Let's modify greedy_search to return the list.
        
        rng = np.random.default_rng(seed)
        code_set = set()
        first_word = rng.integers(0, 2**N)
        code_set.add(first_word)
        code_list = [first_word]
        
        fail_count = 0
        max_fails = 1000
        
        while time.time() - start_total < 55:
            c = rng.integers(0, 2**N)
            if c in code_set:
                continue
            
            valid = True
            for w in code_list:
                if (c ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                code_set.add(c)
                code_list.append(c)
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > max_fails:
                    break
            
            if len(code_list) > best_size:
                best_size = len(code_list)
                best_code = code_list.copy()
        
        trial += 1
        
    return best_code

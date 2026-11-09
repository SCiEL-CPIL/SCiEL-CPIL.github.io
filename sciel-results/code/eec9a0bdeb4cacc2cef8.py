import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    DEADLINE = 85.0
    
    start_time = time.time()
    
    # 1. Initialize with RM(1,4) code (32 words)
    def get_rm_code():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    best_words = get_rm_code()
    best_count = len(best_words)
    
    # Greedy search with restarts
    rng = np.random.default_rng(42)
    
    for restart in range(20): # Try 20 restarts
        current_words = best_words.copy()
        current_count = len(current_words)
        
        # Shuffle current words to break symmetry?
        rng.shuffle(current_words)
        
        no_improvement_count = 0
        max_no_improvement = 50000
        
        while time.time() - start_time < DEADLINE and no_improvement_count < max_no_improvement:
            # Generate a batch of random words
            batch_size = 500
            candidates = rng.integers(0, 1 << N, size=batch_size)
            
            added_any = False
            for w in candidates:
                if time.time() - start_time > DEADLINE:
                    break
                
                valid = True
                for c in current_words:
                    if (w ^ c).bit_count() < MIN_DIST:
                        valid = False
                        break
                
                if valid:
                    current_words.append(int(w))
                    current_count += 1
                    added_any = True
                    no_improvement_count = 0
                    if current_count > best_count:
                        best_words = current_words.copy()
                        best_count = current_count
                else:
                    no_improvement_count += 1
            
            if not added_any and no_improvement_count > 10000:
                # Try a local move: remove a random word, try to add a new one
                if len(current_words) > 32:
                    remove_idx = rng.integers(0, len(current_words))
                    removed_word = current_words.pop(remove_idx)
                    current_count -= 1
                    # Try to add a random word
                    for _ in range(100):
                        w = int(rng.integers(0, 1 << N))
                        valid = True
                        for c in current_words:
                            if (w ^ c).bit_count() < MIN_DIST:
                                valid = False
                                break
                        if valid:
                            current_words.append(w)
                            current_count += 1
                            break
                    # If we didn't add a word, put the old one back?
                    # Or keep the set smaller?
                    # Keeping it smaller might allow future additions.
            
        if time.time() - start_time > DEADLINE:
            break

    return best_words

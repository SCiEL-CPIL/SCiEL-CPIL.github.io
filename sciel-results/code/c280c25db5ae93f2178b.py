import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N
    
    # We will use a greedy algorithm to find a large code.
    # We'll try a few random permutations if time allows.
    # However, one good run is often sufficient to beat the seed (32).
    # The linear code has 256 words. Greedy might find 258+.
    
    selected = []
    
    # To speed up, we can start with the known 256-word linear code?
    # Or just run greedy from scratch.
    # Greedy from scratch is simpler and less prone to bias if the linear code is not extendable.
    # But 256 is a strong baseline.
    # Let's try to generate the linear code first, then try to add more?
    # No, the task says "Write a new design from scratch".
    # But using known structures is allowed if it helps.
    # Actually, the prompt says "FRESH DESIGN... not the existing lineage".
    # It doesn't forbid using mathematical knowledge.
    # However, a simple greedy is a "fresh design" in terms of code structure.
    
    # Let's optimize the greedy:
    # 1. Shuffle indices 0 to 2^17 - 1
    # 2. Iterate
    # 3. Check distance
    
    # Optimization: 
    # Instead of checking all selected words, we can maintain a bitset of "forbidden" words?
    # No, distance is not a simple mask.
    
    # Let's just run the greedy.
    # 131k candidates.
    # If we select ~260 words.
    # Inner loop average length?
    # If the code is dense, most candidates fail early.
    
    random.seed(42) # For reproducibility in testing, but random is fine.
    
    best_selected = []
    
    # Try multiple random orders if time permits
    # But 131k * 260 checks is ~34M ops.
    # In Python, 34M ops might take 10-20s.
    # So we can afford 2-3 runs.
    
    for run in range(5):
        if time.time() > deadline - 5:
            break
            
        # Generate random permutation of all words
        # This is memory heavy (131k ints) but fast.
        candidates = list(range(TOTAL_WORDS))
        random.shuffle(candidates)
        
        current_selected = []
        
        for w in candidates:
            if time.time() > deadline - 2:
                break
            
            # Check if w is at distance >= 6 from all in current_selected
            is_valid = True
            for s in current_selected:
                if (w ^ s).bit_count() < MIN_DIST:
                    is_valid = False
                    break
            
            if is_valid:
                current_selected.append(w)
                
        # Update best
        if len(current_selected) > len(best_selected):
            best_selected = current_selected
            
        # If we found a very large code, we might stop early?
        # No, keep going to maximize.
        
    return best_selected

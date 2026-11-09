import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_code = []
    best_size = 0
    
    # Generate the 32-word First-order Reed-Muller code RM(1,4)
    # Embedded in 17 bits (last bit 0? or just 16 bits?)
    # The task is length 17.
    # RM(1,4) is length 16.
    # We can extend it to 17 by adding a 0th coordinate?
    # Distance in 16 bits is 8. Distance in 17 bits (with 0) is 8.
    # So RM(1,4) + 0 is a valid 17-bit code of size 32, dist 8.
    
    rm_code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # 17th bit is 0
            rm_code.append(word)
            
    # Strategy 1: Start with RM code and complete
    # Strategy 2: Pure random
    
    while time.time() < deadline:
        # Alternate between strategies or just do both in loop?
        # Let's do random restarts primarily, but include some seeded ones.
        
        if random.random() < 0.5:
            # Seeded start
            code = rm_code.copy()
            # Candidates are all other words
            candidates = list(range(1 << 17))
            # Remove RM words from candidates? 
            # It's easier to just check distance. If we pick an RM word again, 
            # distance 0 < 6, so it will be rejected. 
            # But we should avoid wasting time on known bad candidates.
            # We can create a set of used words.
            used = set(code)
            # Filter candidates? 131k filter is fast.
            candidates = [w for w in candidates if w not in used]
            random.shuffle(candidates)
        else:
            # Pure random start
            code = []
            candidates = list(range(1 << 17))
            random.shuffle(candidates)
            
        # Greedy completion
        for w in candidates:
            valid = True
            for c in code:
                if (w ^ c).bit_count() < 6:
                    valid = False
                    break
            if valid:
                code.append(w)
                
        if len(code) > best_size:
            best_size = len(code)
            best_code = code
            
    return best_code

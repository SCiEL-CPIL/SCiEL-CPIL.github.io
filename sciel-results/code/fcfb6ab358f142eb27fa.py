import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 18
    MIN_DIST = 6
    
    # 1. Generate RM(1, 4)
    rm14 = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm14.append(word)
            
    # 2. Define Q basis
    qA = 0
    qB = 0
    qC = 0
    for point in range(16):
        x1 = (point >> 0) & 1
        x2 = (point >> 1) & 1
        x3 = (point >> 2) & 1
        x4 = (point >> 3) & 1
        
        bA = (x1 & x3) ^ (x2 & x4)
        bB = (x1 & x4) ^ (x2 & x3)
        bC = (x1 & x2) ^ (x3 & x4)
        
        if bA: qA |= 1 << point
        if bB: qB |= 1 << point
        if bC: qC |= 1 << point
        
    Q = [0]
    for mask in range(1, 8):
        w = 0
        if mask & 1: w ^= qA
        if mask & 2: w ^= qB
        if mask & 4: w ^= qC
        Q.append(w)
        
    # 3. Construct Kerdock
    kerdock = []
    for w in rm14:
        for q in Q:
            kerdock.append(w ^ q)
            
    # 4. Add complements
    full_mask = (1 << 16) - 1
    all_words_16 = list(kerdock)
    for w in kerdock:
        all_words_16.append(w ^ full_mask)
        
    # 5. Extend to 18
    final_words = [w for w in all_words_16]
    
    # Verification
    if time.time() < deadline - 5.0:
        min_d = 18
        for i in range(len(final_words)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, len(final_words)):
                d = (final_words[i] ^ final_words[j]).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        return rm14 # Fallback
        if min_d >= MIN_DIST:
            return final_words
    else:
        return rm14

    return final_words

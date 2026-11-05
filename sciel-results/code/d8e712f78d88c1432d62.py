import random
import time
import math

def construct():
    N = 256
    
    # Precompute APs
    ap3s = []
    ap4s = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3s.append((a, b, c))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            ap4s.append((a, b, c, dd))

    # Precompute dependencies for fast bitset checks
    four_ap_deps = [[] for _ in range(N + 1)]
    three_ap_deps = [[] for _ in range(N + 1)]
    
    for a, b, c, d in ap4s:
        # Mask of other 3 elements
        four_ap_deps[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (d-1)))
        four_ap_deps[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (d-1)))
        four_ap_deps[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (d-1)))
        four_ap_deps[d].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
        
    for a, b, c in ap3s:
        three_ap_deps[a].append((1 << (b-1)) | (1 << (c-1)))
        three_ap_deps[b].append((1 << (a-1)) | (1 << (c-1)))
        three_ap_deps[c].append((1 << (a-1)) | (1 << (b-1)))

    def get_score(A):
        score = 0
        for a, b, c in ap3s:
            if (A & (1 << (a-1))) and (A & (1 << (b-1))) and (A & (1 << (c-1))):
                score += 1
        return score

    def random_valid_set():
        A = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            valid = True
            for m in four_ap_deps[x]:
                if (A & m) == m:
                    valid = False
                    break
            if valid:
                A |= (1 << (x-1))
        return A

    def local_search(start_A, time_limit):
        A = start_A
        current_score = get_score(A)
        best_A = A
        best_score = current_score
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        start_time = time.time()
        
        while T > T_min and time.time() - start_time < time_limit:
            x = random.randint(1, N)
            bit = 1 << (x-1)
            
            if A & bit:
                # Try removing x
                loss = 0
                for m in three_ap_deps[x]:
                    if (A & m) == m:
                        loss += 1
                
                delta = -loss
                if delta >= 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                    A &= ~bit
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = A
            else:
                # Try adding x
                valid = True
                for m in four_ap_deps[x]:
                    if (A & m) == m:
                        valid = False
                        break
                
                if valid:
                    gain = 0
                    for m in three_ap_deps[x]:
                        if (A & m) == m:
                            gain += 1
                    
                    delta = gain
                    if delta >= 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                        A |= bit
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_A = A
            
            T *= cooling
        
        return best_A, best_score

    global_start = time.time()
    global_deadline = global_start + 55
    
    best_A = 0
    best_score = -1
    
    while time.time() < global_deadline:
        remaining = global_deadline - time.time()
        if remaining < 1:
            break
        
        A = random_valid_set()
        # Run local search for a chunk of time
        # If we have a lot of time left, run longer?
        # Let's do 1s chunks to allow restarts.
        chunk_time = 1.0
        
        A_opt, score_opt = local_search(A, chunk_time)
        
        if score_opt > best_score:
            best_score = score_opt
            best_A = A_opt
            
    # If we have significant time left, do a long polish on the best found
    remaining = global_deadline - time.time()
    if remaining > 5 and best_A != 0:
        A_polished, score_polished = local_search(best_A, remaining - 1)
        if score_polished > best_score:
            best_score = score_polished
            best_A = A_polished

    result = []
    for i in range(N):
        if best_A & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

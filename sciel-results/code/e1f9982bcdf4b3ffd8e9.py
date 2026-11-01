import time
import random
import math

def construct(n):
    start_time = time.time()
    # Keep a safety margin of 10s
    deadline = start_time + 110
    
    def get_canonical_superperm(k):
        if k == 1:
            return [1]
        word = get_canonical_superperm(k-1)
        perms, seen = [], set()
        # m is k in this context, window size k-1
        for i in range(len(word) - (k-1) + 1):
            window = tuple(word[i:i + k-1])
            if len(set(window)) == k-1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [k] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new

    if n <= 6:
        return "".join(map(str, get_canonical_superperm(n)))
    
    if n != 7:
        # Fallback for n > 7, though task is for 7
        # Just return canonical to be safe, though it won't be optimal
        return "".join(map(str, get_canonical_superperm(n)))

    # n == 7
    S_6 = get_canonical_superperm(6)
    
    perms, seen = [], set()
    for i in range(len(S_6) - 6 + 1):
        window = tuple(S_6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    # perms is a list of 5040 tuples
    # Map to indices
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Compute overlap matrix M
    # M[i][j] = overlap(perms[i], perms[j])
    N = 5040
    M = [[0]*N for _ in range(N)]
    
    # Optimization: Use local variables for speed
    p_list = perms
    M_local = M
    
    for i in range(N):
        pi = p_list[i]
        Mi = M_local[i]
        for j in range(N):
            pj = p_list[j]
            # Calculate overlap
            # Check k=6 down to 0
            # pi[-k:] == pj[:k]
            if pi[-6:] == pj[:6]:
                Mi[j] = 6
            elif pi[-5:] == pj[:5]:
                Mi[j] = 5
            elif pi[-4:] == pj[:4]:
                Mi[j] = 4
            elif pi[-3:] == pj[:3]:
                Mi[j] = 3
            elif pi[-2:] == pj[:2]:
                Mi[j] = 2
            elif pi[-1:] == pj[:1]:
                Mi[j] = 1
            else:
                Mi[j] = 0
                
            if time.time() > deadline - 30:
                # If matrix calc is taking too long, abort and use canonical
                return "".join(map(str, get_canonical_superperm(7)))

    seq = list(range(N))
    
    def calc_score(s):
        score = 0
        for i in range(N-1):
            score += M[s[i]][s[i+1]]
        return score
        
    current_score = calc_score(seq)
    best_score = current_score
    best_seq = seq[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # We can run many iterations
    while time.time() < deadline - 5:
        if T < min_T:
            # Restart or break
            break
            
        i = random.randint(0, N-2)
        j = i + 1
        
        si = seq[i]
        sj = seq[j]
        
        left = seq[i-1] if i > 0 else None
        right = seq[i+2] if i+2 < N else None
        
        old_cost = 0
        new_cost = 0
        
        if left is not None:
            old_cost += M[left][si]
            new_cost += M[left][sj]
            
        old_cost += M[si][sj]
        new_cost += M[sj][si]
        
        if right is not None:
            old_cost += M[sj][right]
            new_cost += M[si][right]
            
        delta = new_cost - old_cost
        
        if delta > 0 or random.random() < math.exp(delta / T):
            seq[i], seq[j] = sj, si
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_seq = seq[:]
        
        T *= cooling

    # Reconstruct
    s = best_seq
    word = list(perms[s[0]]) + [7] + list(perms[s[0]])
    for idx in s[1:]:
        block = list(perms[idx]) + [7] + list(perms[idx])
        max_overlap = min(len(word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or word[len(word)-o:] == block[:o]:
                overlap = o
                break
        word += block[overlap:]
        
    return "".join(map(str, word))

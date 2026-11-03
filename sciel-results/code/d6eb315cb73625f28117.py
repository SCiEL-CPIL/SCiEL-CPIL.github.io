import itertools
import time

def classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def construct(n):
    if n <= 1:
        return "1"
    if n <= 6:
        return classic_construct(n)
    
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
    # We maximize this sum.
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            # pi is tuple of length n-1
            max_k = 0
            # Check suffix of pi vs prefix of pj
            for k in range(1, n): # max length is n-1
                if pi[-k:] == pj[:k]:
                    max_k = k
            overlap[i][j] = max_k
            
    # Heuristic: Greedy Nearest Neighbor (Max Overlap)
    # Start from a random node or node 0
    visited = [False] * N
    order = []
    current = 0
    visited[0] = True
    order.append(0)
    
    for _ in range(N - 1):
        best_next = -1
        best_score = -1
        for j in range(N):
            if not visited[j]:
                score = overlap[current][j]
                if score > best_score:
                    best_score = score
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        order.append(best_next)
        current = best_next
        
    # 2-Opt Improvement
    # We want to maximize sum of overlaps.
    # Cost of path = sum(overlap[order[i]][order[i+1]])
    # Reversing a segment changes 2 edges.
    
    def get_path_score(ord):
        s = 0
        for i in range(len(ord)-1):
            s += overlap[ord[i]][ord[i+1]]
        return s
        
    current_score = get_path_score(order)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 2):
            if time.time() > deadline:
                break
            for j in range(i + 1, N - 1):
                # Try reversing order[i+1 : j+1]
                # Old edges: (order[i], order[i+1]) and (order[j], order[j+1])
                # New edges: (order[i], order[j]) and (order[i+1], order[j+1])
                old_score = overlap[order[i]][order[i+1]] + overlap[order[j]][order[j+1]]
                new_score = overlap[order[i]][order[j]] + overlap[order[i+1]][order[j+1]]
                
                if new_score > old_score:
                    # Reverse
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    current_score += new_score - old_score
                    improved = True
                    
    # Construct the word
    word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        if not word:
            word.extend(block)
        else:
            # Find max overlap with current word end
            max_ov = 0
            max_check = min(len(word), len(block))
            for k in range(max_check, -1, -1):
                if word[-k:] == block[:k]:
                    max_ov = k
                    break
            word.extend(block[max_ov:])
            
    return "".join(map(str, word))

# EVOLVE-BLOCK-START
def construct(n):
    if n <= 1:
        return "1"
    if n <= 6:
        return classic_construct(n)
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix
    overlap = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = 0
            for k in range(1, n):
                if pi[-k:] == pj[:k]:
                    max_k = k
            overlap[i][j] = max_k
            
    # Greedy NN
    visited = [False] * N
    order = []
    current = 0
    visited[0] = True
    order.append(0)
    
    for _ in range(N - 1):
        best_next = -1
        best_score = -1
        for j in range(N):
            if not visited[j]:
                score = overlap[current][j]
                if score > best_score:
                    best_score = score
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        order.append(best_next)
        current = best_next
        
    # 2-Opt
    def get_path_score(ord):
        s = 0
        for i in range(len(ord)-1):
            s += overlap[ord[i]][ord[i+1]]
        return s
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 2):
            if time.time() > deadline:
                break
            for j in range(i + 1, N - 1):
                old_score = overlap[order[i]][order[i+1]] + overlap[order[j]][order[j+1]]
                new_score = overlap[order[i]][order[j]] + overlap[order[i+1]][order[j+1]]
                
                if new_score > old_score:
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    improved = True
                    
    # Construct word
    word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        if not word:
            word.extend(block)
        else:
            max_ov = 0
            max_check = min(len(word), len(block))
            for k in range(max_check, -1, -1):
                if word[-k:] == block[:k]:
                    max_ov = k
                    break
            word.extend(block[max_ov:])
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END

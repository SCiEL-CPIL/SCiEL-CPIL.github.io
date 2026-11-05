import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Generate standard superpermutation for n-1
    def gen_std(m):
        if m == 1:
            return [1]
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    prev_word = gen_std(n-1)
    blocks = []
    seen = set()
    for i in range(len(prev_word) - (n-2)):
        window = tuple(prev_word[i:i+n-1])
        if len(set(window)) == n-1 and window not in seen:
            seen.add(window)
            blocks.append(window)
            
    m = len(blocks)
    
    # Precompute overlaps
    ov = [[0]*m for _ in range(m)]
    for i in range(m):
        b_i = blocks[i]
        for j in range(m):
            b_j = blocks[j]
            max_ov = min(len(b_i), len(b_j))
            for k in range(max_ov, 0, -1):
                if b_i[-k:] == b_j[:k]:
                    ov[i][j] = k
                    break
    
    def get_score(order):
        s = 0
        for i in range(len(order)-1):
            s += ov[order[i]][order[i+1]]
        return s

    # Greedy Initialization
    best_order = list(range(m))
    best_score = -1
    start_time = time.time()
    time_limit = 110
    
    # Try standard order first
    std_order = list(range(m))
    std_score = get_score(std_order)
    if std_score > best_score:
        best_score = std_score
        best_order = std_order

    # Random Greedy
    for _ in range(20):
        visited = [False]*m
        order = []
        current = random.randint(0, m-1)
        visited[current] = True
        order.append(current)
        
        for _ in range(m-1):
            best_next = -1
            best_ov_val = -1
            for next_node in range(m):
                if not visited[next_node]:
                    val = ov[current][next_node]
                    if val > best_ov_val:
                        best_ov_val = val
                        best_next = next_node
            if best_next == -1: break
            visited[best_next] = True
            order.append(best_next)
            current = best_next
            
        score = get_score(order)
        if score > best_score:
            best_score = score
            best_order = order
            
        if time.time() - start_time > 20:
            break

    # VNS 2-opt with Prefix Sum Optimization
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Calculate D and PrefixD
        D = [0]*(m-1)
        for k in range(m-1):
            D[k] = ov[best_order[k+1]][best_order[k]] - ov[best_order[k]][best_order[k+1]]
            
        PrefixD = [0]*(m)
        for i in range(1, m):
            PrefixD[i] = PrefixD[i-1] + D[i-1]
            
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(m-1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, m):
                delta = 0
                # Boundary 1
                if i > 0:
                    delta -= ov[best_order[i-1]][best_order[i]]
                    delta += ov[best_order[i-1]][best_order[j]]
                # Boundary 2
                if j < m-1:
                    delta -= ov[best_order[j]][best_order[j+1]]
                    delta += ov[best_order[i]][best_order[j+1]]
                
                # Internal sum: sum of D[k] for k=i to j-1
                # D[k] is defined for k in 0..m-2.
                # Sum D[i]..D[j-1] = PrefixD[j] - PrefixD[i]
                delta += (PrefixD[j] - PrefixD[i])
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_delta > 0:
            best_order[best_i:best_j+1] = best_order[best_i:best_j+1][::-1]
            best_score += best_delta
            improved = True

    # Reconstruct
    word = []
    if m > 0:
        first_block = blocks[best_order[0]]
        word.extend(first_block)
        word.append(n)
        word.extend(first_block)
        
        for i in range(1, m):
            curr_block = blocks[best_order[i]]
            prev_block = blocks[best_order[i-1]]
            
            max_ov = min(len(prev_block), len(curr_block))
            ov_val = 0
            for k in range(max_ov, 0, -1):
                if prev_block[-k:] == curr_block[:k]:
                    ov_val = k
                    break
            
            word.extend(curr_block[ov_val:])
            word.append(n)
            word.extend(curr_block)
            
    return "".join(map(str, word))

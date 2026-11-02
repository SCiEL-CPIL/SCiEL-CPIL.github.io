import time
import random
import sys

def construct(n):
    # For small n, the recursive construction is optimal or near-optimal.
    if n <= 5:
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
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n=6 and n=7, use local search
    # 1. Generate recursive word
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # 2. Extract sequence of permutations (order of first appearance)
    seen = set()
    seq = []
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n and w not in seen:
            seen.add(w)
            seq.append(w)
            
    N = len(seq)
    
    # 3. Build cost matrix
    # cost[i][j] = n - overlap(seq[i], seq[j])
    
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s for search and output
    
    # Use list of lists of small ints
    cost_matrix = [[0] * N for _ in range(N)]
    
    for i in range(N):
        if time.time() > deadline - 30:
            break
        pi = seq[i]
        row = cost_matrix[i]
        for j in range(N):
            pj = seq[j]
            ov = 0
            for k in range(n - 1, 0, -1):
                if pi[n-k:] == pj[:k]:
                    ov = k
                    break
            row[j] = n - ov

    # 4. Local Search
    order = list(range(N))
    
    def get_total_cost(order):
        total = n
        for i in range(len(order) - 1):
            total += cost_matrix[order[i]][order[i+1]]
        return total

    current_cost = get_total_cost(order)
    best_cost = current_cost
    best_order = order.copy()
    
    search_deadline = deadline - 5
    
    while time.time() < search_deadline:
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Swap adjacent
            if N < 2: break
            i = random.randint(0, N - 2)
            j = i + 1
            
            idx_a = order[i]
            idx_b = order[j]
            
            old_cost = 0
            new_cost = 0
            
            if i > 0:
                p_idx = order[i-1]
                old_cost += cost_matrix[p_idx][idx_a]
                new_cost += cost_matrix[p_idx][idx_b]
            
            old_cost += cost_matrix[idx_a][idx_b]
            new_cost += cost_matrix[idx_b][idx_a]
            
            if j < N - 1:
                n_idx = order[j+1]
                old_cost += cost_matrix[idx_b][n_idx]
                new_cost += cost_matrix[idx_a][n_idx]
            
            delta = new_cost - old_cost
            
            if delta < 0 or (delta == 0 and random.random() < 0.1):
                order[i], order[j] = order[j], order[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = order.copy()
                    
        elif move_type == 1:
            # Skip insert for now
            pass

        else:
            # Reverse segment
            if N < 2: break
            i = random.randint(0, N - 1)
            if N - i < 2:
                continue
            
            max_len = min(10, N - i)
            length = random.randint(2, max_len)
            j = i + length - 1
            
            old_cost = 0
            new_cost = 0
            
            if i > 0:
                a_idx = order[i-1]
                old_cost += cost_matrix[a_idx][order[i]]
                new_cost += cost_matrix[a_idx][order[j]]
                
            for k in range(i, j):
                old_cost += cost_matrix[order[k]][order[k+1]]
                new_cost += cost_matrix[order[k+1]][order[k]]
                
            if j < N - 1:
                b_idx = order[j+1]
                old_cost += cost_matrix[order[j]][b_idx]
                new_cost += cost_matrix[order[i]][b_idx]
                
            delta = new_cost - old_cost
            
            if delta < 0 or (delta == 0 and random.random() < 0.1):
                order[i:j+1] = reversed(order[i:j+1])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_order = order.copy()

    # Reconstruct word
    final_word = list(seq[best_order[0]])
    for idx in best_order[1:]:
        p = seq[idx]
        ov = 0
        for k in range(n - 1, -1, -1):
            if k == 0 or final_word[len(final_word)-k:] == list(p[:k]):
                ov = k
                break
        final_word.extend(p[ov:])
        
    return "".join(map(str, final_word))

if __name__ == "__main__":
    pass

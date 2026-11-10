import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Classic construction for n <= 6 is optimal or close enough
    if n <= 6:
        return classic_construct(n)
    
    if n == 7:
        # Step 1: Get S6
        word6 = list(map(int, classic_construct(6)))
        
        # Step 2: Extract permutations of 1..6
        perms = []
        seen = set()
        for i in range(len(word6) - 6 + 1):
            window = tuple(word6[i:i+6])
            if window not in seen:
                seen.add(window)
                perms.append(window)
        
        # There should be 720 perms
        if len(perms) != 720:
             # Fallback to classic if something went wrong
             return classic_construct(n)
             
        last_perm = perms[-1]
        last_idx = perms.index(last_perm)
        
        # Precompute LCP suffix matrix
        # M[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
        N = 720
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            p_i = perms[i]
            for j in range(N):
                p_j = perms[j]
                lcp = 0
                # Max possible LCP is 6
                for k in range(6, 0, -1):
                    if p_i[6-k:] == p_j[:k]:
                        lcp = k
                        break
                M[i][j] = lcp
                
        # We want to find a permutation sigma of 0..719
        # Maximize sum: M[last_idx][sigma[0]] + M[sigma[0]][sigma[1]] + ... + M[sigma[718]][sigma[719]]
        
        best_score = -1
        best_path = None
        
        # Greedy + 2-Opt local search
        # Since N=720 is small, we can do multiple restarts
        
        def greedy_start(start_idx):
            visited = [False]*N
            visited[start_idx] = True
            path = [start_idx]
            curr = start_idx
            for _ in range(N-1):
                best_next = -1
                best_val = -1
                for j in range(N):
                    if not visited[j]:
                        val = M[curr][j]
                        if val > best_val:
                            best_val = val
                            best_next = j
                visited[best_next] = True
                path.append(best_next)
                curr = best_next
            return path

        def path_score(path, start_idx):
            score = M[start_idx][path[0]]
            for k in range(len(path)-1):
                score += M[path[k]][path[k+1]]
            return score

        def two_opt(path, start_idx):
            improved = True
            while improved:
                improved = False
                for i in range(1, len(path)-1):
                    for j in range(i+1, len(path)):
                        # Segment path[i:j+1]
                        # Check if reversing improves score
                        # Old: ... path[i-1] -> path[i] ... path[j] -> path[j+1] ...
                        # New: ... path[i-1] -> path[j] ... path[i] -> path[j+1] ...
                        
                        if i == 1:
                            # Connects to start_idx
                            old = M[start_idx][path[i]] + M[path[j]][path[j+1]] if j < len(path)-1 else M[start_idx][path[i]]
                            # If j is last, no right neighbor
                            if j < len(path)-1:
                                new = M[start_idx][path[j]] + M[path[i]][path[j+1]]
                            else:
                                new = M[start_idx][path[j]]
                        else:
                            old = M[path[i-1]][path[i]] + M[path[j]][path[j+1]] if j < len(path)-1 else M[path[i-1]][path[i]]
                            if j < len(path)-1:
                                new = M[path[i-1]][path[j]] + M[path[i]][path[j+1]]
                            else:
                                new = M[path[i-1]][path[j]]
                        
                        if new > old:
                            path[i:j+1] = reversed(path[i:j+1])
                            improved = True
                            if time.time() > deadline - 5:
                                return path
            return path

        # Run a few greedy restarts
        for _ in range(50):
            if time.time() > deadline - 10: break
            # Random start
            start_node = random.randint(0, N-1)
            path = greedy_start(start_node)
            path = two_opt(path, last_idx)
            score = path_score(path, last_idx)
            if score > best_score:
                best_score = score
                best_path = path
                
        # Construct final word
        word = list(word6)
        curr_idx = last_idx
        for idx in best_path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            # Calculate overlap with current word
            # We know the overlap is M[curr_idx][idx]
            overlap = M[curr_idx][idx]
            word.extend(block[overlap:])
            curr_idx = idx
            
        return "".join(map(str, word))

    else:
        return classic_construct(n)

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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

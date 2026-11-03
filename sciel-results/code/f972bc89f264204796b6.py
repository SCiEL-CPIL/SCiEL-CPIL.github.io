import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # For n <= 4, standard construction is optimal
    if n <= 4:
        return _standard_superperm(n)
    
    # For n = 5, standard is optimal (153)
    if n == 5:
        return _standard_superperm(5)
    
    # For n = 6, standard is 873. Is it optimal?
    # Literature suggests n=6 is also optimal at 873.
    if n == 6:
        return _standard_superperm(6)
    
    # For n = 7, try to improve on 5913
    if n == 7:
        # Get the base permutations of 1..6
        base_perms = list(itertools.permutations(range(1, 7)))
        
        # Standard ordering is the order they appear in S_6
        # We can just use the standard construction to get the order
        # Or we can just use lexicographic order? 
        # The standard recursive order is specific.
        # Let's generate the standard S_6 to get the order.
        s6 = _standard_superperm(6)
        # Extract the order of permutations from s6?
        # Actually, the order in S_7 is derived from S_6.
        # The blocks in S_7 correspond to the permutations in S_6.
        
        # Let's just use the lexicographic order of base_perms for the TSP start,
        # or better, the order from the standard construction.
        
        # To get the standard order:
        # S_6 is built from S_5.
        # The permutations of 1..6 in S_6 are in the order of the blocks.
        
        # Let's just run TSP on the lexicographic order first.
        # If it doesn't work, try the standard order.
        
        # Precompute overlap matrix
        # K[i][j] = max k such that base_perms[i][6-k:] == base_perms[j][:k]
        N = len(base_perms)
        K = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = base_perms[i]
            for j in range(N):
                pj = base_perms[j]
                max_k = 0
                for k in range(6, -1, -1):
                    if pi[6-k:] == pj[:k]:
                        max_k = k
                        break
                K[i][j] = max_k
        
        # Cost matrix C[i][j] = 13 - K[i][j]
        # We want to find a permutation of indices 0..N-1 minimizing sum C[i][i+1]
        
        # Initial solution: Lexicographic order (0, 1, ..., N-1)
        path = list(range(N))
        
        # 2-Opt local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N-1):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i+1, N):
                    # Check if reversing path[i+1..j] improves
                    # Current cost: C[path[i]][path[i+1]] + C[path[j]][path[j+1]] (if j+1 < N)
                    # New cost: C[path[i]][path[j]] + C[path[i+1]][path[j+1]] (if j+1 < N)
                    # Note: path is a cycle? No, path is a sequence.
                    # The string is concatenation of blocks in path order.
                    # Overlaps are between path[k] and path[k+1].
                    
                    cost_old = 0
                    cost_new = 0
                    
                    # Affected edges: (i, i+1) and (j, j+1)
                    # If j == N-1, no edge (j, j+1)
                    
                    # Edge (i, i+1)
                    old_edge_1 = K[path[i]][path[i+1]]
                    
                    # Edge (j, j+1)
                    if j < N-1:
                        old_edge_2 = K[path[j]][path[j+1]]
                    else:
                        old_edge_2 = 0 # No cost for end
                    
                    new_edge_1 = K[path[i]][path[j]]
                    
                    if j < N-1:
                        new_edge_2 = K[path[i+1]][path[j+1]]
                    else:
                        new_edge_2 = 0 # No cost for end
                        
                    # We want to maximize overlap (minimize cost)
                    # Cost = Sum(13 - K)
                    # Delta Cost = (13 - new_edge_1) + (13 - new_edge_2) - ((13 - old_edge_1) + (13 - old_edge_2))
                    # Delta Cost = old_edge_1 + old_edge_2 - new_edge_1 - new_edge_2
                    # If Delta Cost > 0, we improved (overlap increased)
                    
                    if (old_edge_1 + old_edge_2) < (new_edge_1 + new_edge_2):
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
        
        # Construct string from optimized path
        word = []
        for idx in path:
            p = base_perms[idx]
            block = list(p) + [7] + list(p)
            
            # Find overlap with current word
            # The overlap is K[word_end_perm][p]?
            # No, the overlap is determined by the actual string content.
            # But since we know the structure, the overlap is K[prev_p][p]
            # where prev_p is the permutation of the previous block.
            
            if not word:
                word.extend(block)
            else:
                # The last block was base_perms[path[-2]]?
                # Actually, we just need to append block[overlap:]
                # The overlap is K[prev_idx][idx]
                # But wait, the word might have been modified?
                # No, the word is just the concatenation.
                # The overlap between the end of the current word and the start of the new block
                # is exactly the overlap between the previous permutation and the current one.
                
                prev_idx = path[path.index(idx)-1] if idx != path[0] else None
                # This is inefficient. Let's keep track of prev_idx.
        
        # Re-do construction with tracked prev
        word = []
        prev_idx = -1
        for idx in path:
            p = base_perms[idx]
            block = list(p) + [7] + list(p)
            
            if not word:
                word.extend(block)
            else:
                overlap = K[prev_idx][idx]
                word.extend(block[overlap:])
            
            prev_idx = idx
        
        result = "".join(map(str, word))
        
        # Validate
        if _is_valid_superperm(result, 7):
            return result
        else:
            # Fallback to standard
            return _standard_superperm(7)
            
    # Fallback for other n
    return _standard_superperm(n)

def _standard_superperm(n):
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

def _is_valid_superperm(s, n):
    target = set(itertools.permutations(range(1, n+1)))
    found = set()
    for i in range(len(s) - n + 1):
        window = tuple(int(c) for c in s[i:i+n])
        if window in target:
            found.add(window)
            if len(found) == len(target):
                return True
    return False

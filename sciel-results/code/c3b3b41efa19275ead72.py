"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"

    # Step 1: Generate the base permutations of length n-1
    # We use the standard Graham-Lagarias recursive structure to get the "natural" order
    # of permutations, which often provides a good starting point for local search.
    
    def get_gl_perms(m):
        """Returns the list of permutations of 1..m in GL order."""
        if m == 1:
            return [(1,)]
        
        prev_perms = get_gl_perms(m - 1)
        new_perms = []
        for p in prev_perms:
            # The GL construction inserts m into p to form a new perm, 
            # but for the superpermutation step, we just need the list of 
            # (m-1)-perms that will be wrapped.
            # Actually, the GL word for n is built from the perms of n-1.
            # So for n=7, we need perms of 1..6.
            pass
        
        # The standard GL construction for word W_n uses the sequence of (n-1)-permutations 
        # that appear in W_{n-1}.
        # Let's just generate all permutations of 1..n-1. The "order" is what we optimize.
        # But to match the "classic" start, we can just use itertools.permutations.
        return list(itertools.permutations(range(1, m+1)))

    # For n=7, we optimize the order of 1..6 permutations.
    base_perms = list(itertools.permutations(range(1, n)))
    M = len(base_perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that base_perms[i][-k:] == base_perms[j][:k]
    # k can be 0 to n-2 (since they are length n-1, max overlap is n-2? 
    # Wait, if P_i ends with ...a and P_j starts with a..., 
    # In the block P_i, n, P_i, the next block is P_j, n, P_j.
    # The overlap is between the suffix of (P_i, n, P_i) and prefix of (P_j, n, P_j).
    # The suffix of P_i, n, P_i is P_i.
    # The prefix of P_j, n, P_j is P_j.
    # So we need max k such that P_i[-k:] == P_j[:k].
    # Since P_i and P_j are permutations of length n-1, max k is n-2?
    # Example n=3, perms len 2. P_i=(1,2), P_j=(2,1). Overlap 1.
    # Example n=4, perms len 3. P_i=(1,2,3), P_j=(2,3,4) (if 4 was in it, but it's not).
    # P_i=(1,2,3), P_j=(3,1,2). Overlap 2 (3).
    # Max overlap is n-2.
    
    max_k = n - 2
    overlap = [[0] * M for _ in range(M)]
    
    # Optimization: Use string representation for faster slicing? 
    # Tuples are fine for n=7, M=720. 720*720 = 518k pairs.
    for i in range(M):
        pi = base_perms[i]
        for j in range(M):
            pj = base_perms[j]
            # Check from max_k down to 0
            # We can break early if we find the max possible? No, we want max.
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap[i][j] = found

    # Initial order: 0, 1, ..., M-1
    order = list(range(M))
    
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list) - 1):
            total += overlap[ord_list[i]][ord_list[i+1]]
        return total

    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # Local Search
    # We want to maximize overlap.
    # Operations:
    # 1. Swap adjacent
    # 2. 2-opt (reverse segment)
    
    # Heuristic: Greedy construction might be a better start than random or identity.
    # Let's try a simple greedy: start with 0, always pick the next one with max overlap.
    # This is O(M^2).
    visited = [False] * M
    current = 0
    visited[0] = True
    greedy_order = [0]
    
    for _ in range(M - 1):
        max_ov = -1
        next_node = -1
        for j in range(M):
            if not visited[j]:
                ov = overlap[current][j]
                if ov > max_ov:
                    max_ov = ov
                    next_node = j
        if next_node == -1:
            break
        visited[next_node] = True
        greedy_order.append(next_node)
        current = next_node
        
    if len(greedy_order) == M:
        go_ov = calc_total_overlap(greedy_order)
        if go_ov > best_overlap:
            best_overlap = go_ov
            best_order = greedy_order
            order = greedy_order
    
    # 2-Opt Local Search
    # Reverse segment [i, j]
    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
    # Note: Indices in the order list.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try 2-opt
        for i in range(len(order) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(order)):
                if time.time() > deadline:
                    break
                
                # Calculate delta in overlap
                # We want to maximize overlap.
                # Delta = New_Overlap - Old_Overlap
                
                old_ov = 0
                new_ov = 0
                
                # Boundary left
                if i > 0:
                    old_ov += overlap[order[i-1]][order[i]]
                    new_ov += overlap[order[i-1]][order[j]]
                
                # Boundary right
                if j < len(order) - 1:
                    old_ov += overlap[order[j]][order[j+1]]
                    new_ov += overlap[order[i]][order[j+1]]
                
                # Internal edges
                # Old: (i, i+1), ..., (j-1, j)
                # New: (j, j-1), ..., (i+1, i)
                for k in range(i, j):
                    old_ov += overlap[order[k]][order[k+1]]
                    new_ov += overlap[order[k+1]][order[k]]
                
                delta = new_ov - old_ov
                if delta > 0:
                    # Apply reversal
                    order[i:j+1] = reversed(order[i:j+1])
                    best_overlap += delta
                    best_order = order[:]
                    improved = True
                    # Restart i loop? Or continue? 
                    # Continuing is faster, might get stuck in local min but 2-opt is robust.
                    # To be safe and thorough, restart i=0.
                    i = -1 
                    break
        
        # If no 2-opt improvement, try adjacent swaps (1-opt) to escape shallow local minima
        if not improved:
            for i in range(len(order) - 1):
                if time.time() > deadline:
                    break
                # Swap i, i+1
                a = order[i]
                b = order[i+1]
                
                old_ov = 0
                new_ov = 0
                
                if i > 0:
                    old_ov += overlap[order[i-1]][a]
                    new_ov += overlap[order[i-1]][b]
                
                if i+1 < len(order) - 1:
                    old_ov += overlap[b][order[i+2]]
                    new_ov += overlap[a][order[i+2]]
                
                old_ov += overlap[a][b]
                new_ov += overlap[b][a]
                
                delta = new_ov - old_ov
                if delta > 0:
                    order[i] = b
                    order[i+1] = a
                    best_overlap += delta
                    best_order = order[:]
                    improved = True

    # Construct the word from best_order
    word = []
    for idx in best_order:
        p = base_perms[idx]
        block = list(p) + [n] + list(p)
        
        # Overlap with current word
        # We need to find max k such that word[-k:] == block[:k]
        # Since we calculated overlaps between perms, the overlap between 
        # (prev_p, n, prev_p) and (p, n, p) is determined by prev_p and p.
        # Specifically, the suffix of prev_p, n, prev_p is prev_p.
        # The prefix of p, n, p is p.
        # So overlap is overlap[prev_idx][idx].
        
        if not word:
            word = block
        else:
            prev_idx = best_order[best_order.index(idx) - 1] if idx != best_order[0] else None
            # This lookup is O(M). Better to keep track.
            # Let's reconstruct properly.
            pass
            
    # Re-construct efficiently
    word = []
    prev_idx = -1
    for idx in best_order:
        p = base_perms[idx]
        block = list(p) + [n] + list(p)
        
        if not word:
            word = block
        else:
            ov = overlap[prev_idx][idx]
            # The overlap is between the end of the previous block and start of this block.
            # Previous block ends with prev_p.
            # This block starts with p.
            # Overlap is ov.
            # So we append block[ov:]
            word.extend(block[ov:])
        prev_idx = idx
        
    # Final Polish: Remove redundant characters if any
    # The construction above ensures every permutation of 1..n is present?
    # Yes, because for each perm P of 1..n-1, we have the block P, n, P.
    # The window of length n starting at the position of n in this block is P + [n] (if we align right?)
    # Actually, the window containing n is formed by the last n-1 chars of the previous context 
    # and the first n-1 chars of the current context?
    # No, the standard GL proof says that if we list all (n-1)-perms, the word contains all n-perms.
    # Specifically, the permutation (x1...x_{n-1}, n) is present if (x1...x_{n-1}) is in the list.
    # And (n, x1...x_{n-1}) is present if (x1...x_{n-1}) is in the list?
    # Wait, the block is P, n, P.
    # Windows of size n:
    # 1. Inside the first P? No, P is size n-1.
    # 2. Crossing into n: (last n-1 of prev context) + n?
    # The GL construction guarantees coverage.
    
    # Apply the "remove if safe" heuristic from the original program
    improved_word = list(word)
    if time.time() < deadline - 5: # Leave 5s for safety
        def build_coverage(w, n):
            counts = defaultdict(int)
            for i in range(len(w) - n + 1):
                counts[tuple(w[i:i+n])] += 1
            return counts
        
        counts = build_coverage(improved_word, n)
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            i = 0
            while i < len(improved_word):
                if time.time() > deadline - 5:
                    break
                j_min = max(0, i - n + 1)
                j_max = min(i, len(improved_word) - n)
                
                safe = True
                affected = []
                for j in range(j_min, j_max + 1):
                    w = tuple(improved_word[j:j+n])
                    affected.append(w)
                    if counts[w] <= 1:
                        safe = False
                        break
                
                if safe:
                    for w in affected:
                        counts[w] -= 1
                    improved_word.pop(i)
                    changed = True
                else:
                    i += 1
        
        word = improved_word

    return "".join(map(str, word))
# EVOLVE-BLOCK-END

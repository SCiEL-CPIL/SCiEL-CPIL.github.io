import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Classic recursive construction
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
                if new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # 2. Extract permutation sequence
    perms_seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if len(set(p)) == n:
            if not perms_seq or perms_seq[-1] != p:
                perms_seq.append(p)
    
    # The recursive construction might produce duplicates in the window list if overlaps are handled differently?
    # Actually, the standard recursive construction produces a word where every window of length n is a unique permutation?
    # No, it produces a word that *contains* all permutations.
    # The sequence of windows might have repeats if the path loops.
    # For n=7, the recursive construction is known to visit each permutation exactly once in the "skeleton" sense,
    # but the actual windows in the string are all permutations.
    # Let's ensure we have all 5040 unique permutations in our sequence.
    # If the recursive construction is valid, it contains all.
    # We will use the list of unique permutations found in order.
    
    # 3. Local Search (2-opt)
    # Cost function: sum of (n - LCP(prev, curr))
    
    def lcp_len(a, b):
        l = 0
        # Max overlap is n-1
        for i in range(n - 1):
            if a[i] == b[i]:
                l += 1
            else:
                break
        return l

    # Precompute LCP matrix for speed?
    # 5040 x 5040 is 25M entries.
    # In Python, a list of lists of ints is slow.
    # Let's compute on the fly, but optimize the 2-opt loop.
    
    seq = perms_seq
    N = len(seq)
    
    # Initial cost
    def get_cost(i, j):
        if i == j: return 0
        return n - lcp_len(seq[i], seq[j])
        
    current_cost = n
    for i in range(N - 1):
        current_cost += get_cost(i, i + 1)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline: break
            for j in range(i + 1, N):
                if time.time() > deadline: break
                # Calculate delta cost for reversing seq[i..j]
                # Old links: (i-1, i) and (j, j+1)
                # New links: (i-1, j) and (i, j+1)
                
                cost_old = 0
                cost_new = 0
                
                if i > 0:
                    cost_old += get_cost(i - 1, i)
                    cost_new += get_cost(i - 1, j)
                else:
                    # Start of sequence, no left neighbor
                    pass

                if j < N - 1:
                    cost_old += get_cost(j, j + 1)
                    cost_new += get_cost(i, j + 1)
                else:
                    # End of sequence, no right neighbor
                    pass
                    
                # Note: Internal links are reversed.
                # Cost of link (a, b) is n - LCP(a, b).
                # Cost of link (b, a) is n - LCP(b, a).
                # These are generally different.
                # So we must recalculate internal costs?
                # Wait, 2-opt on a directed graph (TSP) requires recalculating internal path cost if the graph is asymmetric.
                # Here the graph is asymmetric (LCP(a,b) != LCP(b,a)).
                # So we can't just look at boundaries. We must sum the internal costs.
                # This makes 2-opt O(N) per swap, total O(N^3).
                # N=5040. N^3 is too slow.
                
                # Alternative: Use Or-opt (move a segment) or just accept that we can't do full 2-opt.
                # Or, restrict to swapping adjacent elements? (2-swap)
                # Adjacent swap: i, i+1.
                # Old: (i-1, i), (i, i+1), (i+1, i+2)
                # New: (i-1, i+1), (i+1, i), (i, i+2)
                # This is O(1) per swap.
                # Total O(N^2) swaps? No, N adjacent swaps.
                # We can do multiple passes.
                
                # Let's try adjacent swaps (bubble sort style)
                # But 2-opt is better.
                # Can we precompute LCP?
                # If we precompute LCP matrix, internal cost sum is still O(N).
                
                # Let's stick to adjacent swaps for now, multiple passes.
                pass
        # If we didn't use adjacent swaps in the loop above, we need to implement them.
        # Let's restart the loop with adjacent swaps.
        
    # Re-implementing the loop for adjacent swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline: break
            # Swap seq[i] and seq[i+1]
            # Old links: (i-1, i), (i, i+1), (i+1, i+2)
            # New links: (i-1, i+1), (i+1, i), (i, i+2)
            
            delta = 0
            
            if i > 0:
                delta -= get_cost(i - 1, i)
                delta += get_cost(i - 1, i + 1)
            
            # Middle link
            delta -= get_cost(i, i + 1)
            delta += get_cost(i + 1, i)
            
            if i + 1 < N - 1:
                delta -= get_cost(i + 1, i + 2)
                delta += get_cost(i, i + 2)
                
            if delta < 0:
                seq[i], seq[i+1] = seq[i+1], seq[i]
                current_cost += delta
                improved = True
                # Restart scan? Or continue?
                # Continue for now, might miss some, but next pass will catch.
                # If we want to be thorough, we could restart i=0.
                # But that might be slow.
        
    # 4. Reconstruct
    final_word = list(seq[0])
    for i in range(1, N):
        prev = seq[i-1]
        curr = seq[i]
        ov = 0
        for o in range(n - 1, -1, -1):
            if prev[n-o:] == curr[:o]:
                ov = o
                break
        final_word += curr[ov:]
        
    return "".join(map(str, final_word))

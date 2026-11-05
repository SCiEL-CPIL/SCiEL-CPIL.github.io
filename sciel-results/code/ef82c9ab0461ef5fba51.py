import time
import random
import numpy as np

N = 192
K = 3
MAX_CARD = 48

def _count_bad_triples(a, S):
    """
    Count the number of triples (x, y, z) from S \ {a} such that (x+y+z) % a == 0.
    S is a list of distinct integers. a is in S.
    """
    T = [x for x in S if x != a]
    m = len(T)
    if m < 3:
        return 0
    
    # Optimization: Use residue counts?
    # For small m, O(m^3) is 47^3 = 103k. Too slow for many iterations.
    # O(m^2) approach:
    # For each pair (i, j) in T, we need k in T, k!=i, k!=j such that (T[i]+T[j]+T[k]) % a == 0.
    # This is still O(m^3) if we search k.
    # Better: Precompute residues of T modulo a.
    # Let counts[r] be the number of elements in T with residue r.
    # We need to count triples of indices (i,j,k) distinct such that (r_i + r_j + r_k) % a == 0.
    # This is counting solutions to r1+r2+r3 = 0 mod a with distinct elements.
    
    # Let's stick to O(m^2) with a lookup table for the third element?
    # For a fixed a, we can build a dictionary: residue -> list of indices?
    # Or just:
    # For each pair (i, j), required residue r_k = (-T[i] - T[j]) % a.
    # We need to know if there is a k in T, k!=i, k!=j, with T[k] % a == r_k.
    
    # Since m is small (~48), we can do:
    # 1. Create array T.
    # 2. Create array res = [x % a for x in T].
    # 3. For each i in range(m):
    #    For each j in range(i+1, m):
    #       target = (-T[i] - T[j]) % a
    #       # Check if any k > j has res[k] == target? 
    #       # We can precompute suffix counts of residues?
    #       # Or just loop k? That's O(m^3).
    
    # Let's try O(m^2) with a set of remaining elements?
    # Iterate i, j. Maintain a count of available residues?
    # This is tricky because we remove i and j.
    
    # Given m <= 48, O(m^3) is ~100k operations.
    # If we do this 1000 times, it's 100M ops. Might be slow in Python (5-10s).
    # We need to be faster.
    
    # Vectorized check using Numpy?
    # T is array.
    # Sums of all pairs: T[:, None] + T[None, :]
    # This is m x m.
    # For each pair (i,j), we need to check if (sum + T[k]) % a == 0 for any k not i,j.
    
    # Alternative:
    # For a fixed a, the condition is that the set of residues of T does not contain a zero-sum triple.
    # We can compute the number of zero-sum triples in O(a^2) or O(m^2) using FFT?
    # Actually, since m is small, let's just optimize the Python loop.
    
    count = 0
    # Use a set of values? No, values are distinct.
    # Precompute T % a
    res = [x % a for x in T]
    
    # We want to count triples i < j < k such that (res[i] + res[j] + res[k]) % a == 0
    # We can iterate i, j and look for k?
    # To make it O(m^2), we can precompute for each pair (i,j) the required residue, 
    # and check if it exists in T with index > j?
    # We can maintain a frequency array of residues for indices > j.
    
    # freq[r] = count of elements in T[j+1:] with residue r
    freq = [0] * a
    # Initialize with all elements
    for r in res:
        freq[r] += 1
        
    # Iterate i from 0 to m-1
    # Before inner loop, T[i] is "active" but we process pairs (i, j) with j > i.
    # We need to exclude i and j from the "k" pool.
    
    # Let's restructure:
    # For i in range(m-2):
    #    Remove T[i] from freq? No, T[i] is part of the pair.
    #    For j in range(i+1, m-1):
    #       Remove T[j] from freq?
    #       target = (-res[i] - res[j]) % a
    #       count += freq[target]
    #       # But freq[target] might include i or j?
    #       # We must ensure k is distinct from i and j.
    #       # If we maintain freq as counts of elements in T[j+1:], then i and j are not in it.
    
    # Let's implement this carefully.
    # freq represents counts of residues in T[current_j+1 : m]
    
    # Initialize freq with counts of T[1:] (since i=0, j starts at 1, k starts at j+1)
    # Actually, let's just rebuild freq for each i? O(m^2) rebuild + O(m^2) loop = O(m^2).
    
    for i in range(m - 2):
        # Build freq for T[i+1:]
        freq = [0] * a
        for x in T[i+1:]:
            freq[x % a] += 1
            
        for j in range(i + 1, m - 1):
            # Before checking, T[j] is in freq. We need to exclude it because k must be > j.
            # So subtract T[j] from freq first.
            res_j = T[j] % a
            freq[res_j] -= 1
            
            target = (-T[i] - T[j]) % a
            count += freq[target]
            
    return count

def _is_valid(S):
    """Check if S is a valid set."""
    for a in S:
        if _count_bad_triples(a, S) > 0:
            return False
    return True

def _conflicts_after_add(S, c):
    """
    Check if adding c to S creates any violations.
    Returns 0 if valid, >0 if invalid.
    """
    S_new = S + [c]
    # Check c as divisor
    if _count_bad_triples(c, S_new) > 0:
        return 1
    
    # Check existing a in S as divisor
    # Only triples involving c can create NEW violations for existing a.
    # Triples not involving c were already valid in S.
    # So we only need to check triples (c, x, y) where x, y in S, x!=y, x!=a, y!=a.
    # And we need (c + x + y) % a == 0.
    
    for a in S:
        T = [x for x in S if x != a]
        # We need to count pairs (x, y) in T such that (c + x + y) % a == 0.
        # This is equivalent to (x + y) % a == (-c) % a.
        
        # Use the same freq trick?
        # We just need to know if ANY such pair exists.
        # m = len(T). m < 48.
        # O(m^2) check is fine.
        m = len(T)
        for i in range(m):
            for j in range(i+1, m):
                if (c + T[i] + T[j]) % a == 0:
                    return 1
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Candidates sorted descending
    candidates = list(range(N, 0, -1))
    
    S = []
    
    # Greedy Phase
    for c in candidates:
        if time.time() > deadline:
            break
        if len(S) >= MAX_CARD:
            break
            
        # Check if we can add c
        if _conflicts_after_add(S, c) == 0:
            S.append(c)
        else:
            # Optional: Try to swap?
            # For now, just skip.
            pass
            
    # If we didn't reach MAX_CARD, try Local Search to improve
    if len(S) < MAX_CARD:
        rng = random.Random(42)
        best_S = list(S)
        best_size = len(S)
        
        # Try to find a valid set of size MAX_CARD via swaps
        # We will perform random swaps and check validity.
        # Since checking full validity is expensive, we can use _conflicts_after_add/remove logic?
        # Or just full check _is_valid.
        
        # Let's try a simple hill climbing on size.
        # State: S
        # Move: Remove a, Add b.
        # If valid and size same, keep? No, we want to reach 48.
        # The greedy already maximized size given the order.
        # Maybe a different order helps?
        
        # Let's try a few random restarts with different orders.
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Shuffle candidates, but keep large numbers prioritized?
            # Or just random order?
            # Random order might pick small numbers first, which are bad.
            # Let's mix: 50% large, 50% random.
            
            order = list(candidates)
            rng.shuffle(order)
            # Bias towards large numbers?
            # Actually, the greedy "add if valid" is order dependent.
            
            S_restart = []
            for c in order:
                if time.time() > deadline - 5:
                    break
                if len(S_restart) >= MAX_CARD:
                    break
                if _conflicts_after_add(S_restart, c) == 0:
                    S_restart.append(c)
            
            if len(S_restart) > best_size:
                best_S = S_restart
                best_size = len_size = len(S_restart)
                if best_size == MAX_CARD:
                    break

    return best_S

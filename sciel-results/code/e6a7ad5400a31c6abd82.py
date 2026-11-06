import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def can_add_descending(S, a):
        """Check if 'a' can be added to S, where all elements in S are > a.
        Condition: No pair b, c in S (b < c) such that a divides (b+c)."""
        n = len(S)
        # Optimization: use residues
        # If a is small, we can use a boolean array of size a
        # But a is up to 256, so just loop.
        # To speed up, we can break early.
        for i in range(n):
            b = S[i]
            # We need c > b.
            # (b+c) % a == 0 => c % a == (-b) % a
            target = (-b) % a
            # Check if any c in S[i+1:] has c % a == target
            # Since S is sorted ascending, we can't easily binary search residues.
            # Just loop.
            for j in range(i + 1, n):
                if S[j] % a == target:
                    return False
        return True

    def greedy_descending():
        S = []
        # S will be kept sorted ascending for ease of b < c check in can_add_descending
        # We iterate a from N down to 1
        for a in range(N, 0, -1):
            if can_add_descending(S, a):
                # Insert a into S maintaining sorted order
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        return S

    best_set = []
    best_len = 0
    
    # Run greedy descending multiple times with slight randomizations?
    # The deterministic greedy is fast. Let's run it first.
    S = greedy_descending()
    best_set = S[:]
    best_len = len(S)
    
    # Local Search / Repair Phase
    # Try to add small elements by removing conflicting large elements
    # We iterate through small candidates a
    # For each a not in S, find conflicts
    # If we can remove k elements to add 1, and k < threshold, do it
    
    # To make it faster, we only consider removing elements if the net gain is positive
    # i.e., we remove 1 element to add 1? That's neutral. 
    # We want to remove 1 to add 2? Or remove 2 to add 3?
    # Actually, removing 1 to add 1 might help add *another* small element later.
    
    # Let's do a systematic local search:
    # 1. Try to add each missing element a.
    # 2. If it fails, identify the "bottleneck" elements in S.
    # 3. Try removing one bottleneck element and see if a can be added.
    # 4. If yes, accept the swap (size unchanged, but structure changed).
    # 5. Then try to add other elements.
    
    for iteration in range(100):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        S_set = set(S)
        
        # Try to add any missing element
        for a in range(1, N + 1):
            if a in S_set:
                continue
            
            # Check if a can be added directly
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
                S_set.add(a)
                best_len = len(S)
                improved = True
                break # Restart loop with new S
        
        if improved:
            continue
            
        # If no direct addition, try swap: remove x, add a
        # We need to find x such that removing x allows a to be added.
        # Conflicts for a are pairs (b,c) in S with a | (b+c).
        # If we remove b, the pair (b,c) is gone.
        # We need to remove enough elements to break ALL conflicting pairs.
        
        # Find all conflicting pairs for a
        conflicts = []
        n = len(S)
        for i in range(n):
            b = S[i]
            target = (-b) % a
            for j in range(i + 1, n):
                if S[j] % a == target:
                    conflicts.append((i, j))
        
        if not conflicts:
            # Should have been added above
            continue
            
        # We want to remove a set of indices such that every conflict pair is covered.
        # This is a Hitting Set problem.
        # Simple heuristic: Pick the index that appears in the most conflicts.
        # Count frequency of each index in conflicts
        freq = {}
        for i, j in conflicts:
            freq[i] = freq.get(i, 0) + 1
            freq[j] = freq.get(j, 0) + 1
        
        # Sort indices by frequency
        sorted_indices = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
        
        # Try removing the most frequent index
        # Limit to top 5 candidates to keep it fast
        for rem_idx in sorted_indices[:5]:
            # Create a new set without S[rem_idx]
            S_new = S[:]
            S_new.pop(rem_idx)
            
            # Check if a can be added to S_new
            if can_add_descending(S_new, a):
                # Accept the swap
                pos = bisect.bisect_left(S_new, a)
                S_new.insert(pos, a)
                S = S_new
                S_set = set(S)
                improved = True
                break
        
        if not improved:
            # Try random perturbation?
            # Or just break if we've tried all swaps?
            # Let's try a random swap
            if len(S) > 1:
                rem_idx = random.randint(0, len(S) - 1)
                a_candidate = random.randint(1, N)
                if a_candidate in S_set:
                    continue
                S_new = S[:]
                S_new.pop(rem_idx)
                if can_add_descending(S_new, a_candidate):
                    pos = bisect.bisect_left(S_new, a_candidate)
                    S_new.insert(pos, a_candidate)
                    S = S_new
                    S_set = set(S)
                    improved = True
        
        if not improved:
            # No more improvements in this pass
            break

    # Final check: maybe we can add more elements now?
    # Run greedy descending on the "gaps"?
    # The current S is not necessarily a suffix.
    # Let's just return the best set found.
    
    if len(S) > best_len:
        best_set = S[:]
        best_len = len(S)
        
    return best_set

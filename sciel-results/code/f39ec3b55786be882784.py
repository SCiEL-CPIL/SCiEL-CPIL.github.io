<<<<<<< SEARCH
import random
import time

def construct():
    n = 256
    max_size = 64 # actually limited by Sidon property to ~16
    deadline = time.time() + 55.0
    
    best_sum = 0
    best_A = []
    best_B = []
    
    # Helper to check Sidon property
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
        
    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    # We aim for |A|=16, |B|=16.
    # Max possible sum is 32.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Try to build A and B
        # Randomize the order of candidates or the sets themselves?
        # Greedy construction of Sidon set is sensitive to order.
        
        # Let's try to construct A first, then B.
        # Random subset of [1, 256]
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        A = []
        B = []
        
        # Greedy for A
        for x in candidates:
            if len(A) >= 16: break
            # Check if adding x keeps A Sidon
            # And check difference conflict with current B? 
            # Since B is empty, just Sidon check.
            # But we also need to reserve differences for B?
            # Hard to know in advance.
            
            # Simple check: is A + [x] Sidon?
            new_A = A + [x]
            if is_sidon(new_A):
                A = new_A
        
        # Now construct B
        # We need B to be Sidon AND D_B disjoint from D_A.
        D_A = get_diffs(A)
        
        for x in candidates:
            if len(B) >= 16: break
            new_B = B + [x]
            if is_sidon(new_B):
                D_new_B = get_diffs(new_B)
                # Check disjoint
                if D_new_B.isdisjoint(D_A):
                    B = new_B
                    # Update D_A? No, D_A is fixed.
                    # But we need to ensure future additions to B don't conflict with D_A.
                    # The check `D_new_B.isdisjoint(D_A)` ensures the *current* B is ok.
                    # But does it ensure we can extend? No.
                    # However, if we greedily pick, we might paint ourselves into a corner.
                    
        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]
            
        if best_sum == 32:
            break
            
    return [best_A, best_B]
=======
import random
import time

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55.0

    best_sum = 0
    best_A = []
    best_B = []

    # Bitset-based helpers for speed
    def can_add_to_sidon(x, S, sums_used):
        """Check if x can be added to Sidon set S. sums_used is a set of used pair sums."""
        for s in S:
            pair_sum = x + s
            if pair_sum in sums_used:
                return False
        return True

    def add_to_sidon(x, S, sums_used):
        """Add x to S, update sums_used."""
        for s in S:
            sums_used.add(x + s)
        sums_used.add(2 * x)
        S.append(x)

    def get_diffs_set(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        """Check if x can be added to B (Sidon + no diff conflict with A)."""
        for s in B:
            pair_sum = x + s
            if pair_sum in sums_B:
                return False
        if 2 * x in sums_B:
            return False
        for s in B:
            d1 = x - s
            d2 = s - x
            if d1 in diffs_A or d2 in diffs_A:
                return False
            if d1 in diffs_B or d2 in diffs_B:
                return False
        return True

    def add_to_B(x, B, sums_B, diffs_B):
        for s in B:
            sums_B.add(x + s)
            d1 = x - s
            d2 = s - x
            diffs_B.add(d1)
            diffs_B.add(d2)
        sums_B.add(2 * x)
        B.append(x)

    def polish(A, B, sums_A, diffs_A, sums_B, diffs_B, deadline):
        """Try to improve (A, B) by inserting skipped elements."""
        target_size = 16
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return A, B, sums_A, diffs_A, sums_B, diffs_B
                if x in A or x in B:
                    continue
                if len(A) >= target_size:
                    break
                if can_add_to_sidon(x, A, sums_A):
                    add_to_sidon(x, A, sums_A)
                    improved = True
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return A, B, sums_A, diffs_A, sums_B, diffs_B
                if x in A or x in B:
                    continue
                if len(B) >= target_size:
                    break
                if can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                    add_to_B(x, B, sums_B, diffs_B)
                    improved = True
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    for restart in range(2000):
        if time.time() > deadline:
            break

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        # Greedy for A
        for x in candidates:
            if len(A) >= 16:
                break
            if can_add_to_sidon(x, A, sums_A):
                add_to_sidon(x, A, sums_A)

        diffs_A = get_diffs_set(A)

        # Greedy for B
        for x in candidates:
            if len(B) >= 16:
                break
            if can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                add_to_B(x, B, sums_B, diffs_B)

        # Polish
        A, B, sums_A, diffs_A, sums_B, diffs_B = polish(A, B, sums_A, diffs_A, sums_B, diffs_B, deadline)

        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]

        if best_sum >= 32:
            break

    return [best_A, best_B]
>>>>>>> REPLACE

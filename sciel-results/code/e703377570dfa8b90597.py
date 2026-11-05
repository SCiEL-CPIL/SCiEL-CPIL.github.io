import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def can_add(x, S, diffs_S, sums_S, diffs_other):
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)

    def try_add(x, S, diffs_S, sums_S, diffs_other):
        if can_add(x, S, diffs_S, sums_S, diffs_other):
            add_to_set(x, S, diffs_S, sums_S)
            return True
        return False

    def greedy_round(A, B, diffs_A, diffs_B, sums_A, sums_B, order, rng):
        """Try adding elements in given order."""
        for x in order:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                # Add to smaller set, or randomly if equal
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if rng.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)

    def local_search_swap(A, B, diffs_A, diffs_B, sums_A, sums_B, rng, max_iter=200):
        """Try swapping elements between sets or removing and re-adding."""
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try swapping an element from A with one from B
            if len(A) > 1 and len(B) > 1:
                for i in range(len(A)):
                    for j in range(len(B)):
                        a_elem = A[i]
                        b_elem = B[j]
                        # Try removing a_elem from A and adding to B, 
                        # and removing b_elem from B and adding to A
                        # First check if removing a_elem from A is clean
                        # Then check if b_elem can go into A
                        # Then check if a_elem can go into B
                        
                        # Remove a_elem from A
                        A.remove(a_elem)
                        # Rebuild diffs_A and sums_A
                        new_diffs_A = set()
                        new_sums_A = set()
                        for k in range(len(A)):
                            new_sums_A.add(2 * A[k])
                            for l in range(k + 1, len(A)):
                                new_diffs_A.add(abs(A[k] - A[l]))
                                new_sums_A.add(A[k] + A[l])
                        
                        # Check if b_elem can be added to A
                        if can_add(b_elem, A, new_diffs_A, new_sums_A, diffs_B):
                            # Add b_elem to A
                            A.append(b_elem)
                            new_sums_A.add(2 * b_elem)
                            for a in A:
                                if a != b_elem:
                                    new_diffs_A.add(abs(b_elem - a))
                                    new_sums_A.add(b_elem + a)
                            
                            # Now remove b_elem from B
                            B.remove(b_elem)
                            new_diffs_B = set()
                            new_sums_B = set()
                            for k in range(len(B)):
                                new_sums_B.add(2 * B[k])
                                for l in range(k + 1, len(B)):
                                    new_diffs_B.add(abs(B[k] - B[l]))
                                    new_sums_B.add(B[k] + B[l])
                            
                            # Check if a_elem can be added to B
                            if can_add(a_elem, B, new_diffs_B, new_sums_B, new_diffs_A):
                                # Valid swap!
                                B.append(a_elem)
                                new_sums_B.add(2 * a_elem)
                                for b in B:
                                    if b != a_elem:
                                        new_diffs_B.add(abs(a_elem - b))
                                        new_sums_B.add(a_elem + b)
                                
                                # Update all
                                diffs_A.clear()
                                diffs_A.update(new_diffs_A)
                                sums_A.clear()
                                sums_A.update(new_sums_A)
                                diffs_B.clear()
                                diffs_B.update(new_diffs_B)
                                sums_B.clear()
                                sums_B.update(new_sums_B)
                                improved = True
                                break
                            else:
                                # Revert: add b_elem back to B
                                B.append(b_elem)
                        # Revert A
                        A.append(a_elem)
                        # Rebuild original diffs_A and sums_A
                        diffs_A.clear()
                        sums_A.clear()
                        for k in range(len(A)):
                            sums_A.add(2 * A[k])
                            for l in range(k + 1, len(A)):
                                diffs_A.add(abs(A[k] - A[l]))
                                sums_A.add(A[k] + A[l])
                    if improved:
                        break
            
            # Try adding any remaining elements
            remaining = [x for x in range(1, 257) if x not in A and x not in B]
            rng.shuffle(remaining)
            for x in remaining:
                if try_add(x, A, diffs_A, sums_A, diffs_B):
                    improved = True
                    break
                if try_add(x, B, diffs_B, sums_B, diffs_A):
                    improved = True
                    break
            if improved:
                break
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        rng = random.Random(attempts * 7919 + 12345)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Multiple ordering strategies
        strategy = attempts % 4
        
        if strategy == 0:
            # Random order
            candidates = list(range(1, 257))
            rng.shuffle(candidates)
        elif strategy == 1:
            # Ascending
            candidates = list(range(1, 257))
        elif strategy == 2:
            # Descending
            candidates = list(range(256, 0, -1))
        else:
            # Alternating small/large
            small = list(range(1, 129))
            large = list(range(129, 257))
            rng.shuffle(small)
            rng.shuffle(large)
            candidates = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small):
                    candidates.append(small[si])
                    si += 1
                if li < len(large):
                    candidates.append(large[li])
                    li += 1
        
        # Greedy phase with the chosen order
        greedy_round(A, B, diffs_A, diffs_B, sums_A, sums_B, candidates, rng)
        
        # Second pass with different order to catch missed elements
        remaining = [x for x in range(1, 257) if x not in A and x not in B]
        rng.shuffle(remaining)
        greedy_round(A, B, diffs_A, diffs_B, sums_A, sums_B, remaining, rng)
        
        # Local search: try swaps
        if len(A) > 1 and len(B) > 1:
            local_search_swap(A, B, diffs_A, diffs_B, sums_A, sums_B, rng, max_iter=50)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()

    return [best_A, best_B]


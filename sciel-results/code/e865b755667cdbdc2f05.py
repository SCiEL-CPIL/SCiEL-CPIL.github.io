import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    best_size = 0
    best_A = []
    best_B = []
    
    def build_set(order, forbidden_diffs_mask, max_size=16):
        """
        Greedily builds a Sidon set from 'order' 
        such that its differences do not intersect with forbidden_diffs_mask.
        Returns the set.
        """
        S = []
        sums_mask = 0
        diffs_mask = 0
        
        for x in order:
            if len(S) >= max_size:
                break
                
            # Check Sidon condition
            valid = True
            temp_sums = 0
            
            # Check sums
            for a in S:
                s = x + a
                if (sums_mask >> s) & 1:
                    valid = False
                    break
                temp_sums |= (1 << s)
            
            if valid:
                if (sums_mask >> (2*x)) & 1:
                    valid = False
                else:
                    temp_sums |= (1 << (2*x))
            
            # Check diff conflicts with forbidden and internal
            if valid:
                for a in S:
                    d = abs(x - a)
                    if d == 0:
                        continue
                    # Must not be in forbidden
                    if (forbidden_diffs_mask >> d) & 1:
                        valid = False
                        break
                    # Must not be in internal diffs? 
                    # Wait, Sidon set requires unique sums.
                    # Does it require unique diffs? No.
                    # But if d is already in diffs_mask, does it matter?
                    # No, diffs can repeat. e.g. {1, 3, 5} diffs: 2, 4, 2.
                    # So we only care about forbidden_diffs_mask.
            
            if valid:
                S.append(x)
                sums_mask |= temp_sums
                # Update diffs_mask? 
                # We don't actually need diffs_mask for Sidon condition.
                # We only need it if we were checking something else.
                # The problem statement says "Pair sums ... must be unique".
                # It does NOT say differences must be unique within the set.
                # It says "Their nonzero difference sets must be disjoint".
                # So D_A and D_B are sets of values.
                # So for B, we just need D_B to be disjoint from D_A.
                # D_B is the set of values |b1-b2|.
                # So we don't need to track diffs_mask for B's internal consistency, 
                # only that new diffs don't hit forbidden_diffs_mask.
                
        return S

    # We want to maximize |A|+|B|.
    # Max possible is 32.
    
    # Strategy:
    # 1. Try to find A of size 16.
    # 2. Then find B of size 16 avoiding D_A.
    
    # We will iterate many times.
    
    # Optimization: Precompute nothing? 
    # The inner loop is small.
    
    # To be faster, we can try to build A and B in parallel?
    # No, sequential is better for "disjoint diffs" constraint.
    
    # Let's try randomizing the order for A and B independently?
    # If we use the same order, it might be correlated.
    # Let's use independent random orders.
    
    while time.time() < deadline:
        # Generate random A
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A = build_set(order_A, 0, 16)
        
        if len(A) == 0:
            continue
            
        # Compute D_A mask
        diffs_A_mask = 0
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = abs(A[i] - A[j])
                diffs_A_mask |= (1 << d)
                
        # Generate random B avoiding D_A
        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B = build_set(order_B, diffs_A_mask, 16)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B
            
        # If we hit 32, we can break?
        # 32 is the theoretical max.
        if best_size == 32:
            break

    # Fallback if nothing found (should not happen)
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]

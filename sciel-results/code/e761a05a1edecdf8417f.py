"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 58.0  # Safety margin for 60s limit (evaluator kills at 90s)
    n = 4096
    best = []
    
    # Strategy: 
    # 1. Use a geometric progression skeleton (powers of 4 or similar) to guarantee unique triple sums.
    # 2. Then greedily insert smaller integers into the gaps.
    # 3. Use random restarts with different skeletons or insertion orders to explore the space.
    # 4. Use a boolean array for O(1) sum lookups.
    
    # The maximum possible sum is 3*n = 12288.
    max_sum = 3 * n + 1
    
    def try_construct(seed_val):
        rng = random.Random(seed_val)
        # Use a boolean array for existing sums
        sums = np.zeros(max_sum, dtype=bool)
        selected = []
        
        # Step 1: Build a skeleton using a geometric progression.
        # We want a base b such that b^k <= n.
        # If we use base 4: 4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024, 4^6=4096.
        # Powers of 4 guarantee that sums of triples are unique in base 4 representation 
        # (since max coefficient is 3, which fits in a single base-4 digit).
        # However, we can also try other bases or scaled versions.
        
        # Let's try a few different geometric progressions.
        # Option A: Pure powers of 4
        # Option B: Scaled powers of 4 (e.g., start at 2, 8, 32...)
        # Option C: Randomized geometric progression with base 4 but with small perturbations?
        # Actually, pure powers of 4 is the safest skeleton.
        
        # Let's generate the skeleton elements.
        # We can choose to include or exclude some powers to allow for more flexibility?
        # No, the skeleton must be valid. Powers of 4 are always valid.
        # But maybe we can use a larger base to get fewer skeleton elements and more room for fillers?
        # Base 4 is the minimum base for B_3[1] (since 3 < 4).
        
        # Let's stick with base 4 for the skeleton to ensure validity.
        # Skeleton: [1, 4, 16, 64, 256, 1024, 4096] (if 4096 is allowed, yes)
        # Wait, 4096 is the limit. 4^6 = 4096.
        # Sums of triples of these will be unique.
        
        # However, we might be able to get more elements by NOT using the full geometric progression
        # and instead using a smarter greedy approach from the start, or by using a different structure.
        
        # Let's try a hybrid:
        # 1. Start with a small skeleton (e.g., first few powers of 4).
        # 2. Greedily add remaining candidates from 1 to n.
        
        # Actually, the current greedy approach from 1 to n is already quite good.
        # The inspiration suggests using a skeleton to maximize "safe" insertions.
        # But greedy from 1 to n is O(n*k^2) which is fast.
        # The issue is that greedy from 1 to n might get stuck in a local optimum.
        # Random restarts with different initial skeletons or random orderings might help.
        
        # Let's try:
        # 1. Generate a random permutation of 1..n.
        # 2. Greedily add elements in that order.
        # This explores different local optima.
        
        # But we also have the idea of a geometric skeleton.
        # Let's combine: 
        # - First, add a geometric skeleton (powers of 4) to ensure a valid base.
        # - Then, greedily add remaining elements in a random order.
        
        # Let's implement this.
        
        # Skeleton: powers of 4 up to n
        skeleton = []
        val = 1
        while val <= n:
            skeleton.append(val)
            val *= 4
        
        # Initialize with skeleton
        for x in skeleton:
            # Check if valid (should be, but let's be safe)
            # For powers of 4, all triple sums are unique.
            # We just add them.
            # But we need to update the sums array.
            # Since they are powers of 4, we can just mark the sums.
            # However, for simplicity and correctness, let's use the same incremental check.
            # But since we know they are valid, we can just add them directly.
            # Let's verify: if we add 1, then 4, then 16...
            # Sums for {1}: {3}
            # Sums for {1,4}: {3, 2+1+4=7, 1+1+4=6? No, unordered triple (a,b,c) with a<=b<=c.
            # (1,1,1)=3, (1,1,4)=6, (1,4,4)=9, (4,4,4)=12.
            # All unique.
            # So yes, we can add them directly.
            
            # Add all sums involving x and previous selected elements.
            # Since selected is small initially, this is fast.
            new_sums = []
            # (x,x,x)
            s = 3*x
            if not sums[s]:
                sums[s] = True
                new_sums.append(s)
            else:
                # Should not happen for skeleton
                return selected
            
            for y in selected:
                # (x,x,y)
                s1 = 2*x + y
                if not sums[s1]:
                    sums[s1] = True
                    new_sums.append(s1)
                else:
                    return selected
                # (x,y,y)
                s2 = x + 2*y
                if not sums[s2]:
                    sums[s2] = True
                    new_sums.append(s2)
                else:
                    return selected
            for i in range(len(selected)):
                y = selected[i]
                for j in range(i+1, len(selected)):
                    z = selected[j]
                    s3 = x + y + z
                    if not sums[s3]:
                        sums[s3] = True
                        new_sums.append(s3)
                    else:
                        return selected
            selected.append(x)
        
        # Now, greedily add remaining elements in a random order.
        # Create a list of all candidates not in skeleton.
        candidates = [x for x in range(1, n+1) if x not in skeleton]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            valid = True
            new_sums = []
            
            # (x,x,x)
            s = 3*x
            if sums[s]:
                valid = False
            else:
                new_sums.append(s)
            
            if valid:
                for y in selected:
                    # (x,x,y)
                    s1 = 2*x + y
                    if sums[s1]:
                        valid = False
                        break
                    new_sums.append(s1)
                    
                    # (x,y,y)
                    s2 = x + 2*y
                    if sums[s2]:
                        valid = False
                        break
                    new_sums.append(s2)
                
                if valid:
                    k = len(selected)
                    for i in range(k):
                        y = selected[i]
                        for j in range(i+1, k):
                            z = selected[j]
                            s3 = x + y + z
                            if sums[s3]:
                                valid = False
                                break
                            new_sums.append(s3)
                        if not valid:
                            break
            
            if valid:
                for s in new_sums:
                    sums[s] = True
                selected.append(x)
        
        return selected
    
    # Try multiple random restarts
    # Start with a few deterministic restarts, then random ones.
    # The current best is 1.7778, which corresponds to a set of size 16 (since 16/9 = 1.7778, assuming seed size is 9? Or is the score value/seed where value is the size? The task says "The evaluator measures the raw value (higher is better). The search score is value/seed". The current program returns a list. The score 1.7778 suggests the size is 16 if the seed size is 9, or 1.7778 * seed. Let's assume the "value" is the length of the list. If the seed program returns 9 elements, then 16/9 = 1.7778. So we are currently at 16 elements.)
    # We want to beat 16.
    
    best_size = 0
    best_set = []
    
    # Try a few different strategies in parallel (sequentially within the time limit)
    
    # Strategy 1: Pure greedy from 1 to n (deterministic)
    # This is the baseline. We already know it gives 16.
    # Let's try random restarts.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Use a different seed for each restart
        result = try_construct(i)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # If we still have time, try more restarts with different skeleton variations?
    # Or just more random restarts.
    
    while time.time() < deadline - 1.0:
        # Random restart
        seed = random.randint(0, 1000000)
        result = try_construct(seed)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    return best_set
# EVOLVE-BLOCK-END

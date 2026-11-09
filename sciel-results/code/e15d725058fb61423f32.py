import time
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    primes_set = set(primes)
    
    N = 10000
    # We want to maximize L such that 1..L is covered.
    # The score is capped at 10^5, but practically with sum(1/p)<=1, L won't be huge.
    # Let's aim for L around 1000-2000.
    
    # Generate candidate prime sets
    # Strategy:
    # 1. Greedy smallest
    # 2. Greedy largest (within budget)
    # 3. Randomized with constraints
    
    candidate_sets = []
    
    # Set 1: Smallest primes
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 2: Skip 2
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if p == 2: continue
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 3: Skip 2, 3
    ps = []
    s = Fraction(0,1)
    for p in primes:
        if p in (2,3): continue
        if s + Fraction(1,p) <= 1:
            ps.append(p)
            s += Fraction(1,p)
    candidate_sets.append(ps)
    
    # Set 4: Mix of small and medium
    # Try to include 2, 3, 5, 7, 11, 13... but maybe swap some for larger ones?
    # Actually, let's just generate random valid sets.
    
    random.seed(42)
    for _ in range(20):
        ps = []
        s = Fraction(0,1)
        # Shuffle primes, but maybe bias towards smaller ones?
        # Let's just shuffle all primes
        shuffled = primes[:]
        random.shuffle(shuffled)
        for p in shuffled:
            if s + Fraction(1,p) <= 1:
                ps.append(p)
                s += Fraction(1,p)
        if ps:
            candidate_sets.append(ps)
            
    # Deduplicate
    unique_sets = []
    seen = set()
    for ps in candidate_sets:
        key = tuple(sorted(ps))
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)

    best_L = 0
    best_classes = []
    
    def optimize_residues(prime_list, time_limit):
        if not prime_list:
            return 0, []
            
        k = len(prime_list)
        # Precompute indices for each prime and residue
        # This allows O(1) access to the list of numbers covered by (p, r)
        prime_indices = []
        for p in prime_list:
            indices_for_p = []
            for r in range(p):
                idxs = []
                start = r if r > 0 else p
                for x in range(start, N + 1, p):
                    idxs.append(x)
                indices_for_p.append(idxs)
            prime_indices.append(indices_for_p)
        
        # State:
        # residues: list of current residues
        # counts: array of size N+1, count of primes covering x
        # min_gap: smallest x in 1..N such that counts[x] == 0. If all covered, N+1.
        
        # Initialize
        residues = [random.randint(0, p-1) for p in prime_list]
        counts = [0] * (N + 1)
        
        for i, p in enumerate(prime_list):
            r = residues[i]
            for x in prime_indices[i][r]:
                counts[x] += 1
        
        # Find initial min_gap
        min_gap = 1
        while min_gap <= N and counts[min_gap] > 0:
            min_gap += 1
        
        current_L = min_gap - 1
        best_L_local = current_L
        best_residues = residues[:]
        
        rng = random.Random(123)
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        # SA Loop
        # Move: Change residue of one prime
        while time.time() < time_limit and T > T_min:
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]
            
            # Choose new residue
            # Bias towards covering min_gap
            if min_gap <= N and rng.random() < 0.6:
                new_r = min_gap % p
            else:
                new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Apply move
            # 1. Remove old coverage
            old_indices = prime_indices[i][old_r]
            new_indices = prime_indices[i][new_r]
            
            # We need to track if min_gap changes.
            # min_gap changes if:
            # a) We uncover a number < min_gap -> min_gap becomes that number.
            # b) We cover min_gap -> min_gap increases (need to scan forward).
            # c) We cover/uncover numbers > min_gap -> min_gap stays same.
            
            # To be safe and simple, we can just update counts and then update min_gap.
            # But updating min_gap efficiently:
            
            # Let's collect changes
            # We will just update counts first.
            for x in old_indices:
                counts[x] -= 1
            for x in new_indices:
                counts[x] += 1
            
            residues[i] = new_r
            
            # Update min_gap
            # Case 1: Did we uncover a number < min_gap?
            # We can check if any x in old_indices has counts[x]==0 and x < min_gap.
            # Actually, if we uncover x < min_gap, min_gap becomes x.
            # If we uncover x == min_gap, min_gap stays min_gap (it's still a gap).
            # If we cover x == min_gap, min_gap increases.
            
            # Efficient update:
            # 1. Check if any x in old_indices where counts[x]==0 and x < min_gap.
            #    If so, min_gap = min(min_gap, min(x for x in old_indices if counts[x]==0 and x < min_gap))
            #    Actually, if we uncover x < min_gap, it becomes the new min_gap.
            #    Since old_indices are sparse, this check is fast.
            
            new_min_gap = min_gap
            
            # Check for new gaps < current min_gap
            # Only possible if we removed coverage (old_indices)
            for x in old_indices:
                if counts[x] == 0 and x < new_min_gap:
                    new_min_gap = x
                    break # Since we just want the minimum, and we iterate old_indices, 
                    # wait, old_indices is not sorted by value? 
                    # prime_indices[i][r] is generated in increasing order: start, start+p, ...
                    # So the first gap found is the smallest gap in that residue class.
                    # But we need the global smallest gap.
                    # If we found a gap x < new_min_gap, it's a candidate.
                    # We should check all old_indices? 
                    # No, if we find one, we can update new_min_gap to x.
                    # But there might be a smaller one?
                    # Since old_indices is sorted, the first gap we encounter is the smallest gap in that set.
                    # But we need to compare with existing min_gap.
                    # Actually, if we uncover x, and x < min_gap, then x is the new min_gap.
                    # Is it possible to uncover multiple numbers < min_gap? Yes.
                    # The smallest one is the new min_gap.
                    # Since old_indices is sorted, we can just take the first one that is a gap and < min_gap.
                    # Wait, if we uncover x1 and x2, both < min_gap, x1 < x2, then x1 is the new min_gap.
                    # So iterating old_indices (sorted) and breaking on first gap < min_gap is correct.
            
            # Case 2: Did we cover min_gap?
            # If counts[min_gap] > 0 (it was covered before, or covered now), and it was the min_gap.
            # If min_gap was a gap (counts[min_gap]==0 before), and we covered it (counts[min_gap]>0 now),
            # then min_gap increases.
            
            # We need to know if min_gap was a gap before the move.
            # We can store old_min_gap.
            
            # Let's refine the logic:
            
            # 1. Determine if min_gap is still a gap.
            # 2. If it is a gap, check if we uncovered anything smaller.
            # 3. If it is covered, scan forward to find next gap.
            
            # This is getting complicated to do incrementally without storing old counts.
            # Given N=10000, scanning forward is O(N) worst case, but usually small.
            # Scanning old_indices is O(N/p).
            
            # Let's just do a hybrid:
            # 1. If we uncovered any x < min_gap, new_min_gap = min(uncovered x).
            # 2. If counts[min_gap] == 0, new_min_gap = min_gap (it's still a gap, or we uncovered it).
            # 3. If counts[min_gap] > 0, we need to find the next gap.
            
            # Let's implement this carefully.
            
            # 1. Check for new gaps < min_gap
            # We only remove coverage from old_indices.
            # So any new gap < min_gap must be in old_indices.
            # old_indices is sorted.
            for x in old_indices:
                if x >= min_gap:
                    break
                if counts[x] == 0:
                    new_min_gap = x
                    break
            
            # 2. If min_gap is now covered, find next gap
            if counts[min_gap] > 0:
                # min_gap is covered. Scan forward.
                # Start from min_gap
                next_gap = min_gap
                while next_gap <= N and counts[next_gap] > 0:
                    next_gap += 1
                new_min_gap = next_gap
            
            # 3. If min_gap was a gap and is still a gap, and we didn't find a smaller gap,
            #    new_min_gap remains min_gap.
            #    This is handled by initialization new_min_gap = min_gap and only updating in steps 1 and 2.
            
            # Wait, what if min_gap was a gap, we covered it, but we also uncovered x < min_gap?
            # Step 1 finds x < min_gap. new_min_gap = x.
            # Step 2 checks counts[min_gap]. It is > 0. It scans forward.
            # But we already have a smaller gap x. So new_min_gap should be x.
            # So we should only do Step 2 if new_min_gap is still min_gap (i.e., no smaller gap found) OR if min_gap was covered.
            # Actually, if we found a smaller gap x, min_gap (the old one) is irrelevant.
            # So:
            
            if new_min_gap == min_gap and counts[min_gap] > 0:
                 # This implies min_gap was covered and no smaller gap was found.
                 # But wait, if min_gap was covered, it was NOT a gap before?
                 # No, min_gap is defined as the FIRST gap. So counts[min_gap] was 0.
                 # If counts[min_gap] is now > 0, it means we covered it.
                 # So we must scan forward.
                 next_gap = min_gap
                 while next_gap <= N and counts[next_gap] > 0:
                     next_gap += 1
                 new_min_gap = next_gap
            
            # What if min_gap was a gap, we didn't cover it, but we uncovered x < min_gap?
            # Step 1 sets new_min_gap = x.
            # Step 2: counts[min_gap] is 0. Condition `counts[min_gap] > 0` is false.
            # So new_min_gap remains x. Correct.
            
            # What if min_gap was a gap, we didn't cover it, and we didn't uncover x < min_gap?
            # Step 1: no change.
            # Step 2: counts[min_gap] is 0. Condition false.
            # new_min_gap remains min_gap. Correct.
            
            # What if min_gap was a gap, we covered it, and we uncovered x < min_gap?
            # Step 1: new_min_gap = x.
            # Step 2: counts[min_gap] > 0. Condition true.
            # It scans forward from min_gap and finds y > min_gap.
            # It sets new_min_gap = y.
            # But x < min_gap < y. So new_min_gap should be x.
            # So we need to fix Step 2 to not override a smaller gap.
            
            # Revised Step 2:
            if counts[min_gap] > 0:
                 next_gap = min_gap
                 while next_gap <= N and counts[next_gap] > 0:
                     next_gap += 1
                 if next_gap < new_min_gap:
                     new_min_gap = next_gap
                 # Wait, if we uncovered x < min_gap, new_min_gap is x.
                 # next_gap is > min_gap. So next_gap > new_min_gap.
                 # So `if next_gap < new_min_gap` will be false.
                 # So new_min_gap stays x. Correct.
            
            min_gap = new_min_gap
            new_L = min_gap - 1
            
            if new_L > current_L or rng.random() < math.exp((new_L - current_L) / T):
                current_L = new_L
                if current_L > best_L_local:
                    best_L_local = current_L
                    best_residues = residues[:]
            else:
                # Revert
                for x in new_indices:
                    counts[x] -= 1
                for x in old_indices:
                    counts[x] += 1
                residues[i] = old_r
                
                # Revert min_gap logic is hard to do incrementally in reverse.
                # Since we reverted counts, we can just recalculate min_gap?
                # Or, since we only accept if it improves, and revert if not,
                # the state is restored. But min_gap variable needs to be restored.
                # We can store old_min_gap.
                
                # Let's just recalculate min_gap on revert?
                # That's O(N). Too slow if we revert often.
                # Better: Store old_min_gap.
                
                # Actually, if we revert, the counts are back to previous state.
                # So min_gap is back to previous min_gap.
                # We can just restore min_gap = old_min_gap.
                
                # Let's add old_min_gap storage.
                
            T *= cooling
            
        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_L_local, classes

    # We need to fix the revert logic in the SA loop above.
    # I will rewrite the SA loop in the actual code to store old_min_gap.
    
    # Let's re-write optimize_residues properly in the final code.
    
    pass # Placeholder, will replace with full implementation below

    # Full Implementation of optimize_residues
    def optimize_residues(prime_list, time_limit):
        if not prime_list:
            return 0, []
            
        k = len(prime_list)
        prime_indices = []
        for p in prime_list:
            indices_for_p = []
            for r in range(p):
                idxs = []
                start = r if r > 0 else p
                for x in range(start, N + 1, p):
                    idxs.append(x)
                indices_for_p.append(idxs)
            prime_indices.append(indices_for_p)
        
        residues = [random.randint(0, p-1) for p in prime_list]
        counts = [0] * (N + 1)
        
        for i, p in enumerate(prime_list):
            r = residues[i]
            for x in prime_indices[i][r]:
                counts[x] += 1
        
        min_gap = 1
        while min_gap <= N and counts[min_gap] > 0:
            min_gap += 1
        
        current_L = min_gap - 1
        best_L_local = current_L
        best_residues = residues[:]
        
        rng = random.Random(123)
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < time_limit and T > T_min:
            i = rng.randint(0, k - 1)
            p = prime_list[i]
            old_r = residues[i]
            old_min_gap = min_gap
            
            if min_gap <= N and rng.random() < 0.6:
                new_r = min_gap % p
            else:
                new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            old_indices = prime_indices[i][old_r]
            new_indices = prime_indices[i][new_r]
            
            for x in old_indices:
                counts[x] -= 1
            for x in new_indices:
                counts[x] += 1
            
            residues[i] = new_r
            
            new_min_gap = min_gap
            
            # 1. Check for new gaps < min_gap
            for x in old_indices:
                if x >= min_gap:
                    break
                if counts[x] == 0:
                    new_min_gap = x
                    break
            
            # 2. If min_gap is now covered, find next gap
            if counts[min_gap] > 0:
                 next_gap = min_gap
                 while next_gap <= N and counts[next_gap] > 0:
                     next_gap += 1
                 if next_gap < new_min_gap:
                     new_min_gap = next_gap
            
            min_gap = new_min_gap
            new_L = min_gap - 1
            
            if new_L > current_L or rng.random() < math.exp((new_L - current_L) / T):
                current_L = new_L
                if current_L > best_L_local:
                    best_L_local = current_L
                    best_residues = residues[:]
            else:
                # Revert
                for x in new_indices:
                    counts[x] -= 1
                for x in old_indices:
                    counts[x] += 1
                residues[i] = old_r
                min_gap = old_min_gap
                
            T *= cooling
            
        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_L_local, classes

    # Run optimization for each set
    time_per_set = (deadline - time.time()) / len(unique_sets)
    
    for ps in unique_sets:
        if time.time() > deadline - 5:
            break
        t_limit = time.time() + min(time_per_set, deadline - time.time() - 5)
        L, classes = optimize_residues(ps, t_limit)
        if L > best_L:
            best_L = L
            best_classes = classes

    # Final check
    if best_classes:
        s = Fraction(0,1)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s <= 1:
            return best_classes
    return [[2, 0]]

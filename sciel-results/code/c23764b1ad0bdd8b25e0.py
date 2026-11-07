    def construct():
        # We want to maximize size.
        # Greedy high-to-low.
        
        # S[k] is a set of sums of subsets of size k.
        # Initially S[0] = {0}, S[k>0] = empty.
        
        # Max possible size is 60.
        # We will try to add numbers from 256 down to 1.
        
        # To be efficient, we only need to store S[k] for k up to current size.
        # But size grows.
        
        # Let's maintain a list of sets.
        # S = [set(), set(), ...]
        # S[0] = {0}
        
        # We need to check validity before adding.
        
        # Optimization:
        # Instead of checking all pairs (i, j), we can just compute the new sets and check intersections.
        
        # Let current set of numbers be A.
        # Let S be the list of sum sets.
        
        # Candidate x.
        # New S' where S'[k] = S[k] U (S[k-1] + x).
        
        # Check:
        # For all i, j such that i+1 != j:
        # (S[i] + x) is disjoint from S[j].
        
        # Note: S[i] + x is the set {s + x for s in S[i]}.
        
        # We can iterate i from 0 to len(S)-1.
        # For each i, compute NewPart = {s + x for s in S[i]}.
        # This NewPart will be added to S[i+1].
        
        # We need to ensure NewPart is disjoint from S[j] for all j != i+1.
        
        # Also, we need to ensure that NewParts from different i's don't collide?
        # We established that if old S's are disjoint, new parts from different i's are disjoint.
        # Proof:
        # Part_i = S[i] + x. Part_j = S[j] + x.
        # If p in Part_i and Part_j, then s1 + x = s2 + x => s1 = s2.
        # s1 in S[i], s2 in S[j].
        # If i != j, S[i] and S[j] are disjoint, so s1 != s2.
        # So Part_i and Part_j are disjoint.
        
        # So we only need to check Part_i against existing S[j] for j != i+1.
        
        # Algorithm:
        # 1. Init S = [{0}]
        # 2. For x in 256 down to 1:
        #    a. Check validity:
        #       valid = True
        #       for i in range(len(S)):
        #           # Part_i = S[i] + x
        #           # We need Part_i disjoint from S[j] for all j != i+1
        #           # Note: S has length L. i ranges 0..L-1.
        #           # i+1 ranges 1..L.
        #           # S[j] exists for j in 0..L-1.
        #           # If i+1 >= L, then S[i+1] doesn't exist yet (it will be created).
        #           # So we check against all existing S[j].
        #           
        #           # Optimization: If Part_i intersects S[j], invalid.
        #           
        #           # We can just compute the intersection.
        #           
        #       If valid:
        #           b. Update S:
        #              # New S will have length L+1.
        #              # NewS[k] = S[k] U (S[k-1] + x)
        #              # Note: S[-1] is empty.
        #              
        #              # We can build new list.
        #              new_S = []
        #              for k in range(L+1):
        #                  new_set = set()
        #                  if k < L:
        #                      new_set.update(S[k])
        #                  if k > 0:
        #                      # Add S[k-1] + x
        #                      for s in S[k-1]:
        #                          new_set.add(s + x)
        #                  new_S.append(new_set)
        #              S = new_S
        #              # Add x to our result list
        #              result.append(x)
        
        # Return result.
        
        # Time complexity:
        # Outer loop: 256 iterations.
        # Inner check: Sum of sizes of S[i].
        # Size of S[i] is at most C(n, i).
        # Total sums is 2^n.
        # If n grows to 30, 2^30 is 1 billion. Too slow.
        
        # We need to be smarter or limit n.
        # The insight said 31 elements.
        # 2^31 is 2 billion.
        # Storing sets of ints for 31 elements might be heavy but Python sets are reasonably fast.
        # However, iterating through all sums for each candidate x (256 times) is expensive.
        
        # Can we prune?
        # If we just want to maximize size, we can stop if we reach a certain size?
        # No, we want max size.
        
        # Maybe we don't need to check all x from 256 down to 1.
        # The greedy choice is "pick the largest available x that is valid".
        
        # Let's try to run this and see if it's fast enough.
        # Deadline is 90s.
        
        # Optimization:
        # Use bitsets?
        # Max sum for 60 elements of 256 is 15360.
        # We can use Python integers as bitsets.
        # Bit i is 1 if sum i is possible.
        
        # S[i] as an integer bitmask.
        # S[i] has bit s set if sum s is in S[i].
        
        # Part_i = S[i] << x.
        
        # Check intersection:
        # (S[i] << x) & S[j] == 0 for all j != i+1.
        
        # This is MUCH faster.
        
        # Let's rewrite with bitmasks.
        
        # S is a list of integers (bitmasks).
        # S[0] = 1 (bit 0 set).
        
        # For x in 256 down to 1:
        #   valid = True
        #   for i in range(len(S)):
        #       part = S[i] << x
        #       for j in range(len(S)):
        #           if j == i + 1:
        #               continue
        #           if part & S[j]:
        #               valid = False
        #               break
        #       if not valid:
        #           break
        #   
        #   if valid:
        #       # Update S
        #       new_S = []
        #       L = len(S)
        #       for k in range(L + 1):
        #           mask = 0
        #           if k < L:
        #               mask |= S[k]
        #           if k > 0:
        #               mask |= (S[k-1] << x)
        #           new_S.append(mask)
        #       S = new_S
        #       result.append(x)
        
        # This should be very fast.
        # Max sum 15360. Integers of 15360 bits are small (2KB).
        # Bitwise ops are native C speed.
        
        # Let's trace complexity.
        # 256 candidates.
        # For each candidate, check up to L^2 intersections.
        # L is current size. Max 60.
        # 60*60 = 3600 checks.
        # Each check is a bitwise AND on 2KB integers.
        # 3600 * 256 * 2KB ops.
        # Very fast.
        
        # We should be able to find the optimal set quickly.
        
        # One detail: "distinct integers in 1..256".
        # The loop `for x in range(256, 0, -1)` handles this.
        
        # We need to return the list of ints.
        
        # Let's double check the logic.
        # "Whenever two subsets have different cardinalities their sums must differ."
        # My check ensures that new sums (formed by adding x to subsets of size i, resulting in size i+1) do not collide with existing sums of size j (where j != i+1).
        # Collisions with size i+1 (same cardinality) are allowed.
        # Collisions between new sums of different sizes (e.g. new size i+1 and new size k+1 where i!=k) are prevented because old S[i] and S[k] were disjoint, so shifted versions are disjoint.
        
        # This logic is sound.
        
        # Implementation details:
        # - Use a deadline guard.
        # - Return list of Python ints.
        
        # Let's write the code.

    def construct():
        import time
        
        start_time = time.time()
        deadline = start_time + 85  # 5s margin
        
        # S[k] is a bitmask where bit s is 1 if sum s is achievable with subset of size k.
        # S[0] = 1 (bit 0)
        S = [1]
        result = []
        
        # Try adding numbers from 256 down to 1
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
                
            valid = True
            L = len(S)
            
            # Check if adding x causes any cross-cardinality collision
            for i in range(L):
                # New sums of size i+1 generated by adding x to subsets of size i
                new_part = S[i] << x
                
                # Check against all existing S[j]
                for j in range(L):
                    # If j == i+1, collision is allowed (same cardinality)
                    if j == i + 1:
                        continue
                    
                    # If intersection is non-zero, collision found
                    if new_part & S[j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x to result
                result.append(x)
                
                # Update S
                # New S will have length L+1
                new_S = []
                for k in range(L + 1):
                    mask = 0
                    # Keep old sums of size k (if k < L)
                    if k < L:
                        mask |= S[k]
                    # Add new sums of size k (from size k-1 + x)
                    if k > 0:
                        mask |= (S[k-1] << x)
                    new_S.append(mask)
                S = new_S
                
        return result
    
def construct():
    # The optimal set for this problem is the largest set of consecutive integers
    # at the top of the range [1, 256] such that the minimum sum of 2 elements
    # is strictly greater than the maximum sum of 1 element.
    # Let the set be {k, k+1, ..., 256}.
    # Max sum of 1 element: 256.
    # Min sum of 2 elements: k + (k+1) = 2k + 1.
    # Condition: 2k + 1 > 256 => 2k > 255 => k >= 128.
    # So the set is {128, 129, ..., 256}.
    # Size: 256 - 128 + 1 = 129.
    
    # We should verify if higher cardinality constraints break this.
    # Min sum of 3: 3k + 3. Max sum of 2: 256 + 255 = 511.
    # 3(128) + 3 = 387. 387 < 511.
    # Wait, 387 is NOT > 511.
    # So S_3 and S_2 might collide?
    # Min S_3 = 128+129+130 = 387.
    # Max S_2 = 256+255 = 511.
    # Is there a collision?
    # We need S_2 ∩ S_3 = empty.
    # This requires Min(S_3) > Max(S_2) OR no overlaps.
    # 387 < 511, so ranges overlap.
    # We must check if any sum of 3 equals any sum of 2.
    # Sum of 3: a+b+c. Sum of 2: d+e.
    # a,b,c,d,e in {128..256}.
    # Min S_3 = 387. Max S_2 = 511.
    # Example: 128+129+130 = 387.
    # Is 387 a sum of 2?
    # 387 = 128 + 259 (259 not in set).
    # 387 = 131 + 256. 131 in set, 256 in set.
    # So 387 is in S_2 (131+256) and in S_3 (128+129+130).
    # Collision!
    # So {128..256} is INVALID.
    
    # The current greedy program would have rejected 128?
    # Let's re-trace.
    # Start with 256.
    # Add 255.
    # ...
    # Add 129. Set {129..256}.
    # Try 128.
    # Check collisions.
    # One collision: 128+129+130 = 387.
    # 387 is in S_2 (from 131+256).
    # When adding 128:
    # We generate new sums.
    # s=130, c=2 (from 128+130? No, 128 is new).
    # s=130 is in current. c=1? No, 130 is in S_1.
    # ns = 130+128 = 258. c+1=2.
    # Is 258 in new?
    # 258 = 132+126? No.
    # 258 = 258? No.
    # 258 = 129+129? No distinct.
    # 258 = 130+128.
    # Is 258 in S_1? No.
    # Is 258 in S_2?
    # 258 = 132+126 (126 not in set).
    # 258 = 131+127 (127 not in set).
    # 258 = 130+128 (this is the new sum).
    # So 258 is not in current S_2.
    # 
    # Check 387.
    # s=130, c=1? No, s=130 is in S_1.
    # ns=258.
    # s=129, c=1. ns=257.
    # s=131, c=1. ns=259.
    # 
    # We need to check if 387 is generated.
    # 387 = 128 + 259? No.
    # 387 = 128 + 129 + 130.
    # This involves 128.
    # In the loop for x=128:
    # We iterate s in current.
    # To get 387, we need s=259, c=2.
    # Is 259 in current?
    # 259 = 129+130. Yes, c=2.
    # ns = 259 + 128 = 387.
    # c+1 = 3.
    # Is 387 in new?
    # 387 = 131+256.
    # 131 is in current (S_1). 256 is in current (S_1).
    # So 131+256 was generated when adding 256? No, when adding 131?
    # 131+256 is in S_2 of current.
    # So 387 is in current with c=2.
    # So new[387] = 2.
    # We are adding 387 with c+1=3.
    # 2 != 3.
    # ok = False.
    # So 128 is rejected.
    
    # So the greedy program stops at 129.
    # Set {129..256}.
    # Size 128.
    # Is this valid?
    # Min S_2 = 129+130=259. Max S_1=256. 259>256. OK.
    # Min S_3 = 129+130+131=390. Max S_2 = 256+255=511.
    # Check collision 390.
    # 390 = 132+258? 258 not in set.
    # 390 = 134+256.
    # 134 in S_1, 256 in S_1.
    # So 390 in S_2.
    # 390 in S_3 (129+130+131).
    # Collision!
    # So {129..256} is INVALID.
    
    # The greedy program would reject 129?
    # When adding 129 to {130..256}:
    # s=131, c=1? No, 131 in S_1.
    # s=130, c=1.
    # We need to check 390.
    # 390 = 129+131+130.
    # s=261 (131+130), c=2.
    # ns=261+129=390. c+1=3.
    # Is 390 in current?
    # 390 = 134+256. Yes, c=2.
    # 2 != 3. Reject 129.
    
    # So the greedy program rejects 129.
    # It keeps {130..256}.
    # Size 127.
    # Check validity.
    # Min S_2 = 130+131=261. Max S_1=256. OK.
    # Min S_3 = 130+131+132=393. Max S_2=256+255=511.
    # 393 = 137+256.
    # 137 in S_1, 256 in S_1.
    # 393 in S_2.
    # 393 in S_3.
    # Collision.
    # Reject 130.
    
    # It seems the greedy program will reject many.
    # It will likely end up with a small set.
    # This explains the low score.
    
    # To improve, we need a set that avoids these collisions.
    # Superincreasing sets avoid them.
    # Powers of 2: {1, 2, 4, ..., 128}. Size 8.
    # Or {128, 256}?
    # 128+256=384.
    # S_1: {128, 256}.
    # S_2: {384}.
    # Disjoint.
    # Size 2.
    
    # Can we get more?
    # {1, 2, 4, 8, 16, 32, 64, 128, 256}. Size 9.
    # S_1: {1, 2, 4, 8, 16, 32, 64, 128, 256}.
    # S_2: {3, 5, 6, 9, 10, 12, 17, ...}.
    # Max S_1 = 256.
    # Min S_2 = 3.
    # Collision? 3 not in S_1.
    # 5 not in S_1.
    # ...
    # 256+128=384.
    # All S_2 sums are distinct and not in S_1?
    # Yes, for powers of 2, binary representation is unique.
    # Sum of k powers of 2 has exactly k bits set.
    # So S_k contains numbers with exactly k bits set.
    # S_i and S_j are disjoint for i != j.
    # So {1, 2, ..., 256} (powers of 2) is valid.
    # Size 9.
    
    # Can we do better than 9?
    # We can replace 1 with something else?
    # If we use {2, 4, 8, ..., 256}, size 8.
    # If we use {1, 2, 4, ..., 128, 255}?
    # 255 = 256-1.
    # 1+255=256. Collision S_2 and S_1.
    
    # What if we use {1, 2, 4, 8, 16, 32, 64, 128, 200}?
    # 128+200=328.
    # 1+200=201.
    # Is 201 in S_1? No.
    # Is 201 in S_2?
    # 201 = 1+200.
    # Is 201 in S_k for k>2?
    # 201 = 1+2+4+...+128+...?
    # 201 = 128+64+8+1.
    # 128, 64, 8, 1 are in the set.
    # So 201 is in S_4.
    # 201 is in S_2 (1+200).
    # Collision S_2 and S_4.
    
    # So we need to be careful.
    # The powers of 2 set is very safe.
    # Size 9.
    
    # Can we add 512? No, max 256.
    # Can we add 256 to {1..128}?
    # {1, 2, 4, 8, 16, 32, 64, 128, 256}.
    # This is valid. Size 9.
    
    # Is there a set of size 10?
    # We need 10 distinct integers in 1..256.
    # Superincreasing: $a_i > \sum_{j<i} a_j$.
    # 1, 2, 4, 8, 16, 32, 64, 128.
    # Next must be > 255.
    # 256 works.
    # Next must be > 255+256=511.
    # No number in 1..256.
    # So max superincreasing size is 9.
    
    # Can we use non-superincreasing?
    # Yes, if we avoid collisions.
    # But collisions are likely.
    
    # I will return the powers of 2 set.
    # It is valid and size 9.
    # The greedy program likely returns size < 9 due to collisions.
    # So this is an improvement.

    result = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    return result

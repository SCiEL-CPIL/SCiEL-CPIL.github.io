# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    # We aim to find the maximum cardinality subset of [1, 22]
    # such that all reciprocal subset sums are distinct.
    
    # We will search for k from 15 down to 1.
    # For each k, we use a backtracking search with an efficient
    # integer-based subset sum check.
    
    def search_max_size(n_max=22, max_k=15):
        # Candidates are 1..n_max
        candidates = list(range(1, n_max + 1))
        
        # We will try to find a set of size max_k.
        # If not found, return None.
        
        # Backtracking state:
        # current_set: list of chosen numbers
        # current_lcm: LCM of current_set
        # current_sums: set of integer subset sums scaled by current_lcm
        #   i.e., { sum_{i in T} (current_lcm / a_i) for T subset of current_set }
        
        # To optimize, we can sort candidates.
        # Does order matter? 
        # If we pick smaller numbers first, LCM grows faster?
        # If we pick larger numbers first, LCM might stay smaller longer?
        # Let's try sorting in descending order or ascending order.
        # Actually, for distinct subset sums of reciprocals, 
        # picking "independent" elements is key.
        # Powers of 2 are very independent.
        
        # Let's just iterate through combinations for small k?
        # No, backtracking with pruning is better.
        
        def backtrack(start_idx, current_set, current_lcm, current_sums, target_size):
            if len(current_set) == target_size:
                return list(current_set)
            
            # Pruning: if remaining elements are not enough
            remaining = n_max - start_idx
            if remaining < target_size - len(current_set):
                return None
            
            # Iterate through candidates from start_idx to end
            # We can try to be smart about which candidates to pick.
            # Let's just iterate normally.
            
            for i in range(start_idx, n_max + 1):
                x = candidates[i-1] # candidates is 1..n_max
                
                new_lcm = math.lcm(current_lcm, x)
                
                # Update sums
                # Old sums S_old are for LCM_old.
                # New sums S_new are for LCM_new.
                # Scale factor M = new_lcm / current_lcm
                M = new_lcm // current_lcm
                
                # The new set of sums consists of:
                # 1. Sums not including x: s * M for s in current_sums
                # 2. Sums including x: s * M + new_lcm // x for s in current_sums
                
                # We need to check if all these 2 * len(current_sums) values are distinct.
                # We can generate them and check for duplicates.
                
                # Optimization: 
                # If M == 1, then new_lcm == current_lcm.
                # Then S_new = current_sums union (current_sums + new_lcm//x).
                # We just need to check if (current_sums + offset) intersects current_sums.
                
                # General case:
                # Create a set of new sums.
                # If size < 2 * len(current_sums), there's a collision.
                
                # To avoid creating huge lists if collision is likely early,
                # we can check intersection.
                
                base_sum = new_lcm // x
                
                # Generate new sums
                # We can do this efficiently by checking if any s1*M + base == s2*M 
                # or s1*M == s2*M + base?
                # Actually, just building the set is fine for small sizes.
                
                # However, building a set of size 2^k at each node is expensive.
                # But k is small (<= 15). 2^15 = 32768.
                # The number of nodes might be large though.
                
                # Let's try to build the new sum set.
                new_sums = set()
                # To speed up, we can use a list and then check uniqueness?
                # Or just add to set and check size.
                
                # We can optimize by noting that if M > 1, the sums are scaled.
                
                # Let's just build the set.
                # If it takes too long, we might need a different heuristic.
                
                # Heuristic: 
                # If we add x, the "granularity" changes.
                
                # Let's try to add x.
                # We'll compute the new sums.
                
                # To avoid modifying current_sums, we create a new one.
                
                # Optimization: 
                # If we just want to check for collisions, we can do:
                # Check if any s1 * M + base == s2 * M for s1, s2 in current_sums?
                # This implies (s1 - s2) * M == -base.
                # And s1 * M == s2 * M + base? Same condition.
                # And collisions within the "not x" group: s1*M == s2*M => s1==s2 (already distinct).
                # And collisions within the "x" group: s1*M + base == s2*M + base => s1==s2 (already distinct).
                # So we only need to check if the two sets intersect!
                
                # Set A = { s * M for s in current_sums }
                # Set B = { s * M + base for s in current_sums }
                # Check if A and B are disjoint.
                
                # This is much faster than building the full set and checking size.
                
                # How to check if A and B are disjoint?
                # A = { s*M }
                # B = { t*M + base }
                # Intersection exists if there is s, t in current_sums such that:
                # s*M = t*M + base
                # M*(s - t) = base
                # s - t = base / M
                # Since s, t are integers, base must be divisible by M.
                
                if base % M == 0:
                    diff = base // M
                    # Check if there exist s, t in current_sums such that s - t = diff
                    # This is equivalent to checking if (current_sums - diff) intersects current_sums.
                    # Or: for s in current_sums, if s - diff in current_sums, collision.
                    
                    # This is an O(N) check where N = len(current_sums).
                    collision = False
                    # To make it faster, we can iterate over the smaller set or just one.
                    # We need to check if any s in current_sums has (s - diff) in current_sums.
                    # We can do this by iterating current_sums and checking membership in current_sums.
                    # O(N).
                    
                    # But wait, current_sums is a set, so membership is O(1).
                    # So total O(N).
                    
                    for s in current_sums:
                        if (s - diff) in current_sums:
                            collision = True
                            break
                    
                    if collision:
                        continue
                else:
                    # If base % M != 0, then s*M and t*M + base can never be equal
                    # because LHS is multiple of M, RHS is not.
                    # So no collision between A and B.
                    pass
                
                # If no collision, we can proceed.
                # But we still need to maintain the new_sums set for the next recursion.
                # We can construct it now.
                
                # new_sums = { s * M for s in current_sums } | { s * M + base for s in current_sums }
                
                # To save memory/time, we can just pass the new set.
                # Constructing it takes O(N).
                
                new_sums_set = set()
                # We can optimize construction if M==1
                if M == 1:
                    # A = current_sums
                    # B = { s + base for s in current_sums }
                    for s in current_sums:
                        new_sums_set.add(s)
                        new_sums_set.add(s + base)
                else:
                    for s in current_sums:
                        val = s * M
                        new_sums_set.add(val)
                        new_sums_set.add(val + base)
                
                current_set.append(x)
                result = backtrack(i + 1, current_set, new_lcm, new_sums_set, target_size)
                if result:
                    return result
                current_set.pop()
            
            return None
        
        # Start search
        # Initial state: empty set, LCM=1, sums={0}
        # Note: The empty sum is 0.
        
        # We want to find a set of size k.
        # We try k from max_k down to 1.
        
        for k in range(max_k, 0, -1):
            # Reset state for each k? 
            # The search is independent for each k.
            res = backtrack(1, [], 1, {0}, k)
            if res:
                return res
        
        return None

    # Try to find a large set
    # Start with 15
    result = search_max_size(22, 15)
    
    if result:
        return result
    
    # Fallback to known good sets if search fails or times out (though we don't have time limit here easily)
    # The previous program found 14. 
    # If my search is too slow, I should return a hardcoded 14 or 15 if I can verify one.
    
    # Let's try a quick check for a hardcoded size 15 set?
    # I don't have one.
    
    # If the search returns None, it means no set of size 15 was found.
    # It might be that the search was too slow or exhaustive.
    # If it's exhaustive and found nothing, then max size < 15.
    
    # To be safe and fast, if the search for 15 is too slow, we might timeout.
    # But the logic is optimized.
    
    # If result is None, try 14?
    # The previous program found 14.
    # Let's assume if 15 fails, we return a known 14 set or search for 14.
    
    if not result:
        # Search for 14
        result = search_max_size(22, 14)
    
    if not result:
        # Fallback
        result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return result
# EVOLVE-BLOCK-END

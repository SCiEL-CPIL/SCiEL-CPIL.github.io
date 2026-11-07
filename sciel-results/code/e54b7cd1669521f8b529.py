import time
import random
import sys

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s (or 90s) kill
    random.seed(42)
    
    # Max possible sum: 30 * 256 = 7680
    MAX_SUM = 7680
    
    # Precompute forbidden multiples bitsets
    # forbidden[k] is a bitset where bit j is set if j is a multiple of k (j >= 2k)
    # Actually, we need: for a new sum s, check if any existing sum t divides s or s divides t
    # Let's precompute for each value v (1..MAX_SUM), the set of values that are multiples of v
    # multiples_of[v] = bitset with bits set at positions 2v, 3v, ... up to MAX_SUM
    # And also the set of divisors: divisors_of[v] = bitset with bits at positions d where d|v and d < v
    
    # Actually, let's think about this differently.
    # When we add a new element x, we generate new sums: for each existing sum s, new sum = s + x
    # We need to check:
    # 1. No collision: new_sums ∩ current_sums = ∅
    # 2. No divisibility: for any a in new_sums, b in new_sums: a ∤ b and b ∤ a (a ≠ b)
    #    for any a in new_sums, b in current_sums: a ∤ b and b ∤ a
    
    # Let's use bitsets. current_sums is a bitset where bit s is set if sum s is present.
    # new_sums = {s + x : s in current_sums}
    
    # For divisibility check, precompute:
    # div_mask[v] = bitset with bit d set for all d that divide v (d < v, d > 0)
    # mult_mask[v] = bitset with bit m set for all m that are multiples of v (m > v, m <= MAX_SUM)
    
    # When adding new sums, for each new sum ns:
    # - Check if any existing sum divides ns: (current_sums & div_mask[ns]) != 0
    # - Check if ns divides any existing sum: (current_sums & mult_mask[ns]) != 0
    # - For pairs within new_sums: need to check all pairs, which is harder with bitsets
    
    # Alternative: precompute for each value v, a bitset bad_with[v] such that
    # bad_with[v] has bit u set if u divides v or v divides u (u ≠ v, u > 0)
    # Then for new sums, we can check: for each ns in new_sums, (current_sums & bad_with[ns]) != 0
    # But for new_sums vs new_sums, we still need pairwise checks.
    
    # Let's precompute bad_with for all values 1..MAX_SUM
    bad_with = [0] * (MAX_SUM + 1)
    for v in range(1, MAX_SUM + 1):
        mask = 0
        # All proper divisors of v
        for d in range(1, int(v**0.5) + 1):
            if v % d == 0:
                if d < v:
                    mask |= (1 << d)
                other = v // d
                if other < v and other != d:
                    mask |= (1 << other)
        # All multiples of v greater than v
        for m in range(2 * v, MAX_SUM + 1, v):
            mask |= (1 << m)
        bad_with[v] = mask
    
    # Now, for a set of sums represented as a bitset, and a new sum ns:
    # Check against existing: (existing_bitset & bad_with[ns]) != 0
    
    # For new_sums vs new_sums, we need to check pairs. Since new_sums size is at most |current_sums|,
    # and |current_sums| can be large, this is still O(n^2) in the worst case.
    # But we can do it more efficiently: for each ns in new_sums, check if any other ns' in new_sums is in bad_with[ns]
    # i.e., (new_sums_bitset & bad_with[ns]) != 0, but we need to exclude ns itself.
    
    # Let's try a randomized greedy approach with random restarts.
    
    best_size = 0
    best_elements = []
    
    def try_build(order):
        """Try to build a set by greedily adding elements in the given order."""
        elements = []
        sums_bitset = 1  # bit 0 set (sum 0)
        sums_set = {0}
        
        for x in order:
            # Generate new sums
            new_sums_set = set()
            for s in sums_set:
                ns = s + x
                if ns <= MAX_SUM:
                    new_sums_set.add(ns)
            
            # Check collision
            if new_sums_set & sums_set:
                continue
            
            # Check divisibility with existing sums
            valid = True
            for ns in new_sums_set:
                if (sums_bitset & bad_with[ns]) != 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check divisibility within new sums
            ns_list = list(new_sums_set)
            for i in range(len(ns_list)):
                a = ns_list[i]
                if a == 0:
                    continue
                for j in range(i + 1, len(ns_list)):
                    b = ns_list[j]
                    if b % a == 0 or a % b == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Add x
            elements.append(x)
            sums_set |= new_sums_set
            for ns in new_sums_set:
                sums_bitset |= (1 << ns)
        
        return elements
    
    # Try different random orderings
    while time.time() < deadline:
        # Create a random ordering of candidates
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Also try some structured orderings
        result = try_build(candidates)
        if len(result) > best_size:
            best_size = len(result)
            best_elements = list(result)
        
        # Also try descending order
        if best_size < 10:
            result = try_build(list(range(256, 0, -1)))
            if len(result) > best_size:
                best_size = len(result)
                best_elements = list(result)
        
        # Try ascending
        if best_size < 10:
            result = try_build(list(range(1, 257)))
            if len(result) > best_size:
                best_size = len(result)
                best_elements = list(result)
    
    return list(best_elements)

if __name__ == "__main__":
    print(construct())

import time
import sys

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    
    Strategy:
    1. Represent all subset sums as integers scaled by L = LCM(1..22).
       This avoids floating-point precision issues entirely.
       L = 232792560.
       The sum of reciprocals of any subset S is (sum_{x in S} L/x) / L.
       Distinctness of reciprocal sums is equivalent to distinctness of the integer numerators.
    2. Use a greedy approach to build the set.
       To maximize cardinality, we want to include as many elements as possible.
       Elements with smaller reciprocals (larger indices) are "cheaper" in terms of range but might be more prone to collision if not spaced well?
       Actually, for dissociated sets, large elements (small reciprocals) are often easier to pack because the sums grow slowly.
       However, small elements (large reciprocals) create large gaps.
       
       Let's analyze the collision condition:
       A collision occurs if there exist two distinct subsets A, B of the current set S such that sum(A) = sum(B).
       When adding a new element x, a collision occurs if there exists a subset A of S such that sum(A) = sum(B) + 1/x for some subset B of S?
       No, the new sums are {s} U {s + 1/x for s in S_sums}.
       A collision happens if (s + 1/x) is already in S_sums.
       i.e., s1 + 1/x = s2 for s1, s2 in S_sums.
       => 1/x = s2 - s1.
       So we just need to check if 1/x (scaled to integer L/x) is in the set of all differences of existing sums.
       Alternatively, we can just maintain the set of all current sums and check if the shifted set intersects the original set.
       
       Since L is fixed, we work with integers.
       Max sum is H_22 * L ~ 3.7 * 2.3e8 ~ 8.6e8. This fits easily in a 64-bit integer.
       
       Greedy Heuristic:
       We want to maximize the size K.
       Let's try to build the set by adding elements one by one.
       Which order?
       If we add small numbers first (1, 2, 3...), the sums grow quickly.
       If we add large numbers first (22, 21, ...), the sums grow slowly.
       
       Let's test both orders with a simple greedy:
       Start with empty set, sums = {0}.
       For candidate x in order:
           new_sums = sums U {s + L//x for s in sums}
           if len(new_sums) == 2 * len(sums): # No collision
               add x to set
               sums = new_sums
       
       We will try multiple orderings and keep the one that yields the largest set.
       Orderings to try:
       1. Descending (22 down to 1)
       2. Ascending (1 up to 22)
       3. Alternating or other simple heuristics?
       
       Given the small N=22, we can even try a few random permutations if time permits, but greedy is O(N * 2^K) which is very fast for K~10-12.
       2^12 = 4096. 22 * 4096 ~ 90k operations. Instant.
       
       To be safe and potentially better, we can also try a "repair" or "local search" if greedy gets stuck, but for N=22, greedy usually finds a good dissociated set.
       Actually, let's just try all single-element removals from the full set? No, we don't have a full set.
       
       Let's stick to trying a few deterministic orderings.
    """
    
    # LCM of 1 to 22
    # 1..22
    # 22 = 2 * 11
    # 21 = 3 * 7
    # 20 = 4 * 5
    # 19 = 19
    # 18 = 2 * 9
    # 17 = 17
    # 16 = 16
    # 15 = 3 * 5
    # 14 = 2 * 7
    # 13 = 13
    # 12 = 4 * 3
    # 11 = 11
    # 10 = 2 * 5
    # 9 = 9
    # 8 = 8
    # 7 = 7
    # 6 = 2 * 3
    # 5 = 5
    # 4 = 4
    # 3 = 3
    # 2 = 2
    # 1 = 1
    # LCM = 16 * 9 * 5 * 7 * 11 * 13 * 17 * 19
    # 16 * 9 = 144
    # 144 * 5 = 720
    # 720 * 7 = 5040
    # 5040 * 11 = 55440
    # 55440 * 13 = 720720
    # 720720 * 17 = 12252240
    # 12252240 * 19 = 232792560
    L = 232792560
    
    # Precompute scaled reciprocals
    # rec[i] = L // i
    rec = [0] * 23
    for i in range(1, 23):
        rec[i] = L // i
        
    def greedy_build(order):
        """
        Builds a dissociated set using the given order of elements.
        Returns the list of elements in the set.
        """
        current_set = []
        sums = {0}
        
        for x in order:
            if _time_left() < 1.0:
                break
            rx = rec[x]
            # Check if adding x causes collision
            # New sums are sums U {s + rx for s in sums}
            # Collision if (s + rx) is in sums for any s in sums
            # i.e., if (sums + rx) intersects sums
            
            # Efficient check:
            # Since sums is a set of ints, we can generate the shifted set and check intersection.
            # Or simpler:
            # for s in sums:
            #    if s + rx in sums: collision
            
            # Optimization:
            # We can just create the new set and check its size.
            # len(new_sums) == 2 * len(sums)
            
            # To avoid creating the set if it fails, we can check intersection first.
            # But creating the set is O(2^k). Checking intersection is O(2^k).
            # Creating the set is necessary if we succeed.
            
            # Let's try to be smart:
            # If min(sums) + rx > max(sums), then no collision?
            # No, sums are not contiguous.
            
            # Just do the standard check.
            # For N=22, max size ~10-12, 2^12=4096. This is very fast.
            
            collision = False
            # We can break early if we find a collision
            # But we need to build the new set anyway if no collision.
            # So let's just build the new set and check size.
            # It's cleaner and likely fast enough.
            
            new_sums = set(sums)
            # Add shifted sums
            for s in sums:
                ns = s + rx
                new_sums.add(ns)
            
            if len(new_sums) == 2 * len(sums):
                # No collision
                current_set.append(x)
                sums = new_sums
            # else: collision, skip x
        
        return current_set

    # Define orderings
    # 1. Descending
    order_desc = list(range(22, 0, -1))
    # 2. Ascending
    order_asc = list(range(1, 23))
    # 3. Alternating small/large?
    # Let's just try a few simple ones.
    
    candidates = []
    
    # Try Descending
    s1 = greedy_build(order_desc)
    candidates.append(s1)
    
    # Try Ascending
    s2 = greedy_build(order_asc)
    candidates.append(s2)
    
    # Try a "mixed" order: even then odd? Or prime then composite?
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19
    # Composites: 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22
    # Actually, let's just try a random shuffle if we have time, but deterministic is safer for reproducibility.
    
    # Another heuristic: try to add elements that are "far" from existing sums?
    # That's complex. Let's stick to simple orderings.
    
    # Let's try one more: "balanced" order
    # 11, 12, 10, 13, 9, 14, ...
    order_balanced = []
    l, r = 1, 22
    while l <= r:
        order_balanced.append(l)
        if l != r:
            order_balanced.append(r)
        l += 1
        r -= 1
    s3 = greedy_build(order_balanced)
    candidates.append(s3)
    
    # Select the best candidate (largest size)
    best_set = max(candidates, key=len)
    
    # If best_set is empty (should not happen, [1] is always valid), return [1]
    if not best_set:
        return [1]
        
    # Final verification (just in case)
    # Check distinctness
    sums = {0}
    valid = True
    for x in best_set:
        rx = rec[x]
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + rx)
        if len(new_sums) != 2 * len(sums):
            valid = False
            break
        sums = new_sums
        
    if valid:
        return best_set
    else:
        # Fallback to a known valid small set
        return [1]

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    L = 232792560
    sums = {0}
    for x in s:
        rx = L // x
        new_sums = {ss + rx for ss in sums}
        if len(new_sums) != 2 * len(sums):
            print("INVALID!")
        sums = new_sums
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")

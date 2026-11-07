import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp_mask[a] is a bitset where bit b is 1 if a and b are comparable
    # Note: A set is always comparable with itself (a & b == a).
    # We want strict subset, but for the purpose of counting pairs in a family of distinct sets,
    # self-comparability doesn't add extra pairs because sets are distinct.
    # However, if we use bitsets to count "neighbors", we must be careful.
    # Let's define comp_mask[a] to include all b such that a subset b OR b subset a.
    
    comp_mask = [0] * N
    for a in range(N):
        mask = 0
        for b in range(N):
            if (a & b) == a or (a & b) == b:
                mask |= (1 << b)
        comp_mask[a] = mask
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    def count_comparable(family):
        """
        family: list of masks
        Returns number of comparable pairs.
        """
        # Build a bitset of the family
        fam_mask = 0
        for x in family:
            fam_mask |= (1 << x)
        
        count = 0
        for x in family:
            # Count how many sets in family are comparable to x
            # This includes x itself, so subtract 1
            # Since all sets in family are distinct, each comparable pair (x, y) is counted twice 
            # (once for x, once for y).
            c = popcount(comp_mask[x] & fam_mask)
            count += c - 1
        
        return count // 2

    def greedy_construct():
        """
        Construct a family greedily by adding the set that maximizes new comparable pairs.
        """
        family = []
        fam_mask = 0
        
        for i in range(m):
            best_val = -1
            best_x = -1
            
            # Heuristic: start with a random or small set? 
            # Let's try a few random starting points for the first element if i==0
            # Or just pick 0 for determinism in greedy, but random restarts will handle diversity.
            # For i=0, any set is equivalent. Let's pick 0.
            if i == 0:
                best_x = 0
            else:
                # Iterate through all candidates not in family
                # Optimization: We can just iterate all N and check if in fam_mask
                # N=4096, m=40, so 40*4096 = 163k checks. Fast.
                for x in range(N):
                    if (fam_mask >> x) & 1:
                        continue
                    
                    # Number of new comparable pairs = number of sets in family comparable to x
                    new_pairs = popcount(comp_mask[x] & fam_mask)
                    
                    if new_pairs > best_val:
                        best_val = new_pairs
                        best_x = x
            
            family.append(best_x)
            fam_mask |= (1 << best_x)
            
        return family

    def local_search(family, time_limit):
        """
        Local search using bitsets for O(1) score change calculation.
        """
        current_score = count_comparable(family)
        best_score = current_score
        best_family = family[:]
        fam_mask = 0
        for x in family:
            fam_mask |= (1 << x)
        
        # Precompute a list of indices for random access? No, list is fine.
        # We will try to improve by swapping one element with another not in family.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try each position
            for i in range(m):
                x = family[i]
                # Current contribution of x to the score (excluding self)
                # Actually, let's just calculate the score change for swapping x with y.
                # Score change = (new pairs formed by y) - (pairs lost by x)
                # Pairs lost by x: number of other sets in family comparable to x.
                # Since x is in family, popcount(comp_mask[x] & fam_mask) includes x.
                # So pairs involving x = popcount(comp_mask[x] & fam_mask) - 1.
                # Let's call this k_x.
                k_x = popcount(comp_mask[x] & fam_mask) - 1
                
                best_y = -1
                best_delta = 0
                
                # Find best y not in family
                # Optimization: We can iterate through all y and skip those in fam_mask.
                for y in range(N):
                    if (fam_mask >> y) & 1:
                        continue
                    
                    # Pairs gained by y: number of sets in family (excluding x, since x is being removed)
                    # comparable to y.
                    # Let fam_mask_without_x = fam_mask ^ (1 << x)
                    # gain = popcount(comp_mask[y] & fam_mask_without_x)
                    # But wait, if y is comparable to x, that pair is lost when x is removed and gained when y is added?
                    # No.
                    # Old family F. New family F' = F \ {x} U {y}.
                    # Pairs in F involving x: k_x.
                    # Pairs in F' involving y: 
                    #   y is comparable with z in F \ {x}.
                    #   gain = popcount(comp_mask[y] & (fam_mask ^ (1 << x)))
                    
                    # Delta = gain - k_x
                    
                    # We can compute gain quickly.
                    # Note: comp_mask[y] & (fam_mask ^ (1 << x))
                    # If y is comparable to x, then x is in comp_mask[y].
                    # So popcount(comp_mask[y] & fam_mask) includes x if comparable.
                    # Let total_y = popcount(comp_mask[y] & fam_mask)
                    # If y comparable to x, gain = total_y - 1 (exclude x)
                    # Else gain = total_y
                    
                    total_y = popcount(comp_mask[y] & fam_mask)
                    if (comp_mask[y] >> x) & 1:
                        gain = total_y - 1
                    else:
                        gain = total_y
                        
                    delta = gain - k_x
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_y = y
                
                if best_y != -1 and best_delta > 0:
                    # Perform swap
                    fam_mask ^= (1 << x)
                    fam_mask |= (1 << best_y)
                    family[i] = best_y
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_family = family[:]
                    improved = True
                    # Break out of inner loop? No, keep checking other i's in this pass.
                    # But updating fam_mask means subsequent checks in this pass use new fam_mask.
                    # This is "First Improvement" style but we update immediately.
                    # Actually, if we update immediately, the 'k_x' for subsequent i's might be affected if they were comparable to x.
                    # This is fine, it's a valid local search step.
            
            # If no improvement in a full pass, try random swaps to escape local minima?
            # For now, let's just break.
            if not improved:
                break
                
        return best_family, best_score

    best_overall_family = None
    best_overall_score = 0
    
    # 1. Greedy Construction
    g_family = greedy_construct()
    g_score = count_comparable(g_family)
    best_overall_family = g_family[:]
    best_overall_score = g_score
    
    # 2. Local Search from Greedy
    ls_family, ls_score = local_search(g_family[:], deadline - 5.0)
    if ls_score > best_overall_score:
        best_overall_score = ls_score
        best_overall_family = ls_family[:]
        
    # 3. Random Restarts with Greedy/Local Search
    # Since greedy is fast, we can try many random variations.
    # Variation: Pick a random starting set for the first element, then greedy.
    
    while time.time() < deadline - 10.0:
        # Random start
        random.seed(int(time.time() * 1000000))
        
        # Generate a random initial family? 
        # Or just randomize the greedy process slightly?
        # Let's try: Pick a random set as the first element.
        start_set = random.randint(0, N - 1)
        
        family = [start_set]
        fam_mask = 1 << start_set
        
        for i in range(1, m):
            best_val = -1
            best_x = -1
            for x in range(N):
                if (fam_mask >> x) & 1:
                    continue
                new_pairs = popcount(comp_mask[x] & fam_mask)
                if new_pairs > best_val:
                    best_val = new_pairs
                    best_x = x
            family.append(best_x)
            fam_mask |= (1 << best_x)
            
        # Local search from this random-greedy start
        ls_family, ls_score = local_search(family, deadline - 5.0)
        
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_family = ls_family[:]

    # 4. Final Polish with more aggressive local search if time permits
    if time.time() < deadline - 2.0:
        # Run local search again on the best found, maybe with a different strategy or just more passes
        # The previous local search might have stopped early.
        # Let's just run it again with a tight deadline.
        ls_family, ls_score = local_search(best_overall_family[:], deadline - 1.0)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_family = ls_family[:]

    # Convert masks to lists of integers
    result = []
    for mask in best_overall_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result

# Ensure types are correct
# The problem states: "Types are checked exactly: integers must be Python int... sequences must be list or tuple"
# Our result is a list of lists of Python ints. This is correct.

if __name__ == "__main__":
    res = construct()
    # Optional: verify
    # assert len(res) == 40
    # assert len(set(tuple(r) for r in res)) == 40
    # for r in res:
    #     assert len(r) == len(set(r))
    #     for x in r:
    #         assert isinstance(x, int)
    #         assert not isinstance(x, bool)
    #         assert 0 <= x < 12

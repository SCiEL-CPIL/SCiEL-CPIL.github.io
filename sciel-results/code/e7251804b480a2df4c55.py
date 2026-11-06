"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    # Deadline guard: return best found before 145s wall-clock
    t0 = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Identify the candidate set.
    # Key Insight: The optimal set consists ONLY of primes.
    # Proof: Let a be a composite number in the set. Let p be a prime factor of a.
    # Then p < a. The term 1/(512-a) < 1/(512-p).
    # If we replace a with p, we must ensure p is coprime to all other elements.
    # Since p divides a, and all other elements are coprime to a, they are coprime to p.
    # Thus, replacing a with p is valid and strictly increases the score.
    # Therefore, the optimal set is a subset of primes < 512.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(511)
    n_primes = len(primes)
    
    # 2. Construct the conflict graph.
    # Two primes p, q conflict if gcd(p, q) != 1.
    # Since p and q are distinct primes, gcd(p, q) = 1 always.
    # Wait! The problem says "pairwise coprime integers".
    # If we pick distinct primes, they are automatically pairwise coprime.
    # So the conflict graph for distinct primes is EMPTY (no edges).
    # This means the problem reduces to selecting a subset of primes to maximize the sum.
    # Since all weights 1/(512-p) are positive, we should select ALL primes!
    
    # Let's double check.
    # "distinct pairwise coprime integers a in 1..511"
    # If I pick {2, 3}, gcd(2,3)=1. OK.
    # If I pick {2, 4}, gcd(2,4)=2. Not OK.
    # So I cannot pick 2 and 4 together.
    # But I CAN pick 2 and 3.
    # Can I pick a composite number instead of its prime factors?
    # Suppose I pick 6. Score: 1/506.
    # If I pick 2 and 3. Score: 1/510 + 1/509.
    # 1/510 + 1/509 > 1/506?
    # 0.00196 + 0.00196 = 0.00392.
    # 1/506 = 0.00197.
    # Yes, splitting composites into their prime factors increases the sum significantly.
    # Generally, $\sum_{p|a} \frac{1}{512-p} > \frac{1}{512-a}$ for composite $a$.
    # Because $a \ge p_1 p_2 \dots p_k$.
    # Actually, even simpler: any composite $a$ has a prime factor $p$.
    # $p < a \implies 512-p > 512-a \implies \frac{1}{512-p} < \frac{1}{512-a}$.
    # Wait. $1/(512-p)$ is LARGER than $1/(512-a)$ because the denominator is smaller.
    # So replacing $a$ with $p$ increases the term.
    # But we must maintain pairwise coprimality.
    # If we replace $a$ with $p$, does $p$ conflict with other elements?
    # If the original set was pairwise coprime, no other element is divisible by any prime factor of $a$.
    # So $p$ will be coprime to all other elements.
    # Thus, we can replace $a$ with $p$ and improve the score.
    # We can repeat this until all elements are primes.
    # So the optimal set is a subset of primes.
    # And since distinct primes are pairwise coprime, we can pick ANY subset of distinct primes.
    # To maximize the sum of positive terms, we pick ALL of them.
    
    # So the solution is simply all primes less than 512.
    
    # Let's verify if there's any catch.
    # "distinct pairwise coprime integers"
    # Yes, distinct primes are pairwise coprime.
    # Is there any reason NOT to pick a prime? No, all contributions are positive.
    # So the greedy strategy of "pick all primes" is optimal.
    
    # Why did the previous program score 1.0?
    # Maybe it didn't pick all primes?
    # The previous program used a greedy baseline that iterated 511 down to 1.
    # It would pick 511 (prime), then 509 (prime), etc.
    # It would skip composites if they shared factors with previously picked numbers.
    # But since it picked primes first (largest numbers), it likely picked most primes.
    # However, if it picked a composite early on, it might block a prime?
    # No, it iterates from 511 down.
    # 511 is 7*73. It picks 511.
    # Then 510 (gcd(510,511)=1). It picks 510?
    # 510 = 2*3*5*17.
    # Then 509 (prime). gcd(509, 511)=1, gcd(509, 510)=1. Picks 509.
    # Then 508. gcd(508, 510) = gcd(4*127, 2*255) = 2. Conflict. Skips 508.
    # So the greedy baseline picks composites too!
    # And composites have LOWER value than their prime factors.
    # Example: 510 vs 2,3,5,17.
    # 1/(512-510) = 1/2 = 0.5.
    # 1/(512-2) + 1/(512-3) + 1/(512-5) + 1/(512-17) = 1/510 + 1/509 + 1/507 + 1/495.
    # 0.00196 + 0.00196 + 0.00197 + 0.00202 = 0.0079.
    # 0.5 >> 0.0079.
    # WAIT. My previous deduction was WRONG.
    # $1/(512-a)$ is LARGE when $a$ is LARGE.
    # $a=510 \implies 1/2 = 0.5$.
    # $a=17 \implies 1/495 \approx 0.002$.
    # So a single large composite number has a MUCH higher score than its small prime factors!
    # My proof that "optimal set is primes" is FALSE.
    # The term $1/(512-a)$ favors large $a$.
    # So we want large numbers.
    # But they must be pairwise coprime.
    
    # So the problem is: Select a subset of $\{1, \dots, 511\}$ that is pairwise coprime to maximize $\sum \frac{1}{512-a}$.
    
    # This is a Maximum Weight Independent Set problem on a graph where:
    # Nodes: 1..511
    # Edge between u, v if gcd(u, v) > 1.
    # Weight of node u: $1/(512-u)$.
    
    # Since the weights are very skewed towards large numbers, a greedy approach
    # that prioritizes large numbers is likely good, but we can do better with local search.
    
    # Let's re-evaluate the "primes only" hypothesis.
    # If I pick 511 (7*73), score 1.
    # If I pick 7 and 73, score 1/505 + 1/439 = 0.00198 + 0.00227 = 0.00425.
    # 1 >> 0.00425.
    # So picking 511 is much better than picking its factors.
    # So we DO want large composites if they are coprime to other large numbers.
    
    # So the strategy should be:
    # 1. Greedy: Sort by weight descending (i.e., value descending, which is $a$ descending).
    #    Add to set if coprime to all existing.
    # 2. Local Search: Try to swap or add/remove to improve.
    
    # The current program does this. Why did it score 1.0?
    # Maybe it's already near optimal?
    # Or maybe the local search is too slow/weak.
    
    # Let's try a different structural approach:
    # Branch and Bound or Integer Programming?
    # With 511 nodes, B&B is too slow without strong pruning.
    # However, the conflict graph is sparse for large numbers?
    # No, many large numbers share small factors.
    
    # Alternative: Simulated Annealing with a better neighborhood.
    # Or: Greedy with multiple random restarts and a stronger local search.
    
    # Let's look at the "Inspiration 3" which used prime factors.
    # It likely failed for the same reason I initially failed: it assumed primes are better.
    
    # Let's stick to the Greedy + Local Search but make it MUCH more robust.
    # Key improvement:
    # 1. Precompute gcd conflicts for all pairs? 511*511/2 = ~130k pairs. Fast.
    # 2. Use bitmasks for conflict checking? 511 bits fits in 8 uint64s or Python ints.
    #    Python ints are arbitrary precision, so we can use a single int as a bitmask.
    #    This makes "is coprime to all in set" very fast (bitwise AND).
    
    # Representation:
    # conflict_mask[a] = integer with bits set for all b != a such that gcd(a,b) > 1.
    # selected_mask = integer with bits set for selected numbers.
    # Check if a can be added: (conflict_mask[a] & selected_mask) == 0.
    
    # This will speed up the greedy and local search significantly.
    
    # Let's implement this.
    
    # Precompute conflict masks
    # conflict_mask[a] will be an int.
    # Bit i is 1 if gcd(a, i) > 1.
    
    conflict_masks = [0] * 512
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if a == b:
                continue
            if gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
    
    # Weights
    weights = [0.0] * 512
    for a in range(1, 512):
        weights[a] = 1.0 / (512.0 - a)
    
    # Greedy Baseline with Bitmask
    def greedy_baseline():
        selected_mask = 0
        selected_set = set()
        # Iterate from largest weight to smallest (a from 511 down to 1)
        for a in range(511, 0, -1):
            if (conflict_masks[a] & selected_mask) == 0:
                selected_mask |= (1 << a)
                selected_set.add(a)
        return selected_set
    
    # Local Search with Bitmask
    def local_search(selected_set, max_time):
        selected_list = list(selected_set)
        selected_mask = 0
        for a in selected_list:
            selected_mask |= (1 << a)
        
        current_score = sum(weights[a] for a in selected_list)
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            # Try to add any unselected node that is compatible
            for a in range(511, 0, -1):
                if a in selected_set:
                    continue
                if (conflict_masks[a] & selected_mask) == 0:
                    # Add it
                    selected_set.add(a)
                    selected_mask |= (1 << a)
                    current_score += weights[a]
                    improved = True
                    # Restart the loop to check for more additions
                    break 
            
            if improved:
                continue
                
            # Try to swap: remove one, add one or more?
            # Simple swap: remove x, add y if y is compatible with (set - x)
            # This is O(N^2).
            # Better: for each x in set, for each y not in set, if y compatible with set\x
            # This is expensive.
            
            # Alternative: Random restarts of Greedy.
            # The greedy is deterministic. We need randomness.
            # Let's do random permutations of the "add" order.
            
        return selected_set, current_score
    
    # Actually, the "Add if compatible" loop above is just a greedy fill.
    # The initial greedy already did that.
    # So local search needs to handle swaps.
    
    # Let's try a different structure:
    # 1. Run Greedy (deterministic, descending a).
    # 2. Run many Random Greedy restarts.
    #    - Shuffle the order of candidates.
    #    - Pick greedily.
    #    - This explores different local optima.
    # 3. Combine best results? No, just keep the best single set.
    
    best_set = greedy_baseline()
    best_score = sum(weights[a] for a in best_set)
    
    # Random Restart Greedy
    import random
    random.seed(42)
    
    t_start = time.time()
    max_time = 140.0
    
    # Number of restarts depends on time.
    # Each restart is O(N^2) with bitmasks? No, O(N * (N/64)) ~ O(N).
    # With N=511, it's very fast.
    
    for i in range(10000):
        if time.time() - t_start > max_time:
            break
            
        # Create a random priority
        # We want to bias towards large numbers but allow some randomness
        # to escape local optima.
        # Or just shuffle 1..511.
        candidates = list(range(1, 512))
        random.shuffle(candidates)
        
        selected_mask = 0
        selected_set = set()
        for a in candidates:
            if (conflict_masks[a] & selected_mask) == 0:
                selected_mask |= (1 << a)
                selected_set.add(a)
        
        s = sum(weights[a] for a in selected_set)
        if s > best_score:
            best_score = s
            best_set = selected_set
            
    # Final local improvement on the best set found
    # Try to swap elements to improve score.
    # Since the search space is large, let's try a few simple swaps on the best set.
    
    def try_swaps(selected_set, max_time):
        selected_list = list(selected_set)
        current_score = sum(weights[a] for a in selected_list)
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            # For each element in the set, try to replace it with a larger one
            # that is compatible with the rest.
            for idx, x in enumerate(selected_list):
                if time.time() > max_time:
                    break
                others_mask = 0
                for y in selected_list:
                    if y != x:
                        others_mask |= (1 << y)
                
                # Find best y not in set, compatible with others_mask
                best_y = None
                best_gain = -1
                # Iterate candidates in descending order of weight (value)
                for y in range(511, 0, -1):
                    if y in selected_set:
                        continue
                    if (conflict_masks[y] & others_mask) == 0:
                        gain = weights[y] - weights[x]
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y
                            # Since we iterate y in descending weight, the first valid y
                            # with weight > weight[x] is the best for this x.
                            # But we should check if weight[y] > weight[x] first.
                            if weights[y] > weights[x]:
                                break
                            else:
                                # If weights[y] <= weights[x], we won't improve by swapping x for y
                                # unless y allows us to add MORE elements later?
                                # No, simple swap keeps size constant.
                                # So we only care if weight[y] > weight[x].
                                # Since y is descending, if weights[y] <= weights[x],
                                # all smaller y will also have <= weight.
                                # So we can break?
                                # No, smaller y might have higher weight?
                                # Weight is 1/(512-a). Higher a -> Higher weight.
                                # So descending a is descending weight.
                                # So if weights[y] <= weights[x], then for all y' < y, weights[y'] <= weights[y] <= weights[x].
                                # So we can break.
                                break
                
                if best_y is not None and best_gain > 0:
                    selected_list[idx] = best_y
                    selected_set.remove(x)
                    selected_set.add(best_y)
                    current_score += best_gain
                    improved = True
        
        return selected_set, current_score

    # Apply try_swaps to the best set
    # Note: try_swaps modifies the set. We should work on a copy.
    final_set, final_score = try_swaps(set(best_set), 145.0)
    
    if final_score > best_score:
        best_set = final_set
        best_score = final_score
        
    return list(best_set)

# EVOLVE-BLOCK-END

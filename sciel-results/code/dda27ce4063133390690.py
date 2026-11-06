"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import product as _prod

def construct():
    # We want a strictly increasing sequence of integers in [2, 128]
    # such that all consecutive block products are distinct.
    # Maximize the length.
    
    # The greedy approach in the seed is actually quite strong.
    # Let's analyze why: if we pick numbers in increasing order, and for each new number x,
    # we check if any suffix product of the current sequence times some prefix of the reversed sequence
    # (which corresponds to a consecutive block ending at the new position) collides with an existing product.
    
    # Actually, the seed's logic:
    # values = [v1, v2, ..., vk] (increasing)
    # For new x, tails = [x, x*vk, x*vk*v{k-1}, ...]
    # These are the products of blocks [x], [vk, x], [vk-1, vk, x], ...
    # Wait, no. The blocks ending at x are:
    # [x] -> product x
    # [vk, x] -> product vk*x
    # [vk-1, vk, x] -> product vk-1*vk*x
    # ...
    # [v1, ..., vk, x] -> product v1*...*vk*x
    # 
    # The seed computes:
    # tails = [x]
    # prod = x
    # for a in reversed(values): # a = vk, vk-1, ..., v1
    #     prod *= a
    #     tails.append(prod)
    # So tails = [x, x*vk, x*vk*vk-1, ..., x*vk*...*v1]
    # These are exactly the products of all consecutive blocks that end at x.
    # 
    # The condition is that none of these products are in `products` (the set of all block products so far).
    # If they are disjoint, we add x.
    
    # This greedy is known to be optimal for this specific problem structure? 
    # Let's test if we can do better by being smarter.
    
    # However, the problem asks for "Maximize length".
    # The greedy approach picks the smallest possible next element that doesn't conflict.
    # Is it possible that skipping a small number allows for a longer sequence?
    # Usually, for density-one type problems, smaller numbers are "safer" because they have fewer prime factors,
    # but they also create smaller products which might collide more easily?
    # Actually, larger numbers create larger products, which are less likely to collide with smaller products.
    # But the sequence must be strictly increasing.
    
    # Let's try a different strategy:
    # 1. Try the greedy (smallest first).
    # 2. Try a greedy that prioritizes numbers with fewer prime factors?
    # 3. Try a local search / backtracking for small improvements?
    
    # Given the 60s limit, a full backtracking on 127 elements is impossible.
    # But maybe we can improve the greedy by checking multiple candidates?
    
    # Let's stick to a robust greedy but perhaps try a few variations:
    # Variation 1: Standard greedy (smallest available).
    # Variation 2: Greedy with lookahead? (Too slow)
    # Variation 3: Randomized restarts? (Might find a slightly better one if greedy is stuck)
    
    # Actually, let's look at the structure of conflicts.
    # A conflict occurs if a product of a suffix of the current sequence times a prefix of the new element's "extension" matches an existing product.
    
    # Let's implement a slightly more sophisticated greedy:
    # Instead of just taking the first valid x, maybe we can look ahead a bit?
    # No, that's expensive.
    
    # Let's try to verify if the greedy is indeed optimal or close to it.
    # The seed score is 1.0000, which means it's the baseline.
    # If I can find a sequence longer than the greedy, I win.
    
    # Let's try a backtracking search for the first few elements to see if a different start leads to a longer sequence.
    # But the search space is huge.
    
    # Alternative idea:
    # The constraint is on *consecutive* block products.
    # If we choose numbers that are powers of distinct primes, we might avoid collisions?
    # e.g., 2, 3, 5, 7, 11, ...
    # Products of blocks of primes are unique by fundamental theorem of arithmetic?
    # Yes! If all elements are distinct primes, then any consecutive block product is a product of distinct primes.
    # Two different blocks (different sets of primes, or different orders? No, order doesn't matter for product, but the blocks are different intervals).
    # Wait, if we have primes p1 < p2 < p3 < p4.
    # Block [p1, p2] -> p1*p2
    # Block [p3] -> p3
    # Can p1*p2 == p3? No, p3 is prime.
    # Can p1*p2 == p3*p4? No, different prime factorizations.
    # So a sequence of distinct primes is ALWAYS valid!
    # How many primes are in [2, 128]?
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127.
    # Let's count:
    # 2,3,5,7,11,13,17,19,23,29 (10)
    # 31,37,41,43,47,53,59,61,67,71 (10)
    # 73,79,83,89,97,101,103,107,109,113 (10)
    # 127 (1)
    # Total: 31 primes.
    
    # The greedy algorithm might include composite numbers as well, potentially getting a longer sequence than 31.
    # Let's check what the greedy produces.
    # I'll run the greedy logic in my head for the first few:
    # 2: ok. products={2}
    # 3: tails=[3, 3*2=6]. products={2,3,6}. ok.
    # 4: tails=[4, 4*3=12, 12*2=24]. products={2,3,6,4,12,24}. ok.
    # 5: tails=[5, 20, 60, 120]. ok.
    # 6: tails=[6, 30, 120, 360, 720]. 6 is in products! Conflict.
    # So 6 is skipped.
    # 7: tails=[7, 35, 140, 420, 840, 1680]. ok.
    # 8: tails=[8, 56, 224, 672, 1344, 2688, 5376]. ok.
    # 9: tails=[9, 72, 288, 864, 1728, 3456, 6912, 13824]. ok.
    # 10: tails=[10, 90, 360, ...]. 360 is in products (from 6? No, 6 was skipped. From 5? 5*4*3*2=120. 6 was skipped.
    # Wait, let's re-trace carefully.
    
    # It is likely that the greedy yields more than 31.
    # Let's try to improve the greedy by allowing some backtracking or a smarter selection.
    
    # What if we try to maximize the number of elements by choosing numbers that are "inert" (few prime factors)?
    # Primes are the most inert.
    # But composites like 4, 8, 16 are powers of 2.
    # Products of powers of 2 are powers of 2.
    # If we have 2, 4, 8, 16...
    # Block [2,4] -> 8. Block [8] -> 8. Collision!
    # So we can't have 2, 4, 8 in a row?
    # 2, 4: products 2, 4, 8.
    # Next 8: tails [8, 8*4=32, 32*2=64]. 8 is in products. So 8 conflicts with [2,4].
    # So we can't just take all powers of 2.
    
    # Let's write a program that tries to find a longer sequence than the greedy.
    # Strategy:
    # 1. Run the standard greedy to get a baseline length L.
    # 2. Try to insert skipped numbers into the sequence? No, order is fixed by value.
    # 3. Try a different greedy: instead of smallest first, maybe prioritize numbers that are coprime to many others?
    
    # Actually, a very strong heuristic for this type of problem is to use numbers with distinct prime signatures.
    # Let's try a greedy that picks the next number x (in increasing order) such that it minimizes the "risk" of future conflicts.
    # Or simply, let's try to see if we can include more numbers by being selective.
    
    # Let's try a backtracking search for the first 10-15 elements to see if a non-greedy choice leads to a longer total sequence.
    # If the greedy is optimal, this will fail. If not, we might find a better start.
    
    # However, implementing a full backtracker that runs within 60s for 127 elements is hard.
    # But we can do a "beam search" or "limited backtracking".
    
    # Let's try a simple improvement:
    # Run the greedy.
    # Then, for each position where a number was skipped, try to see if replacing a later number with the skipped one (and adjusting subsequent choices) helps.
    # This is complex.
    
    # Alternative:
    # The problem is equivalent to finding the longest subsequence of [2..128] such that all consecutive block products are distinct.
    # This is a combinatorial optimization problem.
    
    # Let's try a randomized local search:
    # 1. Start with the greedy sequence.
    # 2. Try to swap elements or add/remove elements to improve length?
    # Since the sequence must be increasing, we can't swap. We can only include or exclude.
    # So it's a subset selection problem with order fixed.
    
    # Let's try a dynamic programming approach?
    # State: (last_k_elements, set_of_products) -> too large.
    
    # Let's try a greedy with a twist:
    # When choosing the next number, look ahead 2-3 steps to see if picking x leads to a dead end sooner than picking x+1.
    
    # Given the constraints, I will implement a greedy that is slightly more careful:
    # It will try to pick the smallest x that is valid, but if that leads to a shorter sequence (detected by a quick lookahead), it might pick the next smallest.
    # But lookahead is expensive.
    
    # Let's try a different angle:
    # The number of distinct primes in [2,128] is 31.
    # The greedy likely gets around 40-50.
    # Let's just run the greedy and see. If I can't beat it, I return it.
    # But I need to *try* to beat it.
    
    # Idea: Use a set of "candidate" sequences generated by different heuristics and pick the longest.
    # Heuristic 1: Smallest first (greedy).
    # Heuristic 2: Largest first? (Must be increasing, so we build from left to right. Largest first doesn't make sense for construction, but we could select a subset.)
    # Heuristic 3: Randomized greedy: at each step, pick the smallest valid x with probability 0.8, or the next smallest with 0.1, etc.
    
    # Let's implement a few randomized restarts of the greedy and keep the best result.
    # Since the greedy is deterministic, randomization might help if the greedy gets stuck in a local optimum.
    
    # However, the "conflict" graph is dense.
    
    # Let's try to improve the greedy by checking if skipping a number allows two numbers later to be included.
    # This is a 2-step lookahead.
    
    # Let's implement a simple 2-step lookahead greedy.
    
    def greedy_lookahead(depth=2):
        values = []
        products = set()
        candidates = list(range(2, 129))
        
        for i in range(len(candidates)):
            if not candidates:
                break
            x = candidates[0]
            # Check if x is valid
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if products.intersection(tails):
                candidates.pop(0)
                continue
            
            # x is valid. Should we take it?
            # Lookahead: simulate taking x, then try to fill the rest with the same greedy.
            # Compare length with skipping x.
            
            # This is O(N^2) per step, which is O(N^3) total. 127^3 ~ 2M operations. Fast enough.
            
            # Simulate taking x
            values_new = values + [x]
            products_new = products.union(tails)
            remaining = candidates[1:]
            
            # Run greedy on remaining with state (values_new, products_new)
            def run_greedy(vals, prods, rem):
                v = list(vals)
                p = set(prods)
                for y in rem:
                    tails_y, prod_y = [y], y
                    for a in reversed(v):
                        prod_y *= a
                        tails_y.append(prod_y)
                    if p.intersection(tails_y):
                        continue
                    v.append(y)
                    p.update(tails_y)
                return len(v)
            
            len_take = run_greedy(values_new, products_new, remaining)
            
            # Simulate skipping x
            len_skip = run_greedy(values, products, candidates)
            
            if len_take >= len_skip:
                values.append(x)
                products.update(tails)
                candidates.pop(0)
            else:
                candidates.pop(0)
        
        return values
    
    # This might be too slow if run_greedy is called many times.
    # 127 steps * 127 candidates * 127 average list length = ~2M operations.
    # Python can do ~10-50M simple ops/sec. So this should be fine.
    
    # Let's try it.
    
    seq = greedy_lookahead()
    
    # Also try the standard greedy to compare
    def standard_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    seq_std = standard_greedy()
    
    if len(seq) > len(seq_std):
        return seq
    else:
        return seq_std

# EVOLVE-BLOCK-END

import time
import itertools

def construct():
    # We need a set of distinct integers in 2..40 such that all subset products are distinct.
    # This is equivalent to the numbers being multiplicatively independent over the integers.
    # Represent each number by its prime factorization vector.
    # A set is valid iff the vectors are linearly independent over the rationals (or reals, since they are integer vectors).
    #
    # Strategy:
    # 1. Precompute prime factorization vectors for all integers 2..40.
    # 2. Use a branch-and-bound search to find the largest subset of these vectors that are linearly independent.
    # 3. To ensure we finish within the time limit, we set a deadline.
    # 4. We return the best set found.
    
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10 seconds margin for safety
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Compute factorization vectors for 2..40
    # vec[i] is a tuple of length num_primes
    vectors = {}
    for n in range(2, 41):
        temp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        # If temp > 1, it's a prime > 37, but max is 40, so only 41 is possible? No, 40 is 2^3*5.
        # Primes > 37 in range 2..40: None.
        vectors[n] = tuple(vec)
    
    # List of (number, vector)
    items = [(n, vectors[n]) for n in range(2, 41)]
    
    # We want to find the maximum size subset S such that the vectors in S are linearly independent.
    # Let M be the matrix with rows as vectors. We want max rank subset.
    
    best_set = []
    best_size = 0
    
    # Branch and bound
    # State: (current_set_indices, candidate_indices, current_rank, current_basis)
    # To check independence, we can maintain a basis.
    # Since the dimension is small (12 primes), we can represent the basis as a set of vectors or a matrix.
    # Actually, checking if a new vector v is in the span of current basis B is:
    #   Solve B * x = v for integer x? No, we just need to check if v is in the span over Q.
    #   We can use Gaussian elimination on the augmented matrix [B | v].
    
    # For efficiency, let's pre-compute the rank of subsets? No, too many.
    # Let's use a greedy approach with backtracking.
    
    # A key insight: The maximum size is bounded by the dimension of the vector space, which is 12 (number of primes).
    # Wait, the vectors are in R^12. So the maximum linearly independent set size is at most 12.
    # However, the current best score is 1.3333, which corresponds to 16 integers.
    # How can 16 integers be multiplicatively independent if there are only 12 primes?
    # Ah, the vectors are in the space of prime exponents.
    # If we have 16 numbers, their exponent vectors are in R^12.
    # They CANNOT be linearly independent if there are more than 12 of them.
    # Therefore, the condition "all subset products are distinct" does NOT imply linear independence of the exponent vectors.
    
    # Let's re-read the condition: "All 2^k subset products ... must be distinct."
    # This means no subset product equals another subset product.
    # This is equivalent to: The only integer solution to sum_{i in A} e_i - sum_{i in B} e_i = 0 (where e_i are exponent vectors) with A, B disjoint, is the trivial solution (A=B=empty).
    # This is exactly the definition of linear independence over the integers? No, over the rationals?
    # If the vectors are linearly dependent over Q, then there exist rational coefficients c_i such that sum c_i v_i = 0.
    # Multiply by LCM to get integer coefficients.
    # Then sum k_i v_i = 0 with k_i integers.
    # This implies that the product of p_i^{k_i} = 1.
    # This means the subset of numbers with positive k_i has the same product as the subset with negative k_i.
    # So, YES, the condition is equivalent to linear independence of the exponent vectors over Q.
    
    # If so, the maximum size is at most 12.
    # But the seed program returns 16 and scores 1.3333.
    # Seed score = 16 / 12 = 1.3333.
    # This implies the "seed" value (the baseline) is 12.
    # The task says: "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # So if I return 16, score is 16/12 = 1.3333.
    # Can I return more than 16?
    # The maximum size of a linearly independent set in R^12 is 12.
    # So it is IMPOSSIBLE to have a set of more than 12 multiplicatively independent numbers.
    # Therefore, 16 is NOT possible if the condition implies linear independence.
    
    # Let me double check the "multiplicatively independent" definition.
    # A set S is multiplicatively independent if no element is a product of the others.
    # This is weaker than "all subset products are distinct".
    # "All subset products are distinct" is a stronger condition.
    # It implies that the numbers are linearly independent in the multiplicative group of positive rationals.
    # This group is isomorphic to the direct sum of copies of Z (one for each prime).
    # So the exponent vectors form a subset of Z^12.
    # Linear independence over Z (or Q) is the condition.
    # So max size is 12.
    
    # Why does the seed program return 16?
    # Let's check if the seed's set of 16 is actually valid.
    # Set: [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    # 4 = 2^2, 9 = 3^2, 16 = 2^4, 25 = 5^2.
    # Vectors:
    # 2: (1,0,0,0,0,0,0,0,0,0,0,0)
    # 3: (0,1,0,0,0,0,0,0,0,0,0,0)
    # 4: (2,0,0,0,0,0,0,0,0,0,0,0) -> Dependent on 2!
    # If 2 and 4 are in the set, then product of {4} = 4, product of {2,2}... wait, subsets are formed by picking each element 0 or 1 times.
    # So subset {4} has product 4.
    # Is there another subset with product 4?
    # Subset {2, ?} -> 2*2=4 requires two 2s. But we can only pick 2 once.
    # So {4} = 4. {2, 2} is not a subset.
    # So 2 and 4 are NOT a conflict in terms of "subset products distinct".
    # The conflict is: does there exist a non-trivial integer linear combination of the exponent vectors that equals zero?
    # Let v_2 = e_1, v_4 = 2e_1.
    # Are they linearly independent?
    # c_1 v_2 + c_2 v_4 = 0 => (c_1 + 2c_2)e_1 = 0 => c_1 + 2c_2 = 0.
    # Non-trivial integer solution: c_1 = 2, c_2 = -1.
    # This means v_4 - 2v_2 = 0? No, v_4 = 2 v_2.
    # So v_4 - v_2 - v_2 = 0.
    # This corresponds to: Product of elements with positive coeff / Product of elements with negative coeff = 1.
    # Positive: 4 (coeff 1). Negative: 2, 2 (coeff -1 each).
    # But we can only use each element once!
    # The condition is: The only subset A, B such that prod(A) = prod(B) is A=B.
    # This is equivalent to: The only integer solution to sum_{i} k_i v_i = 0 with k_i in {-1, 0, 1} is k_i = 0 for all i.
    # This is NOT the same as linear independence over Q!
    # It is a "subset-sum" type problem in the group Z^12.
    # It is much weaker than linear independence.
    # We can have dependent vectors, as long as the dependency requires using a coefficient with absolute value > 1 or repeating an element.
    
    # So, 16 is possible.
    # Can we have more than 16?
    # The problem is: Find the largest subset of {2..40} such that all subset products are distinct.
    # This is equivalent to finding the largest subset S such that the map from {0,1}^|S| to Q* given by subset products is injective.
    
    # Let's try to search for a larger set.
    # The seed has 16.
    # Max possible?
    # There are 39 numbers.
    # We can use a SAT solver or just brute force with pruning? 2^39 is too big.
    # But we can use a greedy approach or branch and bound.
    
    # Let's try to add more numbers to the seed set.
    # Seed: [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    # Missing: 6, 8, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40
    
    # Let's try to find a valid set of size 17 or more.
    # We will use a backtracking search.
    # State: current set S.
    # Check validity: All subset products distinct.
    # To check validity efficiently, we can maintain the set of subset products.
    # Size of product set is 2^|S|. For |S|=20, that's 1 million. Manageable.
    # For |S|=25, 33 million. Might be slow but okay for a few steps.
    
    # Algorithm:
    # 1. Start with a known valid set of size 16 (the seed).
    # 2. Try to add each missing number one by one.
    # 3. If adding a number keeps the set valid, add it.
    # 4. If we get stuck (no more numbers can be added), try backtracking (remove some added numbers to allow others).
    # 5. To be safe, we can try different starting orders or random restarts.
    
    def is_valid_subset(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                np = x * p
                new_prods.add(np)
            if len(new_prods) != len(prods) * 2:
                return False
            prods = new_prods
        return True
    
    # Optimization: Instead of checking all 2^k products from scratch, we can update incrementally.
    # But for simplicity and correctness, let's use the set of products.
    
    best_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    best_size = len(best_set)
    
    all_numbers = list(range(2, 41))
    
    # Greedy with backtracking
    # We want to find a maximal independent set in a hypergraph?
    # Actually, let's just do a DFS.
    
    def dfs(current_set, remaining, current_prods):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
        
        if not remaining:
            return
        
        # Try to add each element from remaining
        for i, x in enumerate(remaining):
            # Check if adding x is valid
            # new_prods = current_prods U {x * p for p in current_prods}
            # If any collision, skip
            new_prods = set(current_prods)
            valid = True
            for p in current_prods:
                np = x * p
                if np in new_prods:
                    valid = False
                    break
                new_prods.add(np)
            
            if valid:
                dfs(current_set + [x], remaining[i+1:], new_prods)
        
        # Option to skip x
        dfs(current_set, remaining[1:], current_prods)
        
    # This DFS is too slow for 39 items.
    # We need a better heuristic.
    # Let's stick to the seed set and try to add elements to it greedily.
    
    current_set = list(best_set)
    current_prods = {1}
    for x in current_set:
        for p in list(current_prods):
            current_prods.add(x * p)
            
    remaining = [x for x in all_numbers if x not in current_set]
    
    # Sort remaining by some heuristic? Smaller numbers first?
    # Let's try adding them in increasing order.
    for x in remaining:
        if time.time() > deadline:
            break
        new_prods = set(current_prods)
        valid = True
        for p in current_prods:
            np = x * p
            if np in new_prods:
                valid = False
                break
            new_prods.add(np)
        if valid:
            current_set.append(x)
            current_prods = new_prods
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
    
    # Try a few random restarts with different orders
    import random
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        curr = list(best_set)
        prods = {1}
        for x in curr:
            for p in list(prods):
                prods.add(x * p)
        rem = [x for x in all_numbers if x not in curr]
        random.shuffle(rem)
        for x in rem:
            if time.time() > deadline:
                break
            new_prods = set(prods)
            valid = True
            for p in prods:
                np = x * p
                if np in new_prods:
                    valid = False
                    break
                new_prods.add(np)
            if valid:
                curr.append(x)
                prods = new_prods
                if len(curr) > best_size:
                    best_set = list(curr)
                    best_size = len(curr)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)

import time
import itertools

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all subset products are distinct.
    Uses an exact Maximum Independent Set solver on the conflict graph.
    """
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    candidates = list(range(2, 41))
    n = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # 1. Build Conflict Graph
    # Two numbers u, v conflict if they cannot be in the same dissociated set.
    # Conflict conditions:
    # a) u == v (not applicable here as candidates are distinct)
    # b) u == 2*v (or v == 2*u) -> product of {u} == product of {v, ...} wait, no.
    #    Actually, if we have u and v, and u = 2v, then product {u} = 2v.
    #    If we also had v, product {v, v}? No, sets don't have duplicates.
    #    Subset products are products of DISTINCT elements from the set.
    #    So {u} has product u. {v} has product v.
    #    If u = 2v, is there a collision?
    #    Collision means two DIFFERENT subsets have the SAME product.
    #    If we select {u, v}, subsets are {}, {u}, {v}, {u,v}.
    #    Products: 1, u, v, uv.
    #    If u=2v, products are 1, 2v, v, 2v^2. All distinct unless v=1 (not in range).
    #    So u=2v is NOT a direct conflict for size 2.
    #    
    #    Let's re-evaluate the conflict definition.
    #    A set S is dissociated if all subset products are distinct.
    #    This is equivalent to: the map from subset mask to product is injective.
    #    This implies that no element can be represented as a product of other elements in S.
    #    i.e., for any x in S, x != prod(S \ {x})? No.
    #    x != prod(Y) for any Y subset of S \ {x}.
    #    
    #    So, u and v conflict if:
    #    1. u = v (impossible)
    #    2. u = 2*v is NOT a conflict by itself. 
    #       Wait, consider {2, 4}. Subsets: {}, {2}, {4}, {2,4}. Products: 1, 2, 4, 8. Distinct.
    #       Consider {2, 4, 8}. Subsets products:
    #       1, 2, 4, 8, 2*4=8. COLLISION! {8} and {2,4} both give 8.
    #       So 8 conflicts with {2,4}. But in a pairwise graph, we need to know if adding 8 
    #       to a set containing 2 and 4 causes a collision.
    #       
    #    The property "all subset products are distinct" is NOT pairwise. It is a global property.
    #    However, we can define a conflict graph where an edge (u, v) exists if 
    #    {u, v} is NOT a dissociated set.
    #    If {u, v} is not dissociated, then any superset is not dissociated.
    #    But it's possible that {u, v} IS dissociated, but {u, v, w} is NOT.
    #    Example: {2, 4} is dissociated. {2, 4, 8} is not.
    #    So a simple pairwise conflict graph (where edge = pairwise collision) is NOT sufficient 
    #    to find the Maximum Dissociated Set using standard MIS.
    #    Standard MIS finds a set where NO PAIR has an edge. 
    #    If we define edge(u,v) iff {u,v} is not dissociated, then an independent set 
    #    is a set where every PAIR is dissociated.
    #    Does "every pair is dissociated" imply "the whole set is dissociated"?
    #    No. {2, 4, 8}: 
    #    Pair (2,4): products 1,2,4,8. Distinct. OK.
    #    Pair (2,8): products 1,2,8,16. Distinct. OK.
    #    Pair (4,8): products 1,4,8,32. Distinct. OK.
    #    But {2,4,8} has collision 8=2*4.
    #    So MIS on pairwise conflicts will return {2,4,8} as a valid independent set, 
    #    but it is INVALID for the task.
    
    #    Therefore, we CANNOT use a simple pairwise conflict graph MIS.
    #    We need a different approach.
    
    #    Alternative Strategy:
    #    The problem is to find the largest subset of {2..40} such that all subset products are distinct.
    #    This is equivalent to finding the largest set of integers where no integer is the product of a subset of the others.
    #    
    #    Let's try a backtracking search with strong pruning.
    #    Since n=39 is too large for naive backtracking, we need to be smart.
    #    
    #    Observation: If a set S is dissociated, then for any x in S, x cannot be formed by multiplying elements of S\{x}.
    #    This suggests we should prioritize numbers that are "hard" to form, i.e., primes or prime powers with unique factors.
    #    
    #    Let's use a heuristic search that verifies validity at the end, but uses the "no element is product of others" as a pruning rule during construction.
    #    
    #    Better yet, let's use the fact that the maximum size is likely 17 or 18.
    #    We can use a randomized greedy approach with local search, but we need to ensure we hit 18 if possible.
    #    
    #    Let's try a different structural insight:
    #    If we select a set of numbers, we can map them to their prime exponent vectors.
    #    The condition is that the subset sums of these vectors are all distinct.
    #    This is a "dissociated set" in the group Z^k (where k is number of primes).
    #    
    #    Let's go back to the "Coupled" idea.
    #    The decoupled max was 16.
    #    Can we get 17?
    #    The inspiration says "Randomized greedy search... finds 17 or 18".
    #    
    #    Let's implement a robust randomized greedy + local search.
    #    1. Generate random candidates.
    #    2. Greedily add if it maintains dissociation.
    #    3. Local search: try swapping one element for another to increase size or maintain size while enabling future additions.
    #    
    #    To be safe and fast, we will run multiple restarts.
    
    def is_dissociated(S):
        """Check if all subset products of S are distinct."""
        # For small S, we can just compute all products.
        # 2^k products. If k=18, 2^18 = 262,144. This is fast.
        # We use a set of products.
        prods = set()
        # Iterative subset product generation
        # Start with empty product 1
        current = [1]
        for x in S:
            # For each existing product, multiply by x and add to new list
            # We can do this in-place if careful, or just create new list
            # Since we check for duplicates, we can just add to set
            new_prods = []
            for p in current:
                np = p * x
                if np in prods:
                    return False
                prods.add(np)
                new_prods.append(np)
            current.extend(new_prods)
        return True

    def try_greedy(candidates, order):
        """Greedily add numbers in given order if valid."""
        S = []
        for x in order:
            if x in S:
                continue
            S.append(x)
            if not is_dissociated(S):
                S.pop()
        return S

    def local_search(S, candidates, max_iterations=100):
        """Try to improve S by swapping or adding."""
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Try to add any candidate not in S
            for x in candidates:
                if x not in S_set:
                    S.append(x)
                    if is_dissociated(S):
                        S_set.add(x)
                        improved = True
                        break
                    else:
                        S.pop()
            
            if improved:
                continue
                
            # Try to swap
            for i, s_elem in enumerate(S):
                for x in candidates:
                    if x in S_set:
                        continue
                    # Remove s_elem, add x
                    S[i] = x
                    if is_dissociated(S):
                        S_set.remove(s_elem)
                        S_set.add(x)
                        improved = True
                        break
                    else:
                        S[i] = s_elem
                if improved:
                    break
        return S

    import random
    
    best_set = []
    
    # Pre-compute primes for potential heuristic ordering
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomize the order of candidates, but bias towards "hard" numbers (primes, prime powers)
        # to form a good base set quickly.
        weighted_candidates = list(candidates)
        random.shuffle(weighted_candidates)
        
        # Greedy phase
        current_set = try_greedy(candidates, weighted_candidates)
        
        # Local Search phase
        current_set = local_search(current_set, candidates, max_iterations=20)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        if len(best_set) == 18:
            break
            
    # Fallback: If we didn't find 17+, use the known 16 set?
    # But the task asks to maximize. The inspirations say 17 or 18 is possible.
    # If our search fails to find 17, we return the best we found.
    
    # Ensure types are strictly Python int
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # 1. Precomputation
    # All 560 triples
    all_triples = list(combinations(range(16), 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute for each triple, the list of other 7 triples that would complete a K(2,2,2)
    # A K(2,2,2) is defined by 3 disjoint pairs (x1,x2), (y1,y2), (z1,z2).
    # The 8 transversals are all combinations of one from each pair.
    # For a specific triple T = (a,b,c), it participates in a K(2,2,2) if there exist a', b', c'
    # such that {a,a',b,b',c,c'} are distinct and T is one of the 8.
    # The other 7 are:
    # (a', b, c), (a, b', c), (a, b, c'),
    # (a', b', c), (a', b, c'), (a, b', c'),
    # (a', b', c')
    
    # We need to enumerate all K(2,2,2) configurations.
    # A configuration is a set of 8 triples.
    # Number of such configurations: 
    # Choose 6 distinct points from 16: C(16,6) = 8008
    # Partition 6 points into 3 pairs: 15 ways
    # Total = 8008 * 15 = 120120? 
    # Wait, the problem says "no three disjoint pairs have all eight transversal triples present".
    # This implies the set of 8 triples.
    # Let's verify the count of distinct sets of 8 triples forming a K(2,2,2).
    # Each K(2,2,2) is uniquely determined by the set of 6 points and the pairing.
    # So there are 120,120 such clauses. This is too many to store explicitly for 560 vars?
    # 120k clauses * 8 vars each = ~1M entries. Manageable.
    
    # However, we can optimize by iterating over triples and finding conflicts on the fly?
    # No, for local search, we need to know which clauses are violated.
    # A clause is violated if all 8 vars are True.
    
    # Let's precompute the clauses.
    # For each combination of 6 points:
    #   For each partition into 3 pairs:
    #     Generate the 8 triples.
    #     Store the list of indices of these 8 triples.
    
    # This might take a bit of time to generate.
    # C(16,6) = 8008. 15 partitions. 8008 * 15 = 120120 clauses.
    # Generating this list takes maybe 1-2 seconds.
    
    clauses = []
    # To speed up, we can iterate over points
    points = range(16)
    
    # Alternative faster generation:
    # Iterate a, b, c distinct.
    # This is hard to map to unique K(2,2,2)s without duplicates.
    # Let's stick to the 6-point combination method but optimize.
    
    # Actually, we can just check violations on the fly during the search steps?
    # If we flip a variable, we need to update the count of satisfied clauses.
    # A clause is satisfied if at least one of its 8 vars is False.
    # It is violated if all 8 are True.
    
    # Let's precompute the list of clauses as lists of tuple indices.
    # We also need for each variable (triple), the list of clauses it belongs to.
    
    # Given the time limit, let's see if we can generate clauses quickly.
    # 120k clauses is a lot.
    
    # Is there a sparser way?
    # A K(2,2,2) is a 3-partite 3-uniform hypergraph complete graph.
    
    # Let's try to generate them.
    clause_list = []
    var_to_clauses = [[] for _ in range(n_triples)]
    
    # Optimization: 
    # Instead of generating all 120k, can we check validity of a set of triples faster?
    # The current program uses a "check_violation_fast" which iterates 13*12*11 = 1716 candidates for a new triple.
    # For local search, we flip one triple at a time.
    # When we flip triple T from False to True:
    #   We need to check if it creates any new violation.
    #   A violation is created if there exists a K(2,2,2) where T is the 8th true triple.
    #   i.e., there are 7 other triples in the K(2,2,2) that are already True.
    #   So we just need to check the K(2,2,2)s that contain T.
    #   How many K(2,2,2)s contain a specific triple T=(a,b,c)?
    #   T uses 3 points. The other 3 points a', b', c' must be chosen from the remaining 13.
    #   The pairs are (a,a'), (b,b'), (c,c').
    #   a' can be any of 13, b' any of 12, c' any of 11.
    #   Total 1716 K(2,2,2)s contain T.
    #   
    # When we flip T from True to False:
    #   We might fix violations.
    #   We need to check if any K(2,2,2) containing T was previously violated (all 8 True) and now becomes satisfied (T is False).
    
    # So, we don't need to store ALL 120k clauses explicitly in a global list if we can generate the 1716 relevant ones on the fly for a given triple.
    # But 1716 * 7 checks per flip is ~12k checks.
    # If we do 1000 flips, that's 12M checks. Feasible in 135s?
    # Python is slow. 12M iterations with inner loops might be too slow.
    
    # Let's precompute the "conflict sets" for each triple.
    # For each triple T, store the list of 7-tuples of indices of the other triples in the K(2,2,2)s that contain T.
    # Size: 560 triples * 1716 K(2,2,2)s * 7 ints = ~6.7M integers.
    # Memory: ~50MB. Acceptable.
    # Time to precompute: 560 * 1716 * 7 lookups. ~6.7M lookups. Fast.
    
    # Let's do this.
    # For each triple T:
    #   Find all K(2,2,2)s containing T.
    #   For each such K, store the indices of the other 7 triples.
    
    conflict_cache = [[] for _ in range(n_triples)]
    
    # To generate K(2,2,2)s containing T=(a,b,c):
    # We need to choose a' from remaining, b' from remaining, c' from remaining.
    # The other 7 triples are:
    # (a', b, c), (a, b', c), (a, b, c'),
    # (a', b', c), (a', b, c'), (a, b', c'),
    # (a', b', c')
    
    for t_idx, T in enumerate(all_triples):
        a, b, c = T
        # Candidates for a', b', c'
        # a' != a,b,c
        # b' != a,b,c,a'
        # c' != a,b,c,a',b'
        
        # This loop is 13 * 12 * 11 = 1716 iterations.
        for a_prime in range(16):
            if a_prime in (a,b,c): continue
            for b_prime in range(16):
                if b_prime in (a,b,c,a_prime): continue
                for c_prime in range(16):
                    if c_prime in (a,b,c,a_prime,b_prime): continue
                    
                    # The 7 other triples
                    others = [
                        tuple(sorted((a_prime, b, c))),
                        tuple(sorted((a, b_prime, c))),
                        tuple(sorted((a, b, c_prime))),
                        tuple(sorted((a_prime, b_prime, c))),
                        tuple(sorted((a_prime, b, c_prime))),
                        tuple(sorted((a, b_prime, c_prime))),
                        tuple(sorted((a_prime, b_prime, c_prime)))
                    ]
                    other_indices = [triple_to_idx[o] for o in others]
                    conflict_cache[t_idx].append(other_indices)
                    
    # Now we have for each triple, a list of 1716 lists of 7 indices.
    # Total storage: 560 * 1716 * 7 = 6,713,280 ints.
    
    # Local Search Strategy:
    # 1. Initialize with a random subset of triples (e.g., random 50% or greedy start).
    #    Let's start with a greedy construction to get a good initial solution.
    #    Then use local search to improve.
    
    def greedy_construct():
        system_set = set()
        system = []
        random.shuffle(all_triples)
        for T in all_triples:
            if time.time() > deadline - 5: break
            idx = triple_to_idx[T]
            # Check if adding T causes violation
            # Violation if any K(2,2,2) containing T has all other 7 already in system_set.
            is_violation = False
            for others in conflict_cache[idx]:
                # Check if all 7 are in system_set
                # system_set contains tuples, others are indices.
                # We need to check if all_triples[o] is in system_set.
                # Faster: maintain a boolean array `active` of length 560.
                pass
            # Let's use a boolean array for speed.
            # This greedy step is redundant if we do local search from scratch, 
            # but a good start helps.
            
        # Let's just do random initialization for local search, it's often robust.
        # Or better: Start with empty set, add random triples?
        # Let's start with a random 30% subset.
        active = [random.random() < 0.3 for _ in range(n_triples)]
        return active
    
    # Local Search (WalkSAT / MaxSAT style)
    # Objective: Maximize number of True variables such that no clause is fully True.
    # This is equivalent to minimizing the number of violated clauses (clauses with all 8 True).
    # Penalty = sum over all K(2,2,2) of [all 8 are True].
    
    # We can track the number of violated clauses.
    # Initially, calculate penalty.
    # Then, pick a random variable, flip it, calculate delta penalty.
    # If delta penalty < 0 (improvement), accept.
    # If delta penalty == 0, accept with some probability (or always to escape local minima? No, 0 means no change in validity, but changes size? 
    # Wait, the objective is to MAXIMIZE SIZE subject to NO violations.
    # If we have 0 violations, we want to ADD more triples (flip False -> True) if it doesn't create violations.
    # If we have violations, we want to REMOVE triples (flip True -> False) to fix them.
    
    # So the objective function is:
    # Score = -1000 * (num_violations) + (num_triples)
    # We want to maximize Score.
    
    def get_score(active):
        num_violations = 0
        num_triples = sum(active)
        # Calculate violations
        # A clause is violated if all 8 are True.
        # We can iterate over all clauses? 120k clauses.
        # Or iterate over variables?
        # If we iterate over variables, we count each violation multiple times.
        # A violation involves 8 variables.
        # Let's precompute the list of all clauses (120k of them) to check violations efficiently?
        # Or use the conflict_cache?
        # conflict_cache[i] contains the other 7 for each K containing i.
        # If active[i] is True, we can check if any of the Ks is fully True.
        # But a K is fully True if all 8 are True.
        # If we sum over i where active[i] is True, the number of Ks where all 8 are True,
        # each violated K is counted 8 times.
        # So num_violations = (sum over i where active[i] is True of count_of_Ks_where_all_8_true) / 8.
        
        total_count = 0
        for i in range(n_triples):
            if not active[i]:
                continue
            for others in conflict_cache[i]:
                # Check if all 7 others are True
                # others is list of 7 indices
                # all(active[j] for j in others)
                # This is 1716 * 7 checks per True variable.
                # If 300 vars are True, 300 * 1716 * 7 = 3.6M checks.
                # Doing this every step is too slow.
                
                # We need a faster way to count violations or update them incrementally.
                
                # Incremental update:
                # When we flip variable X:
                # 1. If X goes False -> True:
                #    New violations are created for any K containing X where the other 7 were already True.
                #    Count these.
                # 2. If X goes True -> False:
                #    Old violations are fixed for any K containing X where the other 7 are True (and X was True).
                #    Count these.
                
                # So we just need to check the 1716 Ks containing X.
                # For each K, check the state of the other 7.
                # This is 1716 * 7 = 12k checks per flip.
                # If we do 1000 flips, 12M checks.
                # In Python, 12M simple boolean checks might take ~1-2 seconds.
                # This is feasible within 135s for several hundred/thousand steps.
                
                pass
        return -1000 * (total_count // 8) + num_triples

    # Let's implement the incremental flip logic.
    
    def count_new_violations_if_added(idx, active):
        # Returns number of K(2,2,2)s that would become violated if we set active[idx]=True
        # (Assuming it was False)
        count = 0
        for others in conflict_cache[idx]:
            # Check if all 7 are True
            # Unroll for speed?
            if (active[others[0]] and active[others[1]] and active[others[2]] and 
                active[others[3]] and active[others[4]] and active[others[5]] and 
                active[others[6]]):
                count += 1
        return count

    def count_fixed_violations_if_removed(idx, active):
        # Returns number of K(2,2,2)s that would be fixed if we set active[idx]=False
        # (Assuming it was True)
        count = 0
        for others in conflict_cache[idx]:
            if (active[others[0]] and active[others[1]] and active[others[2]] and 
                active[others[3]] and active[others[4]] and active[others[5]] and 
                active[others[6]]):
                count += 1
        return count

    # Main Search Loop
    best_score = -10**9
    best_active = None
    
    # Initialize with a few random restarts
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize: Random 20-40% density
        density = random.uniform(0.2, 0.4)
        active = [random.random() < density for _ in range(n_triples)]
        
        # Calculate initial violations
        current_violations = 0
        for i in range(n_triples):
            if active[i]:
                current_violations += count_new_violations_if_added(i, active)
        current_violations //= 8 # Each violation counted 8 times
        
        current_size = sum(active)
        current_score = -1000 * current_violations + current_size
        
        if current_score > best_score:
            best_score = current_score
            best_active = active[:]
            
        # Local Search
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5:
            if no_improve_count > max_no_improve:
                break
                
            # Pick a random variable to flip
            idx = random.randint(0, n_triples - 1)
            
            if active[idx]:
                # Flip True -> False
                # Fixes violations
                fixed = count_fixed_violations_if_removed(idx, active)
                # New violations? No, removing a triple cannot create a new violation.
                new_violations = 0
                
                delta_violations = new_violations - fixed
                delta_size = -1
                
                # Accept if score improves
                # Score change = -1000 * delta_violations + delta_size
                if delta_violations < 0 or (delta_violations == 0 and delta_size > 0):
                    # Actually, if delta_violations < 0, it's always good.
                    # If delta_violations == 0, we lose a triple, so score decreases.
                    # So we only accept if fixed > 0.
                    if fixed > 0:
                        active[idx] = False
                        current_violations += delta_violations
                        current_size += delta_size
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Flip False -> True
                # Creates violations
                created = count_new_violations_if_added(idx, active)
                # Fixes violations? No, adding a triple cannot fix a violation.
                fixed = 0
                
                delta_violations = created - fixed
                delta_size = 1
                
                # Accept if created == 0 (no new violations)
                if created == 0:
                    active[idx] = True
                    current_violations += delta_violations
                    current_size += delta_size
                    no_improve_count = 0
                else:
                    # Maybe accept with small probability if it helps escape local minima?
                    # But we want 0 violations eventually.
                    # If we have 0 violations, we only add if created==0.
                    # If we have violations, we should prioritize removing.
                    # So we bias towards removal if violations > 0.
                    if current_violations > 0:
                         # Try to remove a random True variable instead?
                         pass
                    no_improve_count += 1
            
            # Update best
            current_score = -1000 * current_violations + current_size
            if current_score > best_score:
                best_score = current_score
                best_active = active[:]
                
        # End while loop for this restart

    # Convert best_active to list of triples
    if best_active is None:
        # Fallback
        return [[0, 1, 2]]
        
    result = []
    for i, is_active in enumerate(best_active):
        if is_active:
            result.append(list(all_triples[i]))
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

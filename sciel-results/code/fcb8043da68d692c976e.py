import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Precompute for each 6-point subset: the 24 transversal triples
    # A 6-set {a,b,c,d,e,f} has 3 pairings; each pairing gives 8 transversals
    # Total 24 transversal triples per 6-set
    # But we need to track: for each 6-set, how many of its 24 transversals are present
    # If any 6-set has all 24 present -> violation? No...
    
    # Actually, the condition is: no three disjoint pairs have all eight transversal triples present.
    # Three disjoint pairs on 6 points: the 6 points are partitioned into 3 pairs.
    # There are 15 ways to partition 6 points into 3 pairs.
    # For each such partition (p1, p2, p3), the 8 transversals are all triples picking one from each pair.
    # Violation if all 8 transversals of some partition are present.
    
    # Precompute: for each 6-set, list of (partition_id, [8 transversal triples])
    # 15 partitions per 6-set, 8 transversals each = 120 triple-occurrences per 6-set
    # But many triples repeat across partitions.
    
    # Simpler: for each 6-set, precompute the list of (partition, frozenset_of_8_triples)
    # Then for a given hypergraph, check if any partition has all 8 in the hypergraph.
    
    # For efficient checking during construction:
    # For each 6-set and each of its 15 partitions, track count of present transversals.
    # Adding a triple t: for each 6-set containing t's 3 points, for each partition of that 6-set
    # where t is a transversal, increment count. If any count reaches 8, violation.
    
    # Precompute: for each triple t, list of (six_set, partition_idx) where t is a transversal
    # 6-sets containing a given triple: C(13,3) = 286
    # For each such 6-set, t is a transversal for some partitions.
    # In a 6-set with t = (a,b,c), t is a transversal iff a,b,c are in different pairs.
    # Number of partitions of 6 points where a,b,c are in different pairs: 6 (2^3 = 8? no)
    # Actually: partition 6 into 3 pairs with a,b,c in separate pairs.
    # Fix: a pairs with one of {d,e,f} (3 choices), b pairs with one of remaining 2 (2 choices), c with last (1).
    # But pairs are unordered and pairings are unordered: 3*2*1 / ... hmm.
    # Let's just enumerate all 15 partitions of a 6-set and check.
    
    # Precompute all 15 perfect matchings of 6 elements (labeled 0-5)
    def get_matchings():
        matchings = []
        def helper(remaining, current):
            if not remaining:
                matchings.append(tuple(sorted(current)))
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                current.append((first, second))
                helper(new_remaining, current)
                current.pop()
        helper(list(range(6)), [])
        return matchings
    
    matchings_6 = get_matchings()  # 15 matchings of 6 elements
    
    # For a 6-set S = {s0,...,s5}, matching m = ((i0,j0),(i1,j1),(i2,j2))
    # Transversals: all (s_{i0}, s_{i1}, s_{i2}) with one from each pair, i.e., 2^3=8 combos
    # Specifically: for each matching, the 8 transversals are:
    # for bit in 0..7: pick s_{a_k} or s_{b_k} for each pair k
    
    # Precompute for each 6-set (as frozenset), the list of 15 partitions, each with 8 transversal triples
    six_sets = list(itertools.combinations(range(n), 6))
    six_set_index = {frozenset(s): i for i, s in enumerate(six_sets)}
    
    # For each 6-set, for each of 15 matchings, compute the 8 transversal triples
    # six_set_partitions[six_set_idx] = list of 15 lists of 8 triples
    six_set_partitions = []
    for s in six_sets:
        parts = []
        for m in matchings_6:
            transversals = []
            for bits in range(8):
                tri = []
                for k in range(3):
                    if bits & (1 << k):
                        tri.append(s[m[k][1]])
                    else:
                        tri.append(s[m[k][0]])
                transversals.append(tuple(sorted(tri)))
            parts.append(transversals)
        six_set_partitions.append(parts)
    
    # For each triple t, precompute list of (six_set_idx, partition_idx) where t is in the transversals
    triple_to_occurrences = [[] for _ in range(num_triples)]
    for si, s in enumerate(six_sets):
        s_set = set(s)
        for pi in range(15):
            transversals = six_set_partitions[si][pi]
            for t in transversals:
                ti = triple_index[t]
                triple_to_occurrences[ti].append((si, pi))
    
    # Now greedy + local search
    # State: set of included triples, and for each (six_set, partition), count of present transversals
    # counts[si][pi] = number of the 8 transversals that are in our hypergraph
    
    def is_valid_incremental(counts, si, pi):
        return counts[si][pi] < 8
    
    # Greedy: add triples in random order, skip if would violate
    def greedy_random(seed_val):
        rng = random.Random(seed_val)
        order = list(range(num_triples))
        rng.shuffle(order)
        
        counts = [[0]*15 for _ in range(len(six_sets))]
        included = set()
        
        for ti in order:
            t = all_triples[ti]
            # Check all occurrences
            can_add = True
            occs = triple_to_occurrences[ti]
            for (si, pi) in occs:
                if counts[si][pi] >= 8:
                    can_add = False
                    break
            if can_add:
                included.add(ti)
                for (si, pi) in occs:
                    counts[si][pi] += 1
        
        return included, counts
    
    # Local search: try to swap out a triple that's blocking, add a new one
    def local_search(included, counts, rng, time_limit):
        # Find triples that are "blocking" - i.e., removing them would allow some new triple
        # Simpler: try random swaps - remove a random included triple, add a random excluded triple if valid
        best_included = set(included)
        best_count = len(included)
        
        excluded = set(range(num_triples)) - included
        
        while time.time() < time_limit:
            if not excluded:
                break
            
            # Try a batch of random swaps
            attempts = 0
            improved = False
            while attempts < 500 and not improved:
                attempts += 1
                if not excluded:
                    break
                out_ti = rng.choice(list(included)) if included else None
                if out_ti is None:
                    break
                in_ti = rng.choice(list(excluded))
                
                # Remove out_ti
                for (si, pi) in triple_to_occurrences[out_ti]:
                    counts[si][pi] -= 1
                
                # Try to add in_ti
                can_add = True
                for (si, pi) in triple_to_occurrences[in_ti]:
                    if counts[si][pi] >= 8:
                        can_add = False
                        break
                
                if can_add:
                    # Accept the swap
                    included.discard(out_ti)
                    included.add(in_ti)
                    excluded.discard(in_ti)
                    excluded.add(out_ti)
                    for (si, pi) in triple_to_occurrences[in_ti]:
                        counts[si][pi] += 1
                    improved = True
                    if len(included) > best_count:
                        best_count = len(included)
                        best_included = set(included)
                else:
                    # Revert
                    for (si, pi) in triple_to_occurrences[out_ti]:
                        counts[si][pi] += 1
            
            if not improved:
                # Try hill-climbing: find any single add (no removal)
                # Or try to find a triple to remove that unblocks the most
                break
        
        return best_included, counts
    
    # Run multiple greedy restarts
    best_result = set()
    best_len = 0
    
    num_restarts = 0
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        num_restarts += 1
        included, counts = greedy_random(seed)
        if len(included) > best_len:
            best_len = len(included)
            best_result = set(included)
        
        # Local search on the best found so far
        if time.time() < deadline - 20:
            rng = random.Random(seed)
            improved_included, improved_counts = local_search(
                set(included), 
                [[c for c in row] for row in counts],  # copy
                rng,
                deadline - 20
            )
            if len(improved_included) > best_len:
                best_len = len(improved_included)
                best_result = set(improved_included)
    
    # Final local search on the best result if time permits
    if time.time() < deadline - 10 and best_result:
        # Rebuild counts for best_result
        counts = [[0]*15 for _ in range(len(six_sets))]
        for ti in best_result:
            for (si, pi) in triple_to_occurrences[ti]:
                counts[si][pi] += 1
        
        rng = random.Random(42)
        final_included, final_counts = local_search(
            set(best_result),
            counts,
            rng,
            deadline - 5
        )
        if len(final_included) > best_len:
            best_len = len(final_included)
            best_result = set(final_included)
    
    # Convert to list of lists
    result = [list(all_triples[ti]) for ti in sorted(best_result)]
    return result

# Verify at start for safety
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} triples")

import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    # 1. Generate candidates: integers 2 to 128
    candidates = list(range(2, 129))
    n = len(candidates)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    # 2. Build Conflict Graph
    # Two numbers x, y conflict if they cannot both be in the sequence.
    # Key insight: If we select a set S, all consecutive products must be distinct.
    # A necessary condition for a set to be valid is that no two distinct consecutive blocks
    # have the same product. 
    # 
    # However, checking all pairs is hard. A strong heuristic for "dense" sequences is that 
    # if x * y == z * w for distinct pairs, it creates risks. 
    # More directly: In a strictly increasing sequence, if we have ...x, y... 
    # and ...z, w..., collisions are most likely to happen with short blocks.
    #
    # Let's use a stronger pairwise conflict definition based on the "product graph".
    # Two numbers x and y are "incompatible" if there exist other numbers u, v in the range
    # such that x*y = u*v and the ordering constraints make it likely they appear in 
    # a way that causes collision, OR if x*y is a perfect square of a number in the set 
    # (implying potential symmetry).
    #
    # Actually, the most robust pairwise conflict for this specific problem (unique block products)
    # is:
    # 1. x == y (trivial)
    # 2. x * y == z * w where {x,y} != {z,w}. If we include x, y, z, w, we risk collision 
    #    if they are arranged as x,y,z,w (prod xy) and z,w (prod zw) - no, that's not right.
    #    The collision is between block [x,y] and block [z,w] if xy=zw.
    #    So if xy=zw, we cannot have both pairs {x,y} and {z,w} as *consecutive* elements 
    #    in the sequence, unless the sequence is such that they don't form those specific pairs.
    #    But since we must sort the sequence, if we pick {x,y,z,w}, they will be sorted.
    #    If x<y<z<w, then xy and zw are products of disjoint pairs. They are distinct blocks.
    #    If xy == zw, then the set {x,y,z,w} is INVALID because block [x,y] has product P 
    #    and block [z,w] has product P.
    #
    # Therefore, a pairwise conflict exists between x and y if there exist z, w in the candidate 
    # set (z != x, w != y, etc.) such that x*y == z*w. 
    # Wait, if x*y = z*w, does x conflict with y? No, x and y can be together IF z and w are NOT 
    # both in the set. But x conflicts with z? No.
    # The conflict is between the *pair* {x,y} and the *pair* {z,w}.
    # This is a hypergraph problem, not a simple graph.
    #
    # However, we can approximate this with a graph where we mark edges that are "dangerous".
    # If x*y = z*w, we can't include all four. 
    # To make it a graph MIS: 
    # If x*y = z*w, we can add an edge between x and z? No.
    # We can add an edge between x and w? No.
    # 
    # Let's look at the structure: x*y = z*w => x/z = w/y.
    # This implies a multiplicative dependency.
    # 
    # Alternative approach: 
    # The problem is equivalent to finding a subset S such that the set of products 
    # {prod(S[i:j])} has size n(n+1)/2.
    # 
    # Let's use a different structural insight: 
    # If we choose numbers that are "multiplicatively independent" in a strong sense, 
    # we avoid collisions. 
    # Specifically, if we choose numbers such that no product of a subset equals the product 
    # of another disjoint subset, we are safe? No, only consecutive blocks.
    #
    # Let's stick to the pairwise collision of length-2 blocks first, as it's the most common 
    # source of failure in dense sequences.
    # If x*y = z*w with x<y and z<w and {x,y} != {z,w}, then we cannot have both 
    # (x,y) as adjacent and (z,w) as adjacent.
    # Since the sequence is sorted, if x<y<z<w, then x,y are adjacent only if no element 
    # between x and y is selected. Similarly for z,w.
    # This makes a static conflict graph tricky.
    
    # Let's try a different structural model: 
    # **Exponent Vector Uniqueness**
    # Represent each number by its prime factorization vector.
    # The product of a block is the sum of the vectors.
    # We want all consecutive sums to be distinct.
    # This is a "distinct partial sums" problem on a subset of vectors, but restricted to 
    # contiguous subarrays of the sorted list.
    
    # Given the difficulty of a perfect static graph, let's use a **Greedy Set Cover / 
    # Conflict Resolution** hybrid.
    # 1. Start with all numbers.
    # 2. Identify "bad" configurations: sets of numbers that create duplicate products.
    # 3. Remove the "most central" number from each bad configuration to break it.
    # 4. This is similar to Vertex Cover on a hypergraph, approximated by MIS on a graph.
    
    # Let's build a conflict graph where an edge (u,v) exists if u and v are "close" 
    # in value and their product is likely to collide with another pair's product.
    # Specifically, for each product P, if there are multiple pairs (a,b) with a<b and a*b=P,
    # then we cannot select all pairs. We must select at most one pair from each "product class".
    # This is a matching problem! 
    # If we view the sequence as a path, the edges of the path are the consecutive pairs.
    # The product of each edge (pair) must be distinct.
    # So, we are looking for a path in a graph where vertices are numbers 2..128, 
    # edges are pairs (u,v) with u<v, and we require that the edge weights (products) 
    # are all distinct.
    # Wait, the sequence is a subset of numbers, sorted. So the "path" is fixed by the subset.
    # The condition is: for the chosen subset S = {s1 < s2 < ... < sk}, 
    # the products si*si+1 must be distinct? 
    # NO. The task says "Products of every nonempty consecutive block".
    # Block of length 1: si
    # Block of length 2: si*si+1
    # Block of length 3: si*si+1*si+2
    # etc.
    
    # So, if we choose S, we need:
    # 1. All si distinct (guaranteed by set)
    # 2. All si*si+1 distinct
    # 3. All si*si+1*si+2 distinct
    # ...
    
    # The most restrictive constraint is usually on the shorter blocks.
    # Let's focus on making the length-2 products distinct.
    # If we ensure that all adjacent products in the sorted sequence are distinct, 
    # we solve a large part of the problem.
    
    # Strategy: 
    # 1. Construct a graph G where nodes are 2..128.
    # 2. We want to find a large independent set in a "conflict graph" where conflicts 
    #    arise from product collisions.
    # 3. Define conflict between x and y: 
    #    If x and y are both in the set, and there exist z, w in the set such that 
    #    x*y = z*w, we have a conflict IF the pairs are disjoint.
    #    This is still hypergraph.
    
    # Let's use a **Simulated Annealing** approach on the subset, with a cost function 
    # that penalizes product collisions. This is a different optimization method (stochastic 
    # global search) compared to the deterministic greedy/backtracking.
    
    import random
    
    def get_products(seq):
        """Returns a list of all consecutive block products."""
        prods = []
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        return prods

    def count_collisions(seq):
        """Returns the number of duplicate products."""
        prods = get_products(seq)
        return len(prods) - len(set(prods))

    # Initial state: Random subset or Greedy subset
    # Let's start with the Greedy result from the seed program as a good initial state.
    
    def greedy_initial():
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

    best_seq = greedy_initial()
    best_len = len(best_seq)
    best_collisions = count_collisions(best_seq)
    
    # Simulated Annealing
    current_seq = best_seq[:]
    current_collisions = best_collisions
    
    # We will perform moves: 
    # 1. Swap an element in seq with an element not in seq.
    # 2. Insert an element not in seq.
    # 3. Remove an element from seq.
    
    # To be efficient, we need to update collisions incrementally or use a fast check.
    # Given the small size (max 127 elements), full check is O(N^2) products, 
    # and counting collisions is O(N^2 log N^2). 
    # 127^2 = 16k products. This is fast enough for ~1000 iterations.
    
    T = 100.0
    cooling_rate = 0.995
    iterations = 5000
    
    for i in range(iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Pick a random move
        move_type = random.choice(['swap', 'insert', 'remove'])
        
        new_seq = current_seq[:]
        
        if move_type == 'swap':
            if len(new_seq) < 2: continue
            # Pick one from seq, one from not-in-seq
            in_idx = random.randint(0, len(new_seq)-1)
            in_val = new_seq[in_idx]
            
            # Find a candidate to swap with
            # Heuristic: swap with a number that has similar magnitude to maintain density
            candidates_out = [x for x in range(2, 129) if x not in new_seq]
            if not candidates_out: continue
            
            # Choose a candidate close to in_val to keep the sequence "dense"
            # Sort candidates by distance to in_val
            # For speed, just pick random from a small window around in_val
            window = 10
            near_candidates = [x for x in candidates_out if abs(x - in_val) <= window]
            if near_candidates:
                out_val = random.choice(near_candidates)
            else:
                out_val = random.choice(candidates_out)
            
            new_seq[in_idx] = out_val
            new_seq.sort()
            
        elif move_type == 'insert':
            candidates_out = [x for x in range(2, 129) if x not in new_seq]
            if not candidates_out: continue
            # Insert a random candidate
            val = random.choice(candidates_out)
            # Insert in sorted position
            # Use bisect for efficiency
            import bisect
            pos = bisect.bisect_left(new_seq, val)
            new_seq.insert(pos, val)
            
        elif move_type == 'remove':
            if len(new_seq) <= 1: continue
            idx = random.randint(0, len(new_seq)-1)
            new_seq.pop(idx)
        
        # Evaluate new state
        # Optimization: If the number of elements changed, we can estimate collision change
        # but let's just compute exactly for correctness.
        new_collisions = count_collisions(new_seq)
        
        # Acceptance criterion
        delta = new_collisions - current_collisions
        
        if delta < 0:
            # Better (fewer collisions)
            current_seq = new_seq
            current_collisions = new_collisions
            
            # If we have 0 collisions, we have a valid sequence
            if current_collisions == 0:
                if len(current_seq) > best_len:
                    best_seq = current_seq[:]
                    best_len = len(current_seq)
                    # Try to extend further? 
                    # If we have 0 collisions, we can try to insert more.
                    # Let's do a local greedy fill if we reach 0.
                    # But first, let's continue SA to find even larger valid sets?
                    # No, if collisions is 0, it's valid. We want max length.
                    # So we should try to insert more.
                    
        else:
            # Worse or same
            # Accept with probability exp(-delta/T)
            if random.random() < pow(2.71828, -delta / T):
                current_seq = new_seq
                current_collisions = new_collisions
        
        # Cooling
        T *= cooling_rate
        
        # If we have a valid sequence, try to improve length via greedy insertion
        if current_collisions == 0:
            # Try to insert any remaining number that keeps it valid
            improved = True
            while improved:
                improved = False
                for x in range(2, 129):
                    if x not in current_seq:
                        import bisect
                        pos = bisect.bisect_left(current_seq, x)
                        test_seq = current_seq[:]
                        test_seq.insert(pos, x)
                        if count_collisions(test_seq) == 0:
                            current_seq = test_seq
                            improved = True
                            break
                if improved:
                    if len(current_seq) > best_len:
                        best_seq = current_seq[:]
                        best_len = len(current_seq)

    # Final check: ensure best_seq is valid
    if count_collisions(best_seq) != 0:
        # Fallback to greedy if SA didn't find a valid one (shouldn't happen if we started with one)
        best_seq = greedy_initial()

    return best_seq

if __name__ == "__main__":
    # Quick test
    seq = construct()
    print(f"Length: {len(seq)}")
    print(f"Sequence: {seq[:10]}...")

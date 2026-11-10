import math
import time
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 80.0  # Safety margin before 90s limit

    # 255 = 3 * 5 * 17
    # Numbers coprime to 255 can never be in the solution
    # Build adjacency: two numbers are adjacent if gcd > 1
    # We need a clique containing 255

    # Precompute gcd for all pairs involving numbers that share a factor with 255
    # Numbers that share at least one factor with 255
    relevant = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    
    # Build adjacency bitsets using Python integers as bitsets
    # Map each relevant number to an index
    idx_map = {x: i for i, x in enumerate(relevant)}
    n_rel = len(relevant)
    
    # Build adjacency as bitsets (Python int as bit vector)
    adj = [0] * n_rel
    for i in range(n_rel):
        ri = relevant[i]
        bits = 0
        for j in range(n_rel):
            if i == j:
                continue
            rj = relevant[j]
            if math.gcd(ri, rj) > 1:
                bits |= (1 << j)
        adj[i] = bits
    
    # Find the index of 255
    idx_255 = idx_map[255]
    
    # Bron-Kerbosch with pivot for maximum clique
    # We want the maximum clique that contains 255
    best_clique = [255]
    best_size = 1
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        if not P:
            return
        
        # Pivot selection: choose vertex in P ∪ X with maximum degree in P
        # to minimize recursion
        pivot_candidates = P | X
        pivot = -1
        max_count = -1
        # Pick pivot with max |P ∩ adj[pivot]|
        # Iterate over bits in pivot_candidates
        pc = pivot_candidates
        while pc:
            # Get lowest set bit
            lb = pc & (-pc)
            v = lb.bit_length() - 1
            pc ^= lb
            count = (P & adj[v]).bit_count()
            if count > max_count:
                max_count = count
                pivot = v
            if time.time() > deadline:
                return
        
        if pivot == -1:
            # No pivot, just pick any
            lb = pivot_candidates & (-pivot_candidates)
            pivot = lb.bit_length() - 1
        
        # Recurse on vertices in P not adjacent to pivot
        candidates = P & ~adj[pivot]
        
        while candidates:
            lb = candidates & (-candidates)
            v = lb.bit_length() - 1
            candidates ^= lb
            
            if time.time() > deadline:
                return
            
            bron_kerbosch(R | (1 << v), P & adj[v], X & adj[v])
            P &= ~(1 << v)
            X |= (1 << v)
    
    # Strategy: We need clique containing 255.
    # Start with R = {255}, P = neighbors of 255, X = empty
    # But this might be too large. Let's try a different approach.
    
    # Approach 1: Start with prime-based seeds and expand
    def get_prime_based_set(p):
        """Get all numbers in [1,N] divisible by p (and sharing factor with 255)"""
        return [x for x in range(1, N + 1) if x % p == 0]
    
    # Try different starting points
    # For each prime factor of 255, the set of multiples forms a clique
    # But we can do better by combining
    
    # Method: Try to build a large clique by starting from various seeds
    # and using greedy + local search
    
    def greedy_clique(start_set, adj_list, deadline):
        """Greedy expansion of a clique"""
        clique = set(start_set)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for x in relevant:
                if x in clique:
                    continue
                # Check if x is compatible with all in clique
                compatible = True
                for y in clique:
                    if math.gcd(x, y) == 1:
                        compatible = False
                        break
                if compatible:
                    clique.add(x)
                    improved = True
        return list(clique)
    
    # Try multiple seed strategies
    candidate_seeds = []
    
    # Seed 1: All multiples of 3
    seed1 = [x for x in range(1, N+1) if x % 3 == 0]
    candidate_seeds.append(seed1)
    
    # Seed 2: All multiples of 5
    seed2 = [x for x in range(1, N+1) if x % 5 == 0]
    candidate_seeds.append(seed2)
    
    # Seed 3: All multiples of 17
    seed3 = [x for x in range(1, N+1) if x % 17 == 0]
    candidate_seeds.append(seed3)
    
    # Seed 4: Multiples of 3 or 5 (but need pairwise gcd > 1, so this isn't a clique)
    # Actually, multiples of 3 or 5 is NOT a clique (e.g., 3 and 5 have gcd 1)
    
    # Seed 5: Numbers divisible by at least 2 of {3,5,17}
    seed5 = [x for x in range(1, N+1) if sum(1 for p in [3,5,17] if x % p == 0) >= 2]
    candidate_seeds.append(seed5)
    
    # For each seed, verify it's a valid clique and expand
    for seed in candidate_seeds:
        if time.time() > deadline:
            break
        
        # Verify seed is a clique
        is_clique = True
        seed_set = set(seed)
        if 255 not in seed_set:
            seed_set.add(255)
        
        # Check if seed is a valid clique
        for a, b in combinations(seed, 2):
            if math.gcd(a, b) == 1:
                is_clique = False
                break
        
        if is_clique:
            # Expand greedily
            expanded = greedy_clique(seed_set, adj, deadline)
            if len(expanded) > best_size:
                best_size = len(expanded)
                best_clique = expanded
    
    # Now try Bron-Kerbosch on the subgraph of neighbors of 255
    # P = all neighbors of 255 (excluding 255 itself)
    P = adj[idx_255]  # bitset of neighbors of 255
    R = 1 << idx_255
    X = 0
    
    # But P might be too large. Let's limit the search.
    # Actually, let's try a bounded Bron-Kerbosch with a time limit
    
    # First, let's try a simpler approach: use the bitset Bron-Kerbosch
    # but with a depth limit or by restricting the candidate set
    
    # Alternative: Since we already have a good solution, try local search to improve
    # by swapping elements
    
    def local_search(clique, deadline):
        """Try to improve clique by adding/removing/swapping elements"""
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try adding elements
            for x in relevant:
                if x in clique_set:
                    continue
                compatible = True
                for y in clique_set:
                    if math.gcd(x, y) == 1:
                        compatible = False
                        break
                if compatible:
                    clique_set.add(x)
                    improved = True
            
            # Try swapping: remove one element, add another
            if time.time() > deadline:
                break
            for remove_elem in list(clique_set):
                if remove_elem == 255:
                    continue  # Must keep 255
                if time.time() > deadline:
                    break
                temp_set = clique_set - {remove_elem}
                for x in relevant:
                    if x in temp_set:
                        continue
                    compatible = True
                    for y in temp_set:
                        if math.gcd(x, y) == 1:
                            compatible = False
                            break
                    if compatible and x != remove_elem:
                        # Check if this swap improves or maintains size
                        # Only swap if it allows further additions
                        clique_set = temp_set | {x}
                        improved = True
                        break
                if improved:
                    break
        
        return list(clique_set)
    
    # Apply local search to best solution
    if len(best_clique) > 1:
        improved_clique = local_search(best_clique, deadline)
        if len(improved_clique) > len(best_clique):
            best_clique = improved_clique
            best_size = len(best_clique)
    
    # Try Bron-Kerbosch with restricted P for better results
    # Use the current best solution to guide the search
    if time.time() < deadline:
        # Try to find a larger clique using BK with the constraint that 255 must be included
        # Restrict P to neighbors of 255 that are "promising"
        
        # Get neighbors of 255
        neighbors_255 = []
        for i in range(n_rel):
            if i != idx_255 and (adj[idx_255] & (1 << i)):
                neighbors_255.append(relevant[i])
        
        # Try BK on a subset: start with P = neighbors of 255
        # But this could be expensive. Let's use a time-bounded approach.
        
        # Actually, let's try a different BK formulation:
        # We want max clique containing vertex 255.
        # This is equivalent to max clique in the neighborhood of 255, plus 255.
        
        # Let's try BK with P = neighbors of 255, and track best clique
        # But we need to be careful about time.
        
        # Use a simpler approach: iterated greedy
        # Try many random orderings of the neighbor set and greedily build cliques
        
        import random
        random.seed(42)
        
        for trial in range(100):
            if time.time() > deadline:
                break
            
            # Shuffle neighbors and greedily add
            shuffled = neighbors_255[:]
            random.shuffle(shuffled)
            
            clique = [255]
            for x in shuffled:
                if time.time() > deadline:
                    break
                compatible = True
                for y in clique:
                    if math.gcd(x, y) == 1:
                        compatible = False
                        break
                if compatible:
                    clique.append(x)
            
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
    
    # Final local search pass
    if time.time() < deadline:
        final_clique = local_search(best_clique, deadline)
        if len(final_clique) > len(best_clique):
            best_clique = final_clique
            best_size = len(final_clique)
    
    # Ensure 255 is in the result
    if 255 not in best_clique:
        best_clique.append(255)
    
    return best_clique


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify all pairs have gcd > 1
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                print(f"Invalid pair: {result[i]}, {result[j]}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")

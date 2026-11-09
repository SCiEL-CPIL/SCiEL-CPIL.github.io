import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    N = 128
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    # 1. Precompute Prime Factorization for each number
    # We need to handle products of blocks.
    # A collision occurs if product(block1) == product(block2).
    # Since the sequence is strictly increasing, let the sequence be s_1 < s_2 < ... < s_k.
    # Block products are s_i * ... * s_j.
    # If we select a subset S of {2..128}, and sort it to get the sequence,
    # we need to ensure all distinct subarrays have distinct products.
    
    # Key Insight for Conflict Graph:
    # The problem statement asks for a sequence. The "conflict" is not just pairwise.
    # However, a necessary condition for two numbers u, v to be in the sequence is that they don't form a collision with any other subset of the sequence.
    # Actually, the prompt's directive suggests modeling it as MIS on a conflict graph where an edge exists if "including both... causes any consecutive block product collision".
    # This interpretation is tricky because "causes a collision" depends on the *other* elements in the sequence.
    # 
    # Let's look at the constraint: "Products of every nonempty consecutive block must be distinct."
    # If we have a set S, and we sort it to S = [x_1, ..., x_k].
    # The set of all block products is P(S) = { prod(S[i:j]) for 0 <= i < j <= k }.
    # We require |P(S)| = k(k+1)/2.
    
    # Is there a pairwise conflict?
    # If u, v in S.
    # If u * v == w for some w in S, and w is adjacent to u or v?
    # If S = [u, w, v] and u*w = v, then block [u,w] product is v, and block [v] product is v. Collision.
    # So u, w, v cannot all be in S.
    # This is a 3-way conflict.
    
    # However, if we define the conflict graph G on vertices {2..128} such that:
    # Edge (u, v) exists if there is NO valid sequence containing both u and v.
    # This is a global property. Computing this exactly is hard.
    
    # Alternative Interpretation from Directive:
    # "edge connects two numbers if including both in a valid sequence causes any consecutive block product collision"
    # This phrasing is ambiguous. Does it mean:
    # 1. For ANY valid sequence containing u and v, a collision occurs? (i.e., u and v are mutually exclusive in ANY valid sequence)
    # 2. Or does it mean u and v form a "bad pair" that *can* cause a collision?
    
    # If interpretation 1 is true, then MIS on this graph gives the maximum subset of numbers that can *potentially* be in a valid sequence? No, MIS gives a set of vertices with no edges, meaning no pair in the set is "mutually exclusive".
    # But even if no pair is mutually exclusive, the triple might be.
    
    # Let's reconsider the "Pairwise Conflict" approximation often used in these heuristics:
    # A common necessary condition for a set S to be "product-free" in the sense of block products is that no product of a subset of S equals another element in S (if that element is adjacent in a way that creates a block).
    # Actually, if S is a set of numbers, and we sort them, a collision happens if:
    # prod(S[i..j]) == prod(S[k..l]) with (i,j) != (k,l).
    # This implies prod(S[i..j]) / prod(S[k..l]) = 1.
    
    # Let's try a different structural approach:
    # Instead of a static conflict graph which might be too weak (allowing invalid triples) or too strong (disallowing valid pairs),
    # we can use a **Greedy Set Construction with Backtracking on Conflicts**.
    # But the directive specifically asks for MIS.
    
    # Let's define a "Strong Conflict" graph:
    # u and v conflict if u * v is in {2..128}.
    # Why? If u * v = w, and we have u, w, v in the sequence.
    # If the order is u, w, v (which is possible if u < w < v), then block [u, w] has product w, and block [v] has product w. Collision.
    # If the order is u, v, w (impossible since u<v and u*v=w implies w>v), so w is always larger.
    # So if u, v, w are in S, and u*v=w, then in the sorted sequence, u and v come before w.
    # If u and v are adjacent in S, then block [u,v] product is w, colliding with block [w].
    # If u and v are NOT adjacent in S, say u, x, v, w.
    # Then block [u, x, v] product is u*x*v. Not necessarily w.
    # But block [u, v] is not a consecutive block if x is between them.
    # So the collision only happens if u and v are adjacent in the selected sequence.
    
    # This means the conflict depends on adjacency. MIS on a static graph doesn't capture adjacency.
    
    # HOWEVER, if we enforce that for any u, v in S, u*v is NOT in S, does that help?
    # If u*v = w is not in S, then we don't have that specific collision.
    # What if u*v = x*y where x,y are also in S?
    # Then if u,v are adjacent and x,y are adjacent, we have a collision.
    
    # Given the complexity, let's implement the **MIS on "Multiplicative Dependency" Graph**.
    # Vertices: 2..128.
    # Edge (u, v) if u * v <= 128.
    # Rationale: If u * v <= 128, then w = u * v is a candidate.
    # If we pick u and v, we risk picking w (or w is already picked) and causing a collision if u,v are adjacent.
    # By removing edges where u*v <= 128, we ensure that for any pair in the independent set, their product is > 128 (or not an integer in range, but they are integers).
    # Wait, if u*v > 128, then w is not in the candidate set. So the collision "product of block equals another element" is impossible for that pair.
    # This eliminates the "product equals element" class of collisions.
    # It does NOT eliminate "product of block A equals product of block B" where both products are > 128.
    # But this is a strong necessary condition.
    
    # Let's build this graph and solve MIS.
    
    # Build Adjacency Bitsets
    # n = 127 vertices
    # adj[i] is a bitset of vertices connected to i
    
    adj = [0] * num_candidates
    
    for i in range(num_candidates):
        u = candidates[i]
        for j in range(i + 1, num_candidates):
            v = candidates[j]
            if u * v <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Tomita's Algorithm for Maximum Independent Set
    # We use bitsets (integers) for speed.
    
    # Precompute degrees for ordering? Tomita usually sorts by degree.
    # For simplicity and speed in Python, we'll use a recursive branch and bound with bitsets.
    
    best_set_bits = 0
    best_set_len = 0
    
    # To speed up, we can use a heuristic to order vertices.
    # Vertices with high degree are more "conflicting". 
    # In MIS, we often branch on high-degree vertices or use a specific ordering.
    # Let's stick to the natural order but use a strong pruning bound.
    
    # Bound: The maximum number of vertices we can add from a set U is bounded by the size of a maximal independent set in the subgraph induced by U.
    # A simple upper bound is the size of U.
    # A better bound: Color the graph induced by U. The number of colors is an upper bound on the size of the MIS.
    # Implementing a fast coloring for bitsets in Python might be slow.
    # Let's use a simpler bound: 
    # If we have a candidate set U, and current set size k, 
    # we can prune if k + len(U) <= best_set_len.
    
    # To make this efficient, we need to represent U as a bitset.
    
    def mis_search(U, P):
        """
        U: bitset of available vertices
        P: bitset of vertices in the current independent set
        """
        nonlocal best_set_bits, best_set_len
        
        if U == 0:
            if bin(P).count('1') > best_set_len:
                best_set_len = bin(P).count('1')
                best_set_bits = P
            return
            
        # Pruning
        # Upper bound on size of MIS in U
        # Simple bound: popcount(U)
        if bin(P).count('1') + bin(U).count('1') <= best_set_len:
            return
            
        # Time check
        if time.time() > deadline:
            return
            
        # Pick a vertex from U
        # Heuristic: Pick the vertex with the highest degree in the subgraph induced by U?
        # Or just the lowest index?
        # Let's pick the lowest index to keep it deterministic and simple.
        # Actually, picking a vertex with high degree in U reduces the search space faster.
        
        # Find a vertex v in U
        # To optimize, we can precompute a list of vertices and pick one.
        # For now, lowest index.
        v_idx = (U & -U).bit_length() - 1
        v_bit = 1 << v_idx
        
        # Branch 1: Include v
        # New U is U \ (N(v) U {v})
        # N(v) in the original graph is adj[v].
        # But we only care about neighbors in U.
        # So new_U = U & ~(adj[v] | v_bit)
        
        new_U1 = U & ~(adj[v_idx] | v_bit)
        mis_search(new_U1, P | v_bit)
        
        # Branch 2: Exclude v
        # New U is U \ {v}
        new_U2 = U & ~v_bit
        mis_search(new_U2, P)
        
    # Initial U is all vertices
    initial_U = (1 << num_candidates) - 1
    mis_search(initial_U, 0)
    
    # Convert best_set_bits to list of values
    if best_set_bits == 0:
        # Fallback if MIS failed or found nothing (should not happen, singletons are always valid)
        return [2]
        
    best_seq = []
    for i in range(num_candidates):
        if (best_set_bits >> i) & 1:
            best_seq.append(candidates[i])
            
    # The MIS solution guarantees that for any pair u, v in best_seq, u*v > 128.
    # This ensures no "product equals element" collisions.
    # However, it does NOT guarantee that "product of block A != product of block B" for larger blocks.
    # We MUST verify the final sequence for full validity.
    
    # Verification Function
    def is_valid_sequence(seq):
        if not seq:
            return True
        if len(seq) == 1:
            return True
        if seq != sorted(seq):
            return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        if len(set(seq)) != len(seq):
            return False
            
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                # Optimization: if prod > 128*128... actually products can be huge.
                # But we just need distinctness.
                if prod in prods:
                    return False
                prods.add(prod)
                # Early break if prod is too large? No, large numbers are distinct from small ones.
                # But if prod exceeds some limit, it's unlikely to collide with previous small prods?
                # Previous prods are at most 128^k.
                # Actually, just checking set membership is O(1) average.
                # The number of prods is O(n^2). n is small (<= 127).
                # 127^2 / 2 ~ 8000 checks. Fast.
        return True

    if is_valid_sequence(best_seq):
        return best_seq
    
    # If the MIS solution is invalid (due to higher-order collisions), we need to fix it.
    # We can try to repair the sequence by removing elements that cause collisions.
    # Or, we can use the MIS result as a "seed" for a local search or greedy repair.
    
    # Strategy: Greedy Repair
    # Start with the MIS sequence.
    # Check for collisions.
    # If a collision is found between block A and block B, remove one of the elements involved.
    # Which one? The one that causes the most conflicts?
    # Simpler: Just iterate through the sequence, and if adding an element causes a collision with previous blocks, skip it?
    # No, the sequence is already built.
    
    # Let's do a "Prune and Verify" loop.
    # While invalid:
    #   Find a collision.
    #   Remove an element from the smaller of the two colliding blocks? Or the one with higher "degree" in the conflict sense?
    #   Re-verify.
    
    # Given the time limit, a simple greedy construction from scratch might be faster if the MIS path fails validation.
    # But let's try a simple repair first.
    
    seq = best_seq[:]
    valid = False
    attempts = 0
    while not valid and attempts < 100 and time.time() < deadline - 5:
        valid = is_valid_sequence(seq)
        if valid:
            break
            
        # Find a collision
        n = len(seq)
        prods = {}
        collision_found = False
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    # Collision between (i,j) and (prods[prod][0], prods[prod][1])
                    # Remove an element. Let's remove seq[j] as a heuristic (later elements might be less critical?)
                    # Or remove the element that appears in the most collisions?
                    # For simplicity, remove seq[j].
                    seq.pop(j)
                    collision_found = True
                    break
                else:
                    prods[prod] = (i, j)
            if collision_found:
                break
        
        if not collision_found:
            break # Should not happen if is_valid_sequence returned False
            
        attempts += 1
        
    if is_valid_sequence(seq):
        return seq
        
    # Fallback: Pure Greedy
    seq = []
    prods = set()
    suffix_prods = []
    for x in range(2, 129):
        new_sp = [x]
        valid_add = True
        if x in prods:
            valid_add = False
        else:
            for p in suffix_prods:
                np = x * p
                if np in prods:
                    valid_add = False
                    break
                new_sp.append(np)
        
        if valid_add:
            seq.append(x)
            prods.update(new_sp)
            suffix_prods = new_sp
            
    return seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

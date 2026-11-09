"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import random

def _get_prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _generate_7_smooth(max_val):
    """Generate all 7-smooth numbers <= max_val."""
    smooths = set()
    # 2^a * 3^b * 5^c * 7^d <= max_val
    # Max powers:
    # 2^14 = 16384, 2^15 = 32768
    # 3^8 = 6561, 3^9 = 19683
    # 5^5 = 3125, 5^6 = 15625
    # 7^4 = 2401, 7^5 = 16807
    
    for a in range(0, 15):
        val_a = (2 ** a)
        if val_a > max_val: break
        for b in range(0, 10):
            val_ab = val_a * (3 ** b)
            if val_ab > max_val: break
            for c in range(0, 7):
                val_abc = val_ab * (5 ** c)
                if val_abc > max_val: break
                for d in range(0, 6):
                    val_abcd = val_abc * (7 ** d)
                    if val_abcd > max_val: break
                    smooths.add(val_abcd)
    return sorted(smooths)

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of
    prime factors of all pairwise sums (distinct indices) is minimized.
    
    Strategy:
    1. Target a set of 12 numbers where all pairwise sums are 7-smooth.
       This limits the prime factors to {2, 3, 5, 7}, so the score is 4.
    2. Build a graph G where vertices are integers 1..10000.
       Edge (u, v) exists if u + v is 7-smooth.
    3. Find a clique of size 12 in G.
    
    Optimization:
    - The graph is sparse. We can build adjacency lists by iterating
      through all 7-smooth sums S and adding edges (u, S-u).
    - Use backtracking with forward checking to find the clique.
    """
    max_val = 10000
    n_needed = 12
    max_sum = 2 * max_val
    
    # Generate all 7-smooth numbers up to max_sum
    smooth_sums = _generate_7_smooth(max_sum)
    
    # Build adjacency list for the graph
    # We only care about vertices 1..max_val
    # adj[u] is a set of v such that u < v <= max_val and u+v is 7-smooth
    adj = [set() for _ in range(max_val + 1)]
    
    for s in smooth_sums:
        if s > max_sum:
            break
        # u + v = s, with 1 <= u < v <= max_val
        # u ranges from max(1, s - max_val) to s // 2
        min_u = max(1, s - max_val)
        max_u = s // 2
        
        for u in range(min_u, max_u + 1):
            v = s - u
            if v <= max_val and v > u:
                adj[u].add(v)
                adj[v].add(u)
    
    # Filter vertices that have at least n_needed - 1 neighbors
    # A clique of size k requires each node to have degree >= k-1 within the clique
    # So globally degree >= k-1 is a necessary condition
    potential_vertices = [i for i in range(1, max_val + 1) if len(adj[i]) >= n_needed - 1]
    
    if not potential_vertices:
        # Fallback if no such graph structure exists (unlikely)
        # Try a simpler heuristic: powers of 2
        powers = [2**i for i in range(0, 14) if 2**i <= max_val]
        return powers[:n_needed]
    
    # Sort potential vertices by degree descending to prioritize high-connectivity nodes
    potential_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # Backtracking search for a clique of size n_needed
    # We maintain a list of candidates that are connected to all nodes in the current clique
    def find_clique(clique, candidates):
        if len(clique) == n_needed:
            return clique
        if not candidates:
            return None
        
        # We need to pick a node from candidates that is connected to all in clique
        # And then update candidates to be the intersection of their neighbors with the current candidates
        
        # Optimization: If the number of candidates is less than the remaining needed nodes, fail
        if len(candidates) < n_needed - len(clique):
            return None
            
        # Iterate through candidates
        # We can prune if a candidate's degree within candidates is too low
        # But simple iteration with forward checking is often sufficient for sparse graphs
        
        # To speed up, we can try the most connected candidates first
        # candidates is a list of vertices
        for i in range(len(candidates)):
            v = candidates[i]
            
            # Check if v is connected to all in clique
            # Since we maintain 'candidates' as nodes connected to ALL nodes in 'clique',
            # this check is implicitly satisfied if we update correctly.
            # Let's verify:
            # Initially clique=[], candidates=all potential.
            # Pick v. New clique=[v].
            # New candidates = {u in candidates \ {v} | u connected to v}
            
            new_clique = clique + [v]
            
            # Calculate new candidates
            # Must be in current candidates, not v, and connected to v
            new_candidates = []
            for u in candidates:
                if u == v:
                    continue
                if u in adj[v]:
                    new_candidates.append(u)
            
            # Recurse
            result = find_clique(new_clique, new_candidates)
            if result is not None:
                return result
                
        return None

    # Start search
    # We can start with an empty clique and all potential vertices as candidates
    # However, the first node choice is critical.
    # We iterate through potential_vertices as the first node.
    
    # To avoid redundant work, we can just call find_clique with empty clique
    # and the full list of potential_vertices.
    
    solution = find_clique([], potential_vertices)
    
    if solution:
        # Verify and return
        return solution
    
    # Fallback: If clique search fails (e.g., timeout or no clique),
    # return a set with minimal prime factors using a different heuristic.
    # Try to find 12 numbers with small prime factor sums.
    # Using numbers that are multiples of 2, 3, 5, 7 might help.
    
    # Simple fallback: First 12 7-smooth numbers themselves
    # Their sums might not be 7-smooth, but they are small.
    smooth_numbers = _generate_7_smooth(max_val)
    if len(smooth_numbers) >= n_needed:
        return smooth_numbers[:n_needed]
    
    # Ultimate fallback
    return list(range(1, n_needed + 1))

# EVOLVE-BLOCK-END

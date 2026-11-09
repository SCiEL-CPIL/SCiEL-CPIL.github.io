"""Rewritten program: structural change to graph-theoretic maximum clique search
with fast adjacency via prime factorization, greedy coloring + branch-and-bound,
multiple random restarts, and a final polish pass with single-node swaps."""
import time
import random
from math import gcd

N = 512
DEADLINE = 85.0  # safety margin before 90s kill


def _max_prime_exponent(n, p):
    """Return the exponent of prime p in n."""
    e = 0
    while n % p == 0:
        e += 1
        n //= p
    return e


def _build_adjacency():
    """
    Build adjacency structure for the graph where vertices are 1..512 and an edge
    exists between a and b iff lcm(a,b) <= 512.
    
    We use the fact that lcm(a,b) = a*b / gcd(a,b).
    For speed, precompute prime factorizations and use them to check lcm quickly.
    
    Actually, the fastest approach: for each pair (a,b), compute gcd(a,b) and check
    if a*b//gcd(a,b) <= 512. Since 512 is small, we can precompute all pairs.
    
    Even faster: note that lcm(a,b) <= 512 iff for every prime p, max(v_p(a), v_p(b)) <= v_p(512).
    Since 512 = 2^9, the only prime that matters is 2! Any odd prime factor in a or b
    would make lcm >= that odd prime > 1, but we need lcm <= 512. Wait, no: lcm can
    contain odd primes as long as the product stays <= 512.
    
    Let me reconsider. 512 = 2^9. So lcm(a,b) <= 512 means lcm(a,b) is a divisor of 512...
    No, that's not right either. lcm(a,b) <= 512 just means the integer lcm(a,b) is at most 512.
    
    For example, a=3, b=3: lcm=3 <= 512. a=3, b=5: lcm=15 <= 512. a=3, b=171: lcm=513 > 512.
    
    So we need to check all pairs. Let's just do it directly with gcd.
    """
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _greedy_color_order(adj, candidates):
    """
    Order candidates using greedy coloring for the branch-and-bound.
    Returns list of (vertex, color) sorted by color.
    Uses a faster greedy coloring approach.
    """
    if not candidates:
        return [], {}, 0
    
    # Convert to set for fast lookup
    cand_set = set(candidates)
    
    # Precompute degrees within candidates for sorting
    degrees = {v: len(adj[v] & cand_set) for v in candidates}
    
    # Sort by degree descending for better coloring
    uncolored = sorted(candidates, key=lambda v: -degrees[v])
    
    color_class = {}
    num_colors = 0
    
    while uncolored:
        color_set = []
        remaining = []
        
        for v in uncolored:
            # Check if v can be added to current color class
            can_add = True
            for u in color_set:
                if u in adj[v]:
                    can_add = False
                    break
            
            if can_add:
                color_set.append(v)
            else:
                remaining.append(v)
        
        # Assign this color
        for v in color_set:
            color_class[v] = num_colors
        
        num_colors += 1
        uncolored = remaining
    
    # Order: vertices sorted by their color (higher color first for B&B pruning)
    ordered = sorted(candidates, key=lambda v: color_class[v])
    return ordered, color_class, num_colors


def _max_clique_bb(adj, candidates, current_clique, best_clique, deadline):
    """
    Branch and bound maximum clique using greedy coloring upper bound.
    candidates: list of vertices that can be added (all adjacent to all in current_clique)
    current_clique: list of vertices in the current clique
    best_clique: list of vertices in the best clique found so far
    deadline: time limit for this B&B call
    Returns best_clique.
    """
    if time.time() > deadline:
        return best_clique
    
    if not candidates:
        if len(current_clique) > len(best_clique):
            best_clique = list(current_clique)
        return best_clique
    
    # Greedy coloring to get upper bound
    ordered, color_class, num_colors = _greedy_color_order(adj, candidates)
    
    # If even with all remaining colors we can't beat best, prune
    if len(current_clique) + num_colors <= len(best_clique):
        return best_clique
    
    # Process in reverse color order (higher colors first for better pruning)
    for i in range(len(ordered) - 1, -1, -1):
        if time.time() > deadline:
            break
            
        v = ordered[i]
        # Upper bound: current size + number of colors from i onwards
        if len(current_clique) + i + 1 <= len(best_clique):
            break
        
        # Add v to clique
        current_clique.append(v)
        
        # New candidates: vertices in ordered[:i] that are adjacent to v
        new_candidates = [u for u in ordered[:i] if u in adj[v]]
        
        best_clique = _max_clique_bb(adj, new_candidates, current_clique, best_clique, deadline)
        
        current_clique.pop()
    
    return best_clique


def _greedy_clique(adj, start_set):
    """
    Greedy heuristic: repeatedly pick the vertex with highest degree within remaining candidates.
    """
    clique = []
    candidates = set(start_set)
    
    while candidates:
        # Pick vertex with max degree within candidates
        best_v = max(candidates, key=lambda v: len(adj[v] & candidates))
        clique.append(best_v)
        candidates = candidates & adj[best_v]
    
    return clique


def _try_add_vertex(adj, clique, v):
    """Check if v can be added to clique (adjacent to all members)."""
    for u in clique:
        if u != v and v not in adj[u]:
            return False
    return True


def _polish_swap(adj, clique):
    """
    Try to improve clique by removing one vertex and adding two others,
    or removing one and adding one different vertex.
    """
    clique_set = set(clique)
    improved = False
    n_clique = len(clique)
    
    # Try single swap: remove one, try to add a non-member that's adjacent to all remaining
    for i in range(n_clique):
        removed = clique[i]
        remaining = [clique[j] for j in range(n_clique) if j != i]
        remaining_set = set(remaining)
        
        # Find all vertices adjacent to all in remaining
        addable = set()
        for v in range(1, N + 1):
            if v in remaining_set:
                continue
            if all(v in adj[u] for u in remaining):
                addable.add(v)
        
        # If we can add more than 1 vertex that weren't in original clique
        new_adds = addable - clique_set
        if len(new_adds) >= 1:
            # Try adding the best one (highest degree)
            best_new = max(new_adds, key=lambda v: len(adj[v]))
            new_clique = remaining + [best_new]
            if len(new_clique) > n_clique:
                return new_clique, True
    
    # Try double swap: remove two, add three
    # This is expensive, so only do it if we have time
    for i in range(n_clique):
        for j in range(i + 1, n_clique):
            removed = {clique[i], clique[j]}
            remaining = [clique[k] for k in range(n_clique) if k != i and k != j]
            remaining_set = set(remaining)
            
            addable = set()
            for v in range(1, N + 1):
                if v in remaining_set or v in removed:
                    continue
                if all(v in adj[u] for u in remaining):
                    addable.add(v)
            
            new_adds = addable - clique_set
            if len(new_adds) >= 2:
                # Pick best two
                sorted_adds = sorted(new_adds, key=lambda v: -len(adj[v]))
                # Check if the two are adjacent to each other
                for a in range(len(sorted_adds)):
                    for b in range(a + 1, len(sorted_adds)):
                        va, vb = sorted_adds[a], sorted_adds[b]
                        if vb in adj[va]:
                            new_clique = remaining + [va, vb]
                            if len(new_clique) > n_clique:
                                return new_clique, True
    
    return clique, False


def construct():
    start_time = time.time()
    
    # Build adjacency
    adj = _build_adjacency()
    
    # Initial greedy clique from all vertices
    best_clique = _greedy_clique(adj, set(range(1, N + 1)))
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return list(best_clique)
    
    # Multiple random restarts with greedy + branch-and-bound
    rng = random.Random(42)
    num_restarts = 0
    max_restarts = 50  # Reduced to allow more time per B&B
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # leave 10s for final B&B and polish
            break
        if num_restarts >= max_restarts:
            break
        num_restarts += 1
        
        # Random subset for greedy
        sample_size = min(N, rng.randint(100, 300))
        sample = rng.sample(range(1, N + 1), sample_size)
        greedy_result = _greedy_clique(adj, set(sample))
        
        if len(greedy_result) > len(best_clique):
            best_clique = greedy_result
        
        # Try branch-and-bound on candidates adjacent to all in best_clique
        if len(best_clique) > 0:
            candidates = set(range(1, N + 1))
            for v in best_clique:
                candidates = candidates & adj[v]
            candidates = candidates - set(best_clique)
            
            if candidates:
                # Use B&B with a longer time check (5 seconds per call)
                bb_deadline = time.time() + 5.0
                bb_result = _max_clique_bb(adj, list(candidates), list(best_clique), list(best_clique), bb_deadline)
                if len(bb_result) > len(best_clique):
                    best_clique = bb_result
    
    # Final intensive B&B with remaining time
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        if len(best_clique) > 0:
            candidates = set(range(1, N + 1))
            for v in best_clique:
                candidates = candidates & adj[v]
            candidates = candidates - set(best_clique)
            
            if candidates:
                bb_deadline = DEADLINE - 1.0
                bb_result = _max_clique_bb(adj, list(candidates), list(best_clique), list(best_clique), bb_deadline)
                if len(bb_result) > len(best_clique):
                    best_clique = bb_result
    
    # Polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 1:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            best_clique, improved = _polish_swap(adj, best_clique)
            if not improved:
                break
    
    return list(best_clique)

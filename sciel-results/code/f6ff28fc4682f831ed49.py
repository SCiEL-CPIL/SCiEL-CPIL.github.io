import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _generate_conflicts():
    """
    Generate all forbidden triples (a, b, c) with a < b < c 
    such that 1/a = 1/b + 1/c.
    Returns a list of tuples (a, b, c).
    """
    triples = []
    # 1/a = 1/b + 1/c => c = a*b / (b-a)
    # Let d = b-a, so b = a+d, c = a + a^2/d
    # For c to be integer, d must divide a^2.
    # Also c > b => a + a^2/d > a + d => a^2/d > d => d^2 < a^2 => d < a.
    # So 1 <= d < a.
    
    for a in range(1, 181):
        a2 = a * a
        # d must be a divisor of a^2 and d < a
        # Find divisors of a^2
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if c <= 180 and b < c:
                    triples.append((a, b, c))
    return triples

def _construct_conflict_graph(triples):
    """
    Build a conflict graph where an edge (u, v) exists if u and v 
    participate in a common forbidden triple.
    This is used for heuristic ordering and initial pruning.
    """
    n = 181 # 1..180, index 0 unused or 1..180
    # Using 1-based indexing for simplicity, size 181
    degree = [0] * (n + 1)
    neighbors = [set() for _ in range(n + 1)]
    
    for a, b, c in triples:
        # a conflicts with b, a with c, b with c
        for u, v in [(a, b), (a, c), (b, c)]:
            if u != v:
                if v not in neighbors[u]:
                    neighbors[u].add(v)
                    degree[u] += 1
                if u not in neighbors[v]:
                    neighbors[v].add(u)
                    degree[v] += 1
                    
    return degree, neighbors

def _greedy_seed(triples, degree):
    """
    Generate a good initial independent set using a greedy approach.
    Order vertices by degree (ascending) to pick 'safe' vertices first.
    """
    # Vertices 1..180
    vertices = list(range(1, 181))
    # Sort by degree ascending (less constrained first)
    vertices.sort(key=lambda x: degree[x])
    
    current_set = set()
    # For fast checking, we can just check against existing triples
    # But for greedy, we need to ensure no conflict with already picked.
    # A vertex v can be added if no triple (v, x, y) has x, y in current_set.
    
    # Pre-compute for each vertex, the list of triples it's in
    vertex_triples = {i: [] for i in range(1, 181)}
    for t in triples:
        for v in t:
            vertex_triples[v].append(t)
            
    for v in vertices:
        valid = True
        for t in vertex_triples[v]:
            # Check if the other two are in current_set
            others = [x for x in t if x != v]
            if others[0] in current_set and others[1] in current_set:
                valid = False
                break
        if valid:
            current_set.add(v)
            
    return current_set

def _branch_and_bound(triples, degree, initial_set, deadline):
    """
    Branch and bound search for maximum independent set in the 3-uniform hypergraph.
    We maintain a set of 'available' vertices and a 'chosen' set.
    """
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Pre-process triples for faster access
    # For a given vertex v, what are the triples it's in?
    vertex_triples = {i: [] for i in range(1, 181)}
    for t in triples:
        for v in t:
            vertex_triples[v].append(t)
    
    def is_valid_addition(chosen, v):
        """Check if v can be added to chosen without creating a forbidden triple."""
        for t in vertex_triples[v]:
            others = [x for x in t if x != v]
            if others[0] in chosen and others[1] in chosen:
                return False
        return True

    def search(chosen, available, depth, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: if we can't beat best_size, stop
        if current_size + len(available) <= best_size:
            return
            
        if not available:
            if current_size > best_size:
                best_set = set(chosen)
                best_size = current_size
            return

        # Heuristic: pick a vertex from available with highest degree in the induced subgraph?
        # Or just pick the first one? 
        # Better: Pick a vertex that is in many conflicts with other available vertices.
        # Let's pick a vertex v from available.
        # To maximize pruning, we can use a branching variable.
        # Branch on the first vertex in a sorted order.
        
        # Sort available by degree (descending) to pick constrained ones first?
        # Actually, for independent set, picking a vertex removes its neighbors.
        # Let's pick a vertex v.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # To make it efficient, we should pick a vertex that appears in many triples 
        # involving other available vertices.
        
        # Simple heuristic: pick the first vertex in the list.
        # But let's sort available by degree (in the full graph) descending.
        # This might help prune early if high-degree nodes are excluded.
        
        # Let's pick v = max(available, key=lambda x: degree[x])
        v = max(available, key=lambda x: degree[x])
        
        # Branch 1: Include v
        if is_valid_addition(chosen, v):
            new_chosen = chosen | {v}
            # New available: remove v and any vertex that conflicts with v given new_chosen?
            # No, if we include v, we must remove any u in available such that 
            # there is a triple (v, u, w) where w is in new_chosen.
            # Actually, simpler:
            # If we include v, we cannot include any u such that (v, u, w) is a triple 
            # AND w is in new_chosen.
            # Wait, the condition is: no THREE DISTINCT elements a,b,c in the FINAL set.
            # If we add v to chosen, we must ensure that for any triple (v, u, w), 
            # we don't have both u and w in the final set.
            # Since u, w are in available (not yet decided), we can't just remove them.
            # We must remove u from available if adding v forces a conflict?
            # No. Adding v to chosen doesn't immediately forbid u or w. 
            # It forbids the pair (u, w) if both were to be added later.
            # This makes the state space complex: we need to track pairwise conflicts 
            # induced by chosen vertices.
            
            # This standard B&B for Max Independent Set on Graphs works because 
            # conflicts are pairwise. Here they are 3-wise.
            # If I pick v, I don't remove neighbors. I just add v.
            # But I must ensure that I don't pick u and w later if (v,u,w) is a triple.
            # This requires tracking "forbidden pairs" in the available set.
            
            # This approach is getting complicated for 3-uniform.
            # Let's switch to a simpler, robust heuristic: 
            # Local Search with Restart (Simulated Annealing style) but with 
            # better move operators: 
            # 1. Add a vertex (if valid).
            # 2. Swap: remove v, add u.
            # 3. 2-Swap: remove v1, v2, add u1, u2.
            
            # The previous program did this. 
            # The key improvement from "inspiration 1" was using a conflict hypergraph 
            # and multi-order greedy.
            
            # Let's try a different angle: 
            # Since N=180 is small, we can use a **Randomized Local Search** 
            # with a **Conflict-Based Repair** strategy.
            
            # Or, we can use the fact that the graph is sparse.
            # Let's stick to the Branch and Bound but handle the 3-wise constraint 
            # by maintaining a set of "blocked pairs".
            
            # State: (chosen, available, blocked_pairs)
            # blocked_pairs: set of (u, w) such that we cannot add both u and w 
            # because there is a triple (v, u, w) with v in chosen.
            
            # This is feasible.
            
            pass # Placeholder, we will implement this properly below

    # Re-implementation of search with blocked pairs
    def search_v2(chosen, available, blocked_pairs, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if current_size + len(available) <= best_size:
            return
            
        if not available:
            if current_size > best_size:
                best_set = set(chosen)
                best_size = current_size
            return

        # Pick a vertex to branch on
        # Heuristic: Pick a vertex involved in the most blocked pairs? 
        # Or just the first one?
        # Let's pick a random one or the one with highest degree.
        v = max(available, key=lambda x: degree[x])
        
        # Branch 1: Include v
        # Check if v is blocked by itself? No.
        # Check if v forms a triple with two other vertices in chosen? 
        # That would have been checked before adding v to chosen in the parent call.
        # So v is safe to add to chosen.
        
        # When we add v to chosen, we must add to blocked_pairs:
        # For each triple (v, u, w) where u, w are in available:
        # Add (min(u,w), max(u,w)) to blocked_pairs.
        
        new_chosen = chosen | {v}
        new_blocked = set(blocked_pairs)
        new_available = available - {v}
        
        # Find new blocked pairs
        for t in vertex_triples[v]:
            others = [x for x in t if x != v]
            u, w = others
            if u in new_available and w in new_available:
                if u < w:
                    new_blocked.add((u, w))
                else:
                    new_blocked.add((w, u))
                    
        # Recurse
        search_v2(new_chosen, new_available, new_blocked, current_size + 1)
        
        # Branch 2: Exclude v
        # Just remove v from available.
        search_v2(chosen, available - {v}, blocked_pairs, current_size)

    # Start search
    initial_available = set(range(1, 181)) - initial_set
    # Note: We must ensure initial_set is valid.
    # And we must initialize blocked_pairs based on initial_set.
    initial_blocked = set()
    for v in initial_set:
        for t in vertex_triples[v]:
            others = [x for x in t if x != v]
            u, w = others
            if u in initial_available and w in initial_available:
                if u < w:
                    initial_blocked.add((u, w))
                else:
                    initial_blocked.add((w, u))
                    
    search_v2(initial_set, initial_available, initial_blocked, len(initial_set))
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80 # Safety margin
    
    triples = _generate_conflicts()
    degree, neighbors = _construct_conflict_graph(triples)
    
    # Get a good initial seed
    best_seed = _greedy_seed(triples, degree)
    
    # Run B&B
    best_set = _branch_and_bound(triples, degree, best_seed, deadline)
    
    # If B&B didn't finish or found nothing better, fall back to local search
    # to ensure we return something valid and large.
    
    # Actually, B&B returns the best found. 
    # If it times out, it returns the best found so far.
    
    # Verify result
    if not best_set:
        # Fallback to a simple valid set
        best_set = set(range(91, 181))
        
    return list(best_set)

# EVOLVE-BLOCK-END

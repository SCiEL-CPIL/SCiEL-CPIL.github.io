import time
import random
from itertools import combinations

def _has_c5(adj, n):
    # Check for 5-cycle: v0-v1-v2-v3-v4-v0 with all distinct
    # Use adjacency matrix for speed
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 in (v0, v1):
                    continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1, v2):
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2, v3):
                            continue
                        if v4 in adj[v0] and v4 != v1:
                            return True
    return False

def _max_independent_set(adj, n, limit=9):
    # Bron-Kerbosch variant: find if independent set of size >= limit exists
    # Independent set in G = clique in complement
    # Use greedy + backtracking
    def bronk(R, P, X):
        if len(R) >= limit:
            return True
        if not P and not X:
            return False
        # Pick pivot
        if P:
            pivot = P[0]
            candidates = [v for v in P if v not in adj[pivot]]  # non-neighbors of pivot in G = neighbors in complement
            # Wait: independent set means no edges between them in G
            # So in complement, we want cliques. Let me redo.
            # Independent set in G: set S where no two vertices in S are adjacent in G.
            # This is a clique in the complement graph.
            # Let me use a simpler approach: greedy with backtracking on G directly.
            pass
        return False
    
    # Simpler: use greedy coloring bound + backtracking
    # For n up to 35, we can use a simple branch and bound
    def has_ind_set(adj, n, k):
        # Try to find independent set of size k
        # Use bitmasks for n <= 35
        if n <= 32:
            # Use bitmask
            pass
        # Simple recursive approach
        def rec(remaining, size_needed):
            if size_needed == 0:
                return True
            if len(remaining) < size_needed:
                return False
            # Greedy: pick a vertex, either include or exclude
            v = remaining[0]
            rest = remaining[1:]
            # Include v: remove v and all its neighbors
            new_remaining = [u for u in rest if u not in adj[v]]
            if rec(new_remaining, size_needed - 1):
                return True
            # Exclude v
            if rec(rest, size_needed):
                return True
            return False
        
        return rec(list(range(n)), k)
    
    return has_ind_set(adj, n, limit)

def _is_valid(adj, n):
    if _has_c5(adj, n):
        return False
    if _max_independent_set(adj, n, 9):
        return False
    return True

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Fallback: 8 copies of K4 = 32 vertices
    fallback_edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                fallback_edges.append([base + i, base + j])
    fallback_n = 32
    
    best_n = fallback_n
    best_edges = fallback_edges
    
    # Try to find larger graphs using random search
    # Strategy: start with dense graph, remove edges to eliminate C5, check independence
    # Or: build graph as union of cliques with some inter-clique edges
    
    # Approach: Use a known construction pattern
    # Try n from 35 down to best_n+1
    random.seed(42)
    
    for target_n in range(35, best_n, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random constructions for this n
        attempts = 0
        while time.time() < deadline - 5 and attempts < 200:
            attempts += 1
            
            # Construction: partition into cliques of size 4-5, add some inter-clique edges
            # to reduce independence number
            n = target_n
            edges = []
            
            # Start with disjoint K4s and K5s
            remaining = n
            clique_sizes = []
            while remaining > 0:
                if remaining >= 5 and random.random() < 0.5:
                    clique_sizes.append(5)
                    remaining -= 5
                elif remaining >= 4:
                    clique_sizes.append(4)
                    remaining -= 4
                else:
                    clique_sizes.append(remaining)
                    remaining = 0
            
            # Build cliques
            pos = 0
            for cs in clique_sizes:
                for i in range(cs):
                    for j in range(i+1, cs):
                        edges.append([pos+i, pos+j])
                pos += cs
            
            # Add random inter-clique edges to reduce independence
            # But avoid creating C5
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            # Remove already-present edges
            edge_set = set((min(u,v), max(u,v)) for u, v in edges)
            candidate_edges = [(u,v) for u,v in all_pairs if (min(u,v), max(u,v)) not in edge_set]
            
            # Add some random edges
            num_extra = random.randint(0, min(len(candidate_edges), 15))
            extra_edges = random.sample(candidate_edges, num_extra) if candidate_edges else []
            for u, v in extra_edges:
                edges.append([u, v])
            
            # Check validity
            adj = _build_adj(edges, n)
            if _is_valid(adj, n):
                best_n = n
                best_edges = edges
                break
            # Also try: start from complement perspective
            # High density graph with no C5 and alpha < 9
    
    # If we found nothing better, try a structured approach
    if best_n == fallback_n and time.time() < deadline - 10:
        # Try: 7 K5s = 35 vertices, but alpha = 7 < 9, need to check C5
        # K5 contains C5! So can't use K5.
        # Use K4s: 8 K4s = 32, alpha = 8 < 9. Good.
        # Can we add vertices? 33, 34, 35?
        # Try: 7 K4s + 1 K5? No, K5 has C5.
        # Try: 6 K4s + 2 K5? No.
        # Try: mix of K3, K4 with inter-clique edges
        # For n=33: 8 K4s + 1 vertex. Add vertex 32 connected to some others.
        # If vertex 32 is connected to all vertices in one K4, then independence can't use all 4 from that K4.
        # Actually if v is connected to a, b, c, d (a K4), then {v} ∪ (independent from rest) 
        # The independent set can include v plus at most 3 from each other K4 (since v blocks the whole K4 it's connected to... no)
        # If v is connected to all 4 vertices of K4_0, then an independent set containing v cannot contain any vertex from K4_0.
        # So alpha <= 1 + 7*1 = 8 if v is connected to all of one K4. Wait, from each other K4, we can pick at most 1 vertex (since they form K4).
        # So alpha <= 1 + 7 = 8 < 9. 
        # But does adding v create a C5? v is connected to 4 vertices in K4_0. Those 4 vertices are all connected to each other.
        # A C5 through v: v-a-b-c-d-v. a,b,c,d must be distinct, a in N(v), d in N(v), b adjacent to a, c adjacent to b and d.
        # If a,b,c,d are all in K4_0, then a,b,c,d form K4, so a-b-c-d-v-a is a 5-cycle? 
        # v-a, a-b (yes, in K4), b-c (yes), c-d (yes), d-v (yes). And all distinct. That's a C5!
        # So we can't connect v to 4 vertices in a K4.
        # Connect v to 3 vertices in K4_0: then v-a-b-c-v is a 4-cycle, not 5.
        # Can we get a C5? v-a-x-y-z-v where a in N(v)∩K4_0, z in N(v)∩K4_0, x,y elsewhere.
        # If x,y are in different K4s, they're not adjacent (no inter-clique edges initially).
        # So no C5 if we only connect v to vertices in one K4 and keep other cliques disjoint.
        # So: 8 K4s + vertex 32 connected to 3 vertices of K4_0.
        # alpha: v can be in independent set only if we exclude its 3 neighbors in K4_0. From K4_0, we can pick at most 1 (the non-neighbor). From other 7 K4s, pick 1 each. Total: 1+1+7 = 9. That's not < 9!
        # So alpha = 9, invalid.
        # Connect v to 4 vertices? Creates C5 as shown.
        # Connect v to vertices in multiple K4s?
        # Let's connect v to 2 vertices in K4_0 and 2 vertices in K4_1.
        # alpha: if v is in IS, exclude 2 from K4_0 (can pick at most 2 from K4_0 since 2 are blocked, and the remaining 2 are adjacent so pick at most 1). Similarly for K4_1.
        # From K4_0: 4 vertices, 2 blocked, remaining 2 are adjacent, pick at most 1.
        # From K4_1: same, pick at most 1.
        # From other 6 K4s: pick 1 each.
        # Total: 1 + 1 + 1 + 6 = 9. Still 9!
        # Connect v to 2 in K4_0, 1 in K4_1, 1 in K4_2.
        # K4_0: 2 blocked, 2 remaining adjacent, pick 1.
        # K4_1: 1 blocked, 3 remaining form K3, pick 1.
        # K4_2: same, pick 1.
        # Other 5 K4s: pick 1 each.
        # Total: 1+1+1+1+5 = 9. Still 9!
        # It seems hard to get alpha < 9 with 33 vertices using this structure.
        # The 8 K4s give alpha = 8. Adding any vertex with limited connections keeps alpha >= 9.
        # To get alpha < 9 with n=33, we need a different structure.
        pass
    
    return {'n': best_n, 'edges': best_edges}

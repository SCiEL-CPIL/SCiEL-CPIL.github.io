import random
import time
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Check if graph has a 5-cycle. adj is list of sets."""
    # For each 5-vertex subset, check if it contains a C5
    # A 5-cycle on vertices {a,b,c,d,e} means there's a permutation where each consecutive pair (and wrap) is an edge
    # More efficient: for each pair of non-adjacent vertices in the induced subgraph on 5 vertices,
    # check if the remaining structure forms a path of length 4 between them
    # Actually, let's use: a set of 5 vertices contains a C5 iff the induced subgraph on them has a Hamiltonian cycle
    # For 5 vertices, we can just check all 5! / 2 = 60 possible cycles, or use a simpler check:
    # In a 5-vertex graph, a C5 exists iff there are exactly 5 vertices each of degree 2 in the induced subgraph
    # that form a single cycle. But easier: check if the complement of the induced subgraph on 5 vertices
    # is a C5 as well (since C5 is self-complementary). No, that's not right.
    # Let's just do: for each 5-set, check if the induced subgraph has a cycle of length 5.
    # For n up to 35, C(35,5) = 324632, and for each we do a quick check.
    
    # Faster approach: for each vertex v, look for a path of length 4 starting and ending near v.
    # Even faster: enumerate all 5-tuples (a,b,c,d,e) with a<b<c<d<e and check if 
    # (a,b),(b,c),(c,d),(d,e),(e,a) are edges, or other permutations.
    # There are 12 distinct cycle structures on 5 labeled vertices (up to rotation/reflection).
    # Actually, the number of distinct 5-cycles on a fixed 5-set is (5-1)!/2 = 12.
    
    # Let's use a smarter method: 
    # A 5-cycle exists iff there exist distinct vertices v0,v1,v2,v3,v4 such that
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0 are all edges.
    # We can fix v0 and v1 as adjacent, then look for v2 adjacent to v1 (not v0),
    # v3 adjacent to v2 (not v0,v1), v4 adjacent to v3 (not v0,v1,v2) and v4 adjacent to v0 (not v1,v2,v3).
    
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            # v2 adjacent to v1, v2 != v0, v2 > v0 to avoid duplicates... actually let's not worry about duplicates
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1):
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2):
                            continue
                        if v4 in adj[v0]:
                            # Found a 5-cycle: v0-v1-v2-v3-v4-v0
                            return True
    return False

def _independence_number(n, adj):
    """Compute exact independence number using branch and bound."""
    # adj is list of sets
    # Use greedy lower bound and branch and bound
    
    best = [0]
    
    def greedy_max_independent(vertices):
        """Greedy: repeatedly pick vertex with min degree in remaining subgraph."""
        remaining = set(vertices)
        result = []
        while remaining:
            # Pick vertex with minimum degree in the induced subgraph
            min_deg = float('inf')
            pick = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and u in adj[v])
                if deg < min_deg:
                    min_deg = deg
                    pick = v
            result.append(pick)
            remaining.discard(pick)
        return result
    
    def bb(vertices, current_size, parent_size):
        if not vertices:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(vertices)
        if current_size + len(vertices) <= best[0]:
            return
        
        # Greedy lower bound
        greedy_set = greedy_max_independent(vertices)
        if current_size + len(greedy_set) > best[0]:
            best[0] = current_size + len(greedy_set)
        
        # If even greedy doesn't beat best, still need to check but we can be optimistic
        # Pick a vertex to branch on
        # Choose vertex with highest degree in the subgraph for better pruning
        v = None
        max_deg = -1
        for u in vertices:
            deg = sum(1 for w in vertices if w != u and w in adj[u])
            if deg > max_deg:
                max_deg = deg
                v = u
        
        vset = set(vertices)
        
        # Branch 1: include v
        # All neighbors of v cannot be included
        remaining_without_neighbors = vset - adj[v] - {v}
        bb(remaining_without_neighbors, current_size + 1, parent_size + 1)
        
        # Branch 2: exclude v
        remaining_without_v = vset - {v}
        bb(remaining_without_v, current_size, parent_size + 1)
    
    all_vertices = set(range(n))
    bb(all_vertices, 0, 0)
    return best[0]

def _try_construct_33():
    """Try to find a 33-vertex C5-free graph with alpha <= 8."""
    start_time = time.time()
    deadline = start_time + 100  # 100s budget
    
    # Start with 8xK4 + 1 vertex. The new vertex must be connected to at least 
    # some vertices to avoid creating independent sets.
    # Key insight: alpha(8xK4) = 8, so adding any vertex gives alpha >= 9 unless
    # the new vertex is connected to at least one vertex from each K4 block.
    # Actually, the new vertex v is independent from any set of vertices it's NOT connected to.
    # An independent set of size 9 could be: v + 8 vertices, one from each K4 block, 
    # provided v is not connected to any of those 8.
    # So v must be connected to at least one vertex in each K4 block.
    
    # Let's try: connect new vertex 32 to one vertex in each of the 8 K4 blocks.
    # Then check if there's a C5.
    
    n = 33
    adj = [set() for _ in range(n)]
    
    # Build 8xK4
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
    
    # Connect vertex 32 to one vertex from each block
    # Try different patterns
    best_result = None
    best_n = 0
    
    # Pattern: connect to vertex 0 of each block (indices 0, 4, 8, 12, 16, 20, 24, 28)
    for pattern in range(4**8):  # 65536 patterns, might be too many
        if time.time() > deadline:
            break
        
        adj2 = [set(s) for s in adj]
        
        valid = True
        for b in range(8):
            choice = (pattern >> (3 * b)) & 7
            if choice >= 4:
                # Invalid, skip
                valid = False
                break
            target = 4 * b + choice
            adj2[32].add(target)
            adj2[target].add(32)
        
        if not valid:
            continue
        
        # Check for C5
        if _has_c5(n, adj2):
            continue
        
        # Check independence number
        alpha = _independence_number(n, adj2)
        if alpha <= 8:
            # Found a valid graph!
            edges = []
            for u in range(n):
                for v in adj2[u]:
                    if u < v:
                        edges.append([u, v])
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'edges': edges}
            return best_result
    
    return best_result

def _try_construct_34():
    """Try to find a 34-vertex C5-free graph with alpha <= 8."""
    start_time = time.time()
    deadline = start_time + 100
    
    # 8xK4 + 2 vertices. Much harder.
    # Each new vertex must be connected to at least one vertex in each K4 block
    # to prevent independent 9-sets.
    # This is a search problem. Let's try a heuristic approach.
    
    n = 34
    adj = [set() for _ in range(n)]
    
    # Build 8xK4
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
    
    # Try to add vertices 32 and 33 with connections that avoid C5 and keep alpha <= 8
    # This is complex. Let's try a few random configurations.
    
    random.seed(42)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        adj2 = [set(s) for s in adj]
        
        # For each of vertices 32, 33, connect to one random vertex in each K4 block
        valid = True
        for new_v in [32, 33]:
            for b in range(8):
                choice = random.randint(0, 3)
                target = 4 * b + choice
                adj2[new_v].add(target)
                adj2[target].add(new_v)
        
        # Also consider adding edges between 32 and 33
        if random.random() < 0.5:
            adj2[32].add(33)
            adj2[33].add(32)
        
        # Check for C5
        if _has_c5(n, adj2):
            continue
        
        # Check independence number
        alpha = _independence_number(n, adj2)
        if alpha <= 8:
            edges = []
            for u in range(n):
                for v in adj2[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None

def _base_32():
    """Return the 8xK4 construction with n=32."""
    n = 32
    adj = [set() for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return {'n': n, 'edges': edges}

def construct():
    start_time = time.time()
    total_deadline = start_time + 140  # 140s to be safe within 150s
    
    # Try to find larger constructions
    result_34 = None
    result_33 = None
    
    # Try 34 first (more ambitious)
    if time.time() < total_deadline - 120:
        result_34 = _try_construct_34()
    
    # Try 33
    if result_34 is None and time.time() < total_deadline - 60:
        result_33 = _try_construct_33()
    
    # Return the best found
    if result_34 is not None:
        return result_34
    if result_33 is not None:
        return result_33
    
    # Fall back to 32
    return _base_32()
# EVOLVE-BLOCK-END

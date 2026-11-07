"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 15

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    # The constraint: for any path s-u-v-t (3 edges, s!=t), deg(s) != deg(t)
    # This means: for any edge (u,v), no two neighbors of u (other than v) can have
    # the same degree as any neighbor of v (other than u)
    # Actually: for edge (u,v), pick s in N(u)\{v} and t in N(v)\{u}, s!=t, then deg(s)!=deg(t)
    
    # Key observation: the constraint is on the degree sequence + adjacency structure
    # If two vertices have the same degree, they cannot both be "reachable" as endpoints
    # of a 3-path through a common edge.
    
    # Let's think about it differently:
    # A 3-path s-u-v-t exists when: s~u, u~v, v~t, s!=t
    # The endpoints s and t must have different degrees.
    # So for each edge (u,v), the set of "left endpoints" = N(u)\{v} and "right endpoints" = N(v)\{u}
    # For any s in left, t in right with s!=t: deg(s) != deg(t)
    
    # This means: all vertices in N(u)\{v} must have distinct degrees from all vertices in N(v)\{u} (except possibly s=t cases)
    # Actually s and t just need s!=t, so if deg(s)==deg(t) and s!=t, that's forbidden.
    
    # Strategy: try to build a dense graph. Start with complete graph and remove edges.
    # Or: start from a good structure and add edges greedily.
    
    # Let's try a different approach: use the fact that if we can make all degrees distinct,
    # then no two endpoints can have equal degree, so ANY graph with all distinct degrees is valid!
    # 
    # Wait, that's not right. The constraint is: for a specific path s-u-v-t, deg(s) != deg(t).
    # If ALL degrees are distinct, then certainly deg(s) != deg(t) for any s != t.
    # So a graph with all distinct degrees is automatically valid!
    
    # Can we have a graph on 15 vertices with all distinct degrees?
    # Degrees range from 0 to 14. All distinct means degrees are exactly 0,1,2,...,14.
    # But a vertex of degree 14 is connected to all others, including the vertex of degree 0.
    # Contradiction: vertex of degree 0 can't be connected to anything.
    # So we can't have all 15 distinct degrees.
    
    # But we can have 14 distinct degrees (missing one value), or use other combinations.
    
    # Actually, let's reconsider. The constraint is:
    # For edge (u,v): for all s in N(u)\{v}, t in N(v)\{u}, if s != t then deg(s) != deg(t).
    
    # This is equivalent to: the multiset of degrees in N(u)\{v} and N(v)\{u} can share at most
    # one common value, and if they do, it must correspond to the same vertex (s=t).
    
    # Let me try a constructive approach: build a graph where degrees are mostly distinct.
    
    # Approach: Try to maximize edges by starting with a near-complete graph and removing
    # minimum edges to satisfy constraints.
    
    # Alternative: Use a greedy algorithm.
    # Start with no edges, add edges one by one if they don't violate the constraint.
    
    # Better: Start with a known valid dense graph and try to add more edges.
    
    # Let me think about what structures work well.
    # A star has 14 edges, all leaves have degree 1, center has degree 14.
    # Any 3-path: leaf-center-leaf-... wait, that's only 2 edges.
    # 3-path: s-u-v-t. In a star, the only vertex with degree > 1 is the center.
    # So paths must go through center: s-center-v-t where v is another leaf.
    # But center has degree 14, leaves have degree 1.
    # s is a leaf (deg 1), t is a leaf (deg 1). deg(s) == deg(t) == 1. VIOLATION!
    # Wait, so the star is NOT valid?
    
    # Let me re-read: "Reject if any simple three-edge path has endpoints of equal degree"
    # A three-edge path s-u-v-t has 4 vertices, 3 edges.
    # In a star on 15 vertices: center c, leaves l1,...,l14.
    # Path: l1 - c - l2 - ? There's no edge from l2 to anything except c.
    # So l1-c-l2 is a 2-edge path, not 3-edge.
    # For a 3-edge path, we need 4 distinct vertices with 3 edges.
    # In a star, the only edges are from center to leaves. So any path is at most 2 edges
    # (leaf-center-leaf). There are NO 3-edge paths in a star!
    # So the star IS valid with 14 edges.
    
    # Can we do better? Let's try adding an edge between two leaves.
    # Add edge (l1, l2). Now l1 and l2 have degree 2, others have degree 1, center has degree 14.
    # 3-edge paths: l1-l3-c-l2? No, that's l1-c-l2 which is 2 edges.
    # Actually: s-u-v-t with edges s-u, u-v, v-t.
    # Possible: l3-c-l1-l2: edges l3-c, c-l1, l1-l2. Endpoints l3 (deg 1) and l2 (deg 2). OK.
    # l3-c-l2-l1: edges l3-c, c-l2, l2-l1. Endpoints l3 (deg 1) and l1 (deg 2). OK.
    # l4-c-l1-l2: endpoints l4 (deg 1), l2 (deg 2). OK.
    # What about l1-l2-c-l3? edges l1-l2, l2-c, c-l3. Endpoints l1 (deg 2), l3 (deg 1). OK.
    # Seems fine! So 15 edges works.
    
    # Let's try more. Add edge (l1, l3). Now l1 has deg 3, l2 and l3 have deg 2, others deg 1.
    # Check: l4-c-l2-l1: endpoints l4(1), l1(3). OK.
    # l5-c-l3-l1: endpoints l5(1), l1(3). OK.
    # l2-c-l1-l3: endpoints l2(2), l3(2). VIOLATION! s=l2, u=c, v=l1, t=l3. deg(l2)=2, deg(l3)=2.
    # So we can't have both (l1,l2) and (l1,l3) if l2 and l3 both have degree 2.
    
    # So the structure matters. Let me use a greedy approach.
    
    def is_valid(edge_set, adj, deg):
        """Check if adding an edge would create a violation."""
        # Actually, let's check the full graph validity
        pass
    
    def check_graph_valid(adj, deg):
        """Check if the graph satisfies the constraint."""
        for u in range(N):
            for v in adj[u]:
                if v > u:  # avoid double counting
                    # For edge (u,v), check all s in N(u)\{v}, t in N(v)\{u}
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u:
                                continue
                        if s != t and deg[s] == deg[t]:
                            return False
        return True
    
    # Greedy: start with star, try to add edges between leaves
    # Vertices: 0 = center, 1..14 = leaves
    # Start with star: edges (0,1), (0,2), ..., (0,14)
    
    adj = [set() for _ in range(N)]
    edge_list = []
    
    # Add star edges
    for i in range(1, N):
        adj[0].add(i)
        adj[i].add(0)
        edge_list.append([0, i])
    
    deg = [len(adj[i]) for i in range(N)]
    
    # Try to add edges between leaves greedily
    # Sort leaf pairs by some heuristic
    leaf_pairs = [(i, j) for i in range(1, N) for j in range(i+1, N)]
    
    # Try each pair, add if valid
    added = 0
    for i, j in leaf_pairs:
        if time.time() > deadline:
            break
        # Tentatively add edge (i,j)
        adj[i].add(j)
        adj[j].add(i)
        deg[i] += 1
        deg[j] += 1
        
        if check_graph_valid(adj, deg):
            edge_list.append([i, j])
            added += 1
        else:
            # Remove
            adj[i].remove(j)
            adj[j].remove(i)
            deg[i] -= 1
            deg[j] -= 1
    
    # Now try to add edges involving the center? No, center already connected to all.
    # Try a different strategy: multiple restarts with different orderings
    
    best_edges = len(edge_list)
    best_result = [list(e) for e in edge_list]
    
    # Try randomized restarts
    import random
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        adj = [set() for _ in range(N)]
        edge_list = []
        
        for i in range(1, N):
            adj[0].add(i)
            adj[i].add(0)
            edge_list.append([0, i])
        
        deg = [len(adj[i]) for i in range(N)]
        
        leaf_pairs = [(i, j) for i in range(1, N) for j in range(i+1, N)]
        rng.shuffle(leaf_pairs)
        
        for i, j in leaf_pairs:
            if time.time() > deadline - 5:
                break
            adj[i].add(j)
            adj[j].add(i)
            deg[i] += 1
            deg[j] += 1
            
            if check_graph_valid(adj, deg):
                edge_list.append([i, j])
            else:
                adj[i].remove(j)
                adj[j].remove(i)
                deg[i] -= 1
                deg[j] -= 1
        
        if len(edge_list) > best_edges:
            best_edges = len(edge_list)
            best_result = [list(e) for e in edge_list]
    
    # Ensure proper types
    result = []
    for e in best_result:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END

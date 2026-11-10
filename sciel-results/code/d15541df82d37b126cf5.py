import time
import random
from collections import defaultdict
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n = 31
    
    # Build a C4-free graph with 90 edges using polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph
    edge_set = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    if len(edge_set) != 90:
        # Fallback if polarity graph construction fails (should not happen for q=5)
        # But we assume it works as per standard theory.
        pass
        
    # Convert to NumPy matrix for fast operations
    A = np.zeros((n, n), dtype=np.int32)
    for (u, v) in edge_set:
        A[u, v] = 1
        A[v, u] = 1
        
    # Function to count C4s using NumPy
    # Number of C4s = 0.5 * sum_{i,j} (M^2)_{ij}^2 - ... 
    # Actually, (A^2)_{ij} is number of paths of length 2 between i and j.
    # A C4 is i-j-k-l-i. 
    # The number of closed walks of length 4 is trace(A^4).
    # trace(A^4) = sum_i (A^4)_{ii} = sum_i sum_j sum_k sum_l A_{ij}A_{jk}A_{kl}A_{li}.
    # This counts each C4 multiple times.
    # Specifically, each C4 is counted 8 times (4 starting vertices * 2 directions).
    # Also, it counts "back-and-forth" walks like i-j-i-j-i which are not C4s.
    # Let's use the common neighbor formula which is easier to vectorize or just use the trace formula with correction.
    # 
    # Alternative: 
    # C4s = 1/2 * sum_{i<j} C(common_neighbors(i,j), 2)
    # Let M = A^2. M_{ij} = common neighbors.
    # Sum_{i<j} M_{ij}^2 - M_{ij} (since C(x,2) = x(x-1)/2, sum x(x-1)/2 = 0.5 * sum (x^2 - x))
    # Sum_{i<j} (M_{ij}^2 - M_{ij}) = 0.5 * (sum_{i != j} M_{ij}^2 - sum_{i != j} M_{ij})
    # Note sum_{i != j} M_{ij} = sum_i (sum_j M_{ij}) = sum_i deg(i)^2 ? No.
    # sum_j M_{ij} = deg(i). So sum_{i != j} M_{ij} = sum_i deg(i) = 2E.
    # Wait, M_{ii} = deg(i).
    # Sum_{i,j} M_{ij}^2 = trace(M^2) = trace(A^4).
    # Sum_{i,j} M_{ij} = sum_i deg(i) = 2E.
    # We want sum_{i<j} (M_{ij}^2 - M_{ij}).
    # Let S = sum_{i<j} (M_{ij}^2 - M_{ij}).
    # sum_{i,j, i!=j} (M_{ij}^2 - M_{ij}) = 2S.
    # sum_{i,j, i!=j} M_{ij}^2 = sum_{i,j} M_{ij}^2 - sum_i M_{ii}^2 = trace(A^4) - sum_i deg(i)^2.
    # sum_{i,j, i!=j} M_{ij} = 2E - sum_i M_{ii} = 2E - sum_i deg(i) = 2E - 2E = 0? No.
    # sum_{i,j} M_{ij} = sum_i deg(i) = 2E.
    # sum_{i} M_{ii} = sum_i deg(i) = 2E.
    # So sum_{i,j, i!=j} M_{ij} = 0? That's impossible.
    # Ah, M_{ij} is count of paths i-k-j.
    # If i=j, M_{ii} = deg(i).
    # Sum_{i,j} M_{ij} = sum_i sum_j M_{ij} = sum_i deg(i) = 2E.
    # Sum_{i} M_{ii} = sum_i deg(i) = 2E.
    # So sum_{i!=j} M_{ij} = 0? 
    # Let's check a triangle. 1-2, 2-3, 3-1.
    # M = A^2.
    # M_11 = 2 (1-2-1, 1-3-1).
    # M_12 = 1 (1-3-2).
    # M_13 = 1 (1-2-3).
    # Sum M = 2+1+1 + 1+2+1 + 1+1+2 = 12. 2E = 6. 12 != 6.
    # My formula sum_j M_{ij} = deg(i) is WRONG.
    # (A^2)_{ij} = sum_k A_{ik} A_{kj}.
    # Sum_j (A^2)_{ij} = sum_j sum_k A_{ik} A_{kj} = sum_k A_{ik} (sum_j A_{kj}) = sum_k A_{ik} deg(k).
    # This is not deg(i).
    
    # Let's stick to the O(N^3) or O(N^2 * deg) method for accuracy if NumPy is tricky, 
    # but N=31 is tiny. 
    # Let's optimize the Python loop for C4 count.
    
    def count_c4_python(adj):
        c4_count = 0
        # adj is list of sets
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2

    # Let's use a faster NumPy approach for the initial count and local search steps if possible.
    # For N=31, pure Python with sets is ~ 31*31*31 ops ~ 30k ops, which is fast.
    # 120s is plenty.
    
    # Current state
    current_adj = [set(adj[i]) for i in range(n)]
    current_c4 = count_c4_python(current_adj)
    
    # Precompute non-edges
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
            
    current_edge_set = set(edge_set)
    non_edge_list = list(all_pairs - current_edge_set)
    
    # Add one edge to reach 91
    # Find best edge to add
    best_edge = None
    min_cost = float('inf')
    
    # Cost of adding (i,j) is number of C4s created.
    # This is number of paths i-a-b-j.
    # a in N(i), b in N(j), a~b.
    for (i, j) in non_edge_list:
        cost = 0
        for a in current_adj[i]:
            for b in current_adj[j]:
                if a != b and b in current_adj[a]:
                    cost += 1
        if cost < min_cost:
            min_cost = cost
            best_edge = (i, j)
            
    if best_edge:
        i, j = best_edge
        current_edge_set.add((i, j))
        current_adj[i].add(j)
        current_adj[j].add(i)
        non_edge_list.remove((i, j))
        current_c4 += min_cost # Approximation? No, exact C4 count change is min_cost.
        # Wait, adding an edge creates exactly 'cost' new C4s.
        # So current_c4 = count_c4_python(current_adj) is safer.
        current_c4 = count_c4_python(current_adj)

    best_c4 = current_c4
    best_adj = [set(s) for s in current_adj]
    best_edge_set = set(current_edge_set)
    
    # Local Search: Best Improvement
    # Iterate:
    # 1. For each edge e, calculate "loss" if removed.
    #    Loss(e) = number of C4s containing e.
    # 2. For each non-edge f, calculate "gain" if added.
    #    Gain(f) = number of new C4s created by f.
    # 3. Find e* with max Loss and f* with min Gain.
    # 4. If Loss(e*) > Gain(f*), swap them.
    
    def get_edge_c4_contributions(adj):
        """Returns dict: edge -> count of C4s containing it."""
        contrib = defaultdict(int)
        # A C4 is v1-v2-v3-v4-v1.
        # Edges are (v1,v2), (v2,v3), (v3,v4), (v4,v1).
        # We can find C4s by looking at pairs of vertices (u,v) with common neighbors.
        # If u,v have common neighbors a,b, then u-a-v-b-u is a C4.
        # Edges involved: (u,a), (a,v), (v,b), (b,u).
        # Also u-b-v-a-u is the same C4.
        # So for each pair (u,v) with common neighbors C, 
        # for each pair (a,b) in C:
        #   C4: u-a-v-b-u
        #   Edges: (u,a), (a,v), (v,b), (b,u)
        
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    common_list = list(common)
                    for k in range(len(common_list)):
                        for l in range(k+1, len(common_list)):
                            a = common_list[k]
                            b = common_list[l]
                            # C4: u-a-v-b-u
                            # Edges:
                            e1 = (min(u,a), max(u,a))
                            e2 = (min(a,v), max(a,v))
                            e3 = (min(v,b), max(v,b))
                            e4 = (min(b,u), max(b,u))
                            contrib[e1] += 1
                            contrib[e2] += 1
                            contrib[e3] += 1
                            contrib[e4] += 1
        return contrib

    def get_non_edge_c4_costs(adj, non_edges):
        """Returns dict: non_edge -> count of new C4s if added."""
        costs = {}
        for (i, j) in non_edges:
            cost = 0
            # New C4s are i-a-b-j-i where a~i, b~j, a~b.
            # a in N(i), b in N(j), a!=b, a~b.
            for a in adj[i]:
                for b in adj[j]:
                    if a != b and b in adj[a]:
                        cost += 1
            costs[(i,j)] = cost
        return costs

    max_iterations = 1000
    no_improve = 0
    max_no_improve = 50
    
    while no_improve < max_no_improve and time.time() < deadline - 10:
        # 1. Calculate contributions of current edges
        edge_contribs = get_edge_c4_contributions(current_adj)
        # 2. Calculate costs of non-edges
        non_edge_costs = get_non_edge_c4_costs(current_adj, non_edge_list)
        
        # Find best edge to remove (max contribution)
        # If an edge is not in contribs, its contribution is 0.
        # We want to remove an edge that is in many C4s.
        # Note: Removing an edge (u,v) removes exactly contrib[(u,v)] C4s?
        # No. A C4 contains 4 edges. If we remove one, the C4 is broken.
        # So yes, removing edge e removes exactly contrib[e] C4s from the graph.
        
        max_contrib = -1
        best_remove = None
        for e in current_edge_set:
            c = edge_contribs.get(e, 0)
            if c > max_contrib:
                max_contrib = c
                best_remove = e
                
        # Find best non-edge to add (min cost)
        min_cost = float('inf')
        best_add = None
        for e in non_edge_list:
            c = non_edge_costs.get(e, 0)
            if c < min_cost:
                min_cost = c
                best_add = e
                
        if best_remove is None or best_add is None:
            break
            
        # Check if swap improves
        # Delta = -max_contrib + min_cost
        # We want Delta < 0 => min_cost < max_contrib
        if min_cost < max_contrib:
            # Perform swap
            u, v = best_remove
            x, y = best_add
            
            # Remove (u,v)
            current_edge_set.remove((u, v))
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            
            # Add (x,y)
            current_edge_set.add((x, y))
            current_adj[x].add(y)
            current_adj[y].add(x)
            
            # Update lists
            non_edge_list.remove((x, y))
            non_edge_list.append((u, v))
            
            # Recount C4s to be safe (or update incrementally)
            # Incremental: current_c4 -= max_contrib; current_c4 += min_cost
            # Is this exact?
            # Removing (u,v) removes all C4s containing (u,v).
            # Adding (x,y) creates all new C4s containing (x,y).
            # Are there interactions?
            # The new C4s created by (x,y) might use the edge (u,v) if it was present?
            # No, (u,v) is removed.
            # Could the new C4s created by (x,y) be the same as some C4s that were destroyed?
            # No, destroyed C4s contained (u,v). New C4s contain (x,y).
            # Since (u,v) != (x,y), they are distinct sets of edges.
            # So the change is exactly -max_contrib + min_cost.
            
            current_c4 -= max_contrib
            current_c4 += min_cost
            
            no_improve = 0
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = [set(s) for s in current_adj]
                best_edge_set = set(current_edge_set)
        else:
            no_improve += 1
            
    # Use best found
    if best_c4 < current_c4:
        final_adj = best_adj
        final_edge_set = best_edge_set
    else:
        final_adj = current_adj
        final_edge_set = current_edge_set

    edges = [[u, v] for (u, v) in final_edge_set]
    
    # Ensure exactly 91 distinct edges
    if len(edges) != 91:
        # Fallback fix
        final_edge_set = set(final_edge_set)
        edges = [[u, v] for (u, v) in final_edge_set]
        if len(edges) < 91:
            for i in range(n):
                for j in range(i+1, n):
                    if len(edges) >= 91: break
                    if (i, j) not in final_edge_set:
                        final_edge_set.add((i, j))
                        edges.append([i, j])
                if len(edges) >= 91: break
        if len(edges) > 91:
            edges = edges[:91]
            final_edge_set = set(tuple(sorted(e)) for e in edges)
            
    assert len(edges) == 91
    assert len(set(tuple(sorted(e)) for e in edges)) == 91
    for e in edges:
        assert 0 <= e[0] <= 30
        assert 0 <= e[1] <= 30
        assert e[0] != e[1]
        
    return edges

"""Construct a large (5,5)-free graph via Local Search on fixed N."""

import time
import random
import sys

def _find_k5(adj, n):
    """
    Find a K5 in the graph. Returns the set of 5 vertices or None.
    adj: list of sets of neighbors.
    """
    for v in range(n):
        # N(v) is the neighborhood of v
        # We need a K4 in N(v)
        nbrs_v = list(adj[v])
        m = len(nbrs_v)
        if m < 4:
            continue
        
        # Iterate over pairs in N(v) to find a K4?
        # Better: Iterate over u in N(v).
        # We need a K3 in N(u) intersect N(v).
        for u in nbrs_v:
            # T = N(u) intersect N(v)
            # We need a triangle in T.
            # Let's get the list of candidates for the triangle
            # Candidates are in N(u) and N(v)
            common = adj[u] & adj[v]
            # We need a triangle in 'common'
            # A triangle in 'common' is a set {x, y, z} such that 
            # x,y,z are in common and edges (x,y), (y,z), (z,x) exist.
            # Since x,y,z are in common, they are neighbors of u and v.
            # So we just need to check if there is a triangle in the induced subgraph on 'common'.
            
            # Efficient triangle finding in 'common':
            # For each x in common:
            #   For each y in common with y > x:
            #     If (x,y) is an edge:
            #       Check if there is a z in common such that (z,x) and (z,y) are edges.
            #       z must be in adj[x] & adj[y] & common.
            
            common_list = list(common)
            L = len(common_list)
            if L < 3:
                continue
            
            for i in range(L):
                x = common_list[i]
                for j in range(i + 1, L):
                    y = common_list[j]
                    if y in adj[x]:
                        # Check for z
                        # z must be in adj[x] & adj[y] & common
                        # and z != x, z != y
                        candidates_z = adj[x] & adj[y] & common
                        # Remove x and y from candidates? 
                        # adj[x] does not contain x. adj[y] does not contain y.
                        # So candidates_z does not contain x or y.
                        if candidates_z:
                            # Found a triangle x, y, z
                            z = next(iter(candidates_z))
                            return {v, u, x, y, z}
    return None

def _find_ind5(adj, n):
    """
    Find an independent set of size 5.
    This is equivalent to finding a K5 in the complement graph.
    """
    # Construct complement adjacency on the fly? 
    # Or just check non-edges.
    # Ind5 {v, u, x, y, z} means:
    # v is non-adjacent to u, x, y, z.
    # u is non-adjacent to x, y, z.
    # x is non-adjacent to y, z.
    # y is non-adjacent to z.
    
    # Let's use the same logic as _find_k5 but on non-edges.
    # Non-neighborhood of v: N_bar(v) = {w | w != v and w not in adj[v]}
    
    for v in range(n):
        # Get non-neighbors
        nbrs_v = adj[v]
        non_nbrs_v = set(range(n)) - nbrs_v - {v}
        m = len(non_nbrs_v)
        if m < 4:
            continue
        
        for u in non_nbrs_v:
            # u is non-adjacent to v.
            # We need x, y, z in non_nbrs_v \ {u}
            # such that u is non-adjacent to x, y, z.
            # AND x, y, z form an independent triangle (i.e., K3 in complement).
            
            # Candidates for x,y,z must be in non_nbrs_v AND non_neighbors(u)
            non_nbrs_u = set(range(n)) - adj[u] - {u}
            common = non_nbrs_v & non_nbrs_u
            # common contains v? 
            # v is in non_nbrs_v. 
            # Is v in non_nbrs_u? Yes, because u is non-adjacent to v.
            # So v is in common. But we already picked v and u.
            # We need x, y, z distinct from v, u.
            common.discard(v)
            common.discard(u)
            
            if len(common) < 3:
                continue
                
            # Now find an independent triangle in 'common'.
            # This means x,y,z in common such that:
            # x not in adj[y], y not in adj[z], z not in adj[x].
            
            common_list = list(common)
            L = len(common_list)
            
            for i in range(L):
                x = common_list[i]
                for j in range(i + 1, L):
                    y = common_list[j]
                    # Check if x-y is a non-edge
                    if y not in adj[x]:
                        # Check for z
                        # z must be in common
                        # z not in adj[x]
                        # z not in adj[y]
                        candidates_z = common
                        # Remove x, y
                        candidates_z.discard(x)
                        candidates_z.discard(y)
                        # Filter for non-adjacency to x and y
                        valid_z = set()
                        for z in candidates_z:
                            if z not in adj[x] and z not in adj[y]:
                                valid_z.add(z)
                        
                        if valid_z:
                            z = next(iter(valid_z))
                            return {v, u, x, y, z}
    return None

def _local_search(n, max_time=200):
    """
    Perform local search to find a (5,5)-free graph on n vertices.
    Returns (n, edges) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initialize random graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
    
    # Local search loop
    while time.time() < deadline:
        # Find a violation
        k5 = _find_k5(adj, n)
        if k5:
            # Break a random edge in the K5
            verts = list(k5)
            # Pick a random pair
            i, j = random.sample(verts, 2)
            # Flip edge (i, j)
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
            continue
        
        ind5 = _find_ind5(adj, n)
        if ind5:
            # Break a random non-edge in the Ind5 (i.e., add an edge)
            verts = list(ind5)
            i, j = random.sample(verts, 2)
            # Flip edge (i, j)
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
            continue
        
        # No violations found
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    return None

def construct():
    deadline = time.time() + 280 # Safety margin
    
    # Try to find a valid graph for N=42, then 41, etc.
    # 42 is the known lower bound.
    for n in range(42, 37, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this n
        # If we find one, we return it.
        # We don't need to exhaust all restarts if we find one quickly.
        # But local search might get stuck.
        
        # Give each n a budget
        # 42: 100s, 41: 50s, 40: 30s, 39: 20s, 38: 10s
        budget = 100 if n == 42 else (50 if n == 41 else 30)
        
        result = _local_search(n, max_time=budget)
        if result:
            return result
            
    # Fallback: Return the Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges.")

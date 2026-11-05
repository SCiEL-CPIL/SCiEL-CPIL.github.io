import random
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Initialize paths as single edges
        paths = []
        edge_set = set(edges)
        for u, v in edges:
            paths.append([u, v])
        
        # Graph adjacency for quick edge lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        changed = True
        while changed:
            changed = False
            n_paths = len(paths)
            for i in range(n_paths):
                if changed:
                    break # Restart loop if a merge occurred
                for j in range(i + 1, n_paths):
                    pi = paths[i]
                    pj = paths[j]
                    
                    set_i = set(pi)
                    set_j = set(pj)
                    intersection = set_i & set_j
                    
                    # Case 1: Vertex Disjoint
                    if not intersection:
                        # Check if endpoints are connected by an edge
                        # Endpoints of pi: pi[0], pi[-1]
                        # Endpoints of pj: pj[0], pj[-1]
                        
                        # Try pi + pj
                        if pi[-1] in adj and pj[0] in adj[pi[-1]]:
                            # Check if edge (pi[-1], pj[0]) is not already used?
                            # It's in the graph. Since paths are edge-disjoint, and they are vertex disjoint,
                            # this edge is not in pi or pj. So it's available.
                            # New path: pi + pj
                            new_path = pi + pj
                            # Verify simplicity (should be guaranteed by vertex disjointness)
                            if len(new_path) == len(set(new_path)):
                                paths[i] = new_path
                                paths.pop(j)
                                changed = True
                                break
                        
                        # Try pi + reversed(pj)
                        if pi[-1] in adj and pj[-1] in adj[pi[-1]]:
                            new_path = pi + pj[::-1]
                            if len(new_path) == len(set(new_path)):
                                paths[i] = new_path
                                paths.pop(j)
                                changed = True
                                break

                        # Try reversed(pi) + pj
                        if pi[0] in adj and pj[0] in adj[pi[0]]:
                            new_path = pi[::-1] + pj
                            if len(new_path) == len(set(new_path)):
                                paths[i] = new_path
                                paths.pop(j)
                                changed = True
                                break

                        # Try reversed(pi) + reversed(pj)
                        if pi[0] in adj and pj[-1] in adj[pi[0]]:
                            new_path = pi[::-1] + pj[::-1]
                            if len(new_path) == len(set(new_path)):
                                paths[i] = new_path
                                paths.pop(j)
                                changed = True
                                break

                    # Case 2: Share exactly one vertex
                    elif len(intersection) == 1:
                        v = intersection.pop()
                        
                        # v must be an endpoint of both paths
                        is_endpoint_i = (v == pi[0] or v == pi[-1])
                        is_endpoint_j = (v == pj[0] or v == pj[-1])
                        
                        if is_endpoint_i and is_endpoint_j:
                            # We can merge them at v
                            # We need to orient them such that v is at the junction
                            
                            # Orient pi so v is at the end
                            if pi[0] == v:
                                pi_oriented = pi[::-1]
                            else:
                                pi_oriented = pi
                            
                            # Orient pj so v is at the start
                            if pj[-1] == v:
                                pj_oriented = pj[::-1]
                            else:
                                pj_oriented = pj
                                
                            # Now pi_oriented ends with v, pj_oriented starts with v
                            # Merge: pi_oriented + pj_oriented[1:]
                            new_path = pi_oriented + pj_oriented[1:]
                            
                            # Verify simplicity
                            if len(new_path) == len(set(new_path)):
                                paths[i] = new_path
                                paths.pop(j)
                                changed = True
                                break

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results

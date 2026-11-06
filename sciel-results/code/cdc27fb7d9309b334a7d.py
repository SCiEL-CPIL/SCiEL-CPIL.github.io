import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def creates_q3(u, v, adj_masks):
        # Check if adding edge (u, v) creates a Q3 subgraph.
        # A Q3 containing (u, v) must have u and v in opposite partitions.
        # Let u be in A, v in B.
        # u is connected to v, b1, b2 in the Q3.
        # v is connected to u, a1, a2 in the Q3.
        # The other vertices are a3, b3.
        # Structure:
        # A = {u, a1, a2, a3}
        # B = {v, b1, b2, b3}
        # Edges:
        # (u, v), (u, b1), (u, b2)
        # (v, a1), (v, a2)
        # (a1, b1), (a1, b3)
        # (a2, b2), (a2, b3)
        # (b1, a3), (b2, a3)
        # (a3, b3)
        
        # Candidates for b1, b2 are neighbors of u (excluding v)
        # Candidates for a1, a2 are neighbors of v (excluding u)
        
        # Get neighbors of u and v
        # adj_masks is a list of bitmasks
        nu_mask = adj_masks[u] & ~(1 << v)
        nv_mask = adj_masks[v] & ~(1 << u)
        
        # Extract neighbor lists
        nu = [i for i in range(n) if (nu_mask >> i) & 1]
        nv = [i for i in range(n) if (nv_mask >> i) & 1]
        
        # If u or v doesn't have at least 2 other neighbors, no Q3 can be formed
        if len(nu) < 2 or len(nv) < 2:
            return False
            
        # Iterate over pairs (b1, b2) from nu
        for i in range(len(nu)):
            b1 = nu[i]
            for j in range(i + 1, len(nu)):
                b2 = nu[j]
                
                # Iterate over pairs (a1, a2) from nv
                for k in range(len(nv)):
                    a1 = nv[k]
                    for l in range(k + 1, len(nv)):
                        a2 = nv[l]
                        
                        # Vertices used so far: u, v, b1, b2, a1, a2
                        used = {u, v, b1, b2, a1, a2}
                        
                        # Check edges (a1, b1) and (a2, b2)
                        # If these don't exist, this combination is invalid for this specific Q3 structure mapping
                        # Note: The mapping of a1/a2 to b1/b2 is fixed by the structure?
                        # In Q3, if u=000, v=001, b1=010, b2=100, a1=011, a2=101.
                        # a1(011) is connected to b1(010). Yes.
                        # a2(101) is connected to b2(100). Yes.
                        # So we must have (a1, b1) and (a2, b2) in the graph.
                        
                        if not ((adj_masks[a1] >> b1) & 1):
                            continue
                        if not ((adj_masks[a2] >> b2) & 1):
                            continue
                            
                        # Now we need to find a3 and b3.
                        # a3 must be connected to b1 and b2.
                        # b3 must be connected to a1 and a2.
                        # a3 must be connected to b3.
                        # a3, b3 must be distinct from used vertices.
                        
                        # Candidates for a3: neighbors of b1 and b2, not in used
                        # Candidates for b3: neighbors of a1 and a2, not in used
                        
                        # Let's find candidates for a3
                        a3_candidates = []
                        for w in range(n):
                            if w in used:
                                continue
                            if (adj_masks[b1] >> w) & 1 and (adj_masks[b2] >> w) & 1:
                                a3_candidates.append(w)
                                
                        if not a3_candidates:
                            continue
                            
                        # For each a3 candidate, find b3
                        for a3 in a3_candidates:
                            # b3 must be neighbor of a1 and a2, and a3
                            # And not in used
                            b3_found = False
                            for w in range(n):
                                if w in used or w == a3:
                                    continue
                                if ((adj_masks[a1] >> w) & 1) and \
                                   ((adj_masks[a2] >> w) & 1) and \
                                   ((adj_masks[a3] >> w) & 1):
                                    # Found b3
                                    b3 = w
                                    b3_found = True
                                    break
                            if b3_found:
                                return True
        return False

    best_edges = set()
    best_count = 0
    
    # Restart loop
    for restart in range(2000):
        if time.time() > deadline - 2:
            break
            
        adj_masks = [0] * n
        current_edges = set()
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 1:
                break
            if not creates_q3(u, v, adj_masks):
                # Add edge
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                current_edges.add((u, v))
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
        # Optional: Local improvement (swap edges)
        # But greedy random restarts are often sufficient for small n.
        # Given time constraints, we rely on restarts.
        
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")

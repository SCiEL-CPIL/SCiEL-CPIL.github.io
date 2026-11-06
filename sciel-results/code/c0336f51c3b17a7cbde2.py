import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    
    # Bitsets for adjacency
    # adj[u] is an int where bit v is 1 if (u,v) is an edge
    
    def has_q3_with_edge(adj, u, v):
        # Check if adding edge (u,v) creates a Q3 containing (u,v).
        # We assume (u,v) is already in adj when calling this? 
        # Better: pass adj without (u,v) and check if it *would* create one?
        # Or pass adj with (u,v) and check if it contains a Q3.
        # The previous code added then checked. Let's keep that but optimize the check.
        # Actually, it's easier to check if the graph *contains* a Q3.
        # But checking only Q3s containing (u,v) is faster if we know (u,v) is the "last" edge added?
        # No, adding (u,v) might complete a Q3 that didn't exist before.
        # Any Q3 in the new graph that wasn't in the old graph must contain the new edge (u,v).
        # So we only need to check Q3s containing (u,v).
        
        # u, v are the endpoints of the new edge.
        # In Q3, u and v are adjacent.
        # N(u) in Q3: {v, w1, w2}
        # N(v) in Q3: {u, z1, z2}
        
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # Need at least 2 other neighbors for u and v
        if (nu.bit_count() < 2) or (nv.bit_count() < 2):
            return False
            
        # Get lists of neighbors
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        # Iterate pairs w1, w2 from nu_list
        for i in range(len(nu_list)):
            w1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                w2 = nu_list[j]
                
                # Find y1 in N(w1) ∩ N(w2) \ {u}
                # y1 must be connected to w1 and w2.
                inter_w1_w2 = adj[w1] & adj[w2] & ~(1 << u)
                if inter_w1_w2 == 0:
                    continue
                # In Q3, y1 is unique for this configuration?
                # w1 and w2 are distance 2 from u. 
                # In Q3, if u=000, w1=010, w2=100, then y1=110 is the unique common neighbor not u.
                # If there are multiple, we check each.
                y1_candidates = [k for k in range(n) if (inter_w1_w2 >> k) & 1]
                
                for y1 in y1_candidates:
                    # Iterate pairs z1, z2 from nv_list
                    for k in range(len(nv_list)):
                        z1 = nv_list[k]
                        for l in range(k+1, len(nv_list)):
                            z2 = nv_list[l]
                            
                            # Find y2 in N(z1) ∩ N(z2) \ {v}
                            inter_z1_z2 = adj[z1] & adj[z2] & ~(1 << v)
                            if inter_z1_z2 == 0:
                                continue
                            
                            y2_candidates = [m for m in range(n) if (inter_z1_z2 >> m) & 1]
                            
                            for y2 in y2_candidates:
                                # Check distinctness
                                # Vertices: u, v, w1, w2, z1, z2, y1, y2
                                # All must be distinct.
                                if y1 == y2: continue
                                if y1 in (u, v, w1, w2, z1, z2): continue
                                if y2 in (u, v, w1, w2, z1, z2): continue
                                if w1 == w2 or z1 == z2: continue # handled by loops
                                if w1 == z1 or w1 == z2 or w2 == z1 or w2 == z2: continue
                                # u, v are distinct from others by construction (nu/nv exclude v/u)
                                
                                # Check remaining edges:
                                # 1. y1-y2
                                if not ((adj[y1] >> y2) & 1):
                                    continue
                                    
                                # 2. w1-z1 and w2-z2 OR w1-z2 and w2-z1
                                # In Q3:
                                # w1 (010) connects to z1 (011).
                                # w2 (100) connects to z2 (101).
                                # So we need (w1, z1) and (w2, z2) OR (w1, z2) and (w2, z1).
                                # Note: z1, z2 are neighbors of v. w1, w2 are neighbors of u.
                                
                                if ((adj[w1] >> z1) & 1) and ((adj[w2] >> z2) & 1):
                                    return True
                                if ((adj[w1] >> z2) & 1) and ((adj[w2] >> z1) & 1):
                                    return True
                                    
        return False

    def is_q3_free(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check all edges if they form a Q3
        for u, v in edge_list:
            if has_q3_with_edge(adj, u, v):
                return False
        return True

    def get_weight2_graph_edges():
        # Vertices 0..15 are 4-bit numbers.
        # Edge if XOR has weight 2.
        edges = []
        for i in range(16):
            for j in range(i+1, 16):
                x = i ^ j
                if bin(x).count('1') == 2:
                    edges.append((i, j))
        return edges

    # Try the weight-2 graph (Shrikhande candidate)
    w2_edges = get_weight2_graph_edges()
    if len(w2_edges) > best_count and is_q3_free(w2_edges):
        best_count = len(w2_edges)
        best_edges = w2_edges

    def run_greedy(seed):
        random.seed(seed)
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = [0] * n
        edge_set = set()
        
        count = 0
        for u, v in edges:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(adj, u, v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edge_set.add((u, v))
                count += 1
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize order of local search to escape local optima
            local_edges = all_edges[:]
            random.shuffle(local_edges)
            for u, v in local_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                # Try add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return count, list(edge_set)

    # Run multiple restarts
    restart = 0
    while time.time() < deadline:
        count, edges_list = run_greedy(restart)
        if count > best_count:
            best_count = count
            best_edges = edges_list
        
        restart += 1

        if best_count >= 55:
             # Could break or continue, but 55 is high. 
             # Let's just keep going to see if we can get higher.
             pass

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())

import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for 120s limit
    
    # We want to maximize n. The known record is 29.
    # We will try to construct a graph of size 29.
    # If we fail, we try 28, 27, etc., but we expect 29 to be reachable.
    
    # Helper to check if adding a new vertex with given adjacencies creates a K2,2,2
    # adj_color0: bitmask of vertices connected to new vertex via color 0
    # adj_color1: bitmask of vertices connected to new vertex via color 1
    # adj0_list: list of bitmasks for color 0 adjacencies of existing vertices
    # adj1_list: list of bitmasks for color 1 adjacencies of existing vertices
    
    def check_validity(new_adj0, new_adj1, adj0_list, adj1_list, n_old):
        # Check if any pair of existing vertices (u, v) forms a K2,2,2 with the new vertex?
        # No, K2,2,2 requires 6 vertices. The new vertex is one part of the 2-sets.
        # Actually, the new vertex can be part of the 2-set in one partition or the other.
        # A K2,2,2 in color c is formed by vertices {a,b} in one part and {c,d,e,f}... no, K2,2,2 has 6 vertices: two sets of 3? No, K_{2,2,2} is the complete tripartite graph with parts of size 2. It has 6 vertices.
        # Wait, K_{2,2,2} is the octahedron graph. It has 6 vertices.
        # It is the complement of 3 disjoint edges.
        # In a 2-coloring, a monochromatic K_{2,2,2} in color 0 means there are 6 vertices such that the edges between the 3 pairs are all color 0?
        # No. K_{2,2,2} is a graph on 6 vertices. It consists of 3 pairs of vertices. Every vertex is connected to all vertices not in its own pair.
        # So, if we have 6 vertices $v_1..v_6$ partitioned into $\{v_1,v_2\}, \{v_3,v_4\}, \{v_5,v_6\}$, a monochromatic K_{2,2,2} in color 0 exists if all edges between different pairs are color 0.
        # This is equivalent to saying that the subgraph induced by these 6 vertices in color 0 is the complete tripartite graph $K_{2,2,2}$.
        
        # How to check efficiently?
        # When adding a new vertex $x$, any new $K_{2,2,2}$ must include $x$.
        # Let the $K_{2,2,2}$ be in color $c$.
        # The 6 vertices are partitioned into 3 pairs. $x$ is in one of these pairs, say pair $P_1$.
        # Let $y$ be the other vertex in $P_1$.
        # Then $x$ and $y$ are NOT connected by color $c$ (since they are in the same part of the tripartite graph, there is no edge between them in $K_{2,2,2}$).
        # Wait, in $K_{2,2,2}$, there are no edges within the parts. So if $x$ and $y$ are in the same part, the edge $(x,y)$ is NOT color $c$.
        # The other 4 vertices form two pairs $P_2=\{u,v\}$ and $P_3=\{w,z\}$.
        # All edges between $P_1, P_2, P_3$ must be color $c$.
        # So $x$ must be connected to $u,v,w,z$ by color $c$.
        # $y$ must be connected to $u,v,w,z$ by color $c$.
        # And the edges between $P_2$ and $P_3$ must be color $c$. i.e., $(u,w), (u,z), (v,w), (v,z)$ are color $c$.
        # Also, edges within $P_2$ ($u,v$) and within $P_3$ ($w,z$) are NOT color $c$.
        
        # So, for a fixed color $c$:
        # 1. Choose a vertex $y$ among existing vertices such that edge $(x,y)$ is NOT color $c$.
        # 2. Let $N_c(x)$ be the set of vertices connected to $x$ by color $c$.
        # 3. Let $N_c(y)$ be the set of vertices connected to $y$ by color $c$.
        # 4. We need to find 4 distinct vertices $u,v,w,z$ in $N_c(x) \cap N_c(y)$ such that:
        #    a. $u,v$ are not connected by color $c$ (edge is $1-c$).
        #    b. $w,z$ are not connected by color $c$ (edge is $1-c$).
        #    c. All edges between $\{u,v\}$ and $\{w,z\}$ are color $c$.
        
        # This is still complex. Let's simplify the check.
        # A faster check:
        # Iterate over all pairs of existing vertices $(y, z)$? No, that's $O(n^2)$.
        # With $n \approx 29$, $n^2 \approx 800$. This is fast.
        
        # Let's refine the condition for a new K2,2,2 involving x in color c.
        # The 6 vertices are $x, y$ (pair 1), $u, v$ (pair 2), $w, z$ (pair 3).
        # Conditions:
        # 1. $(x,y)$ is not color c.
        # 2. $x$ is connected to $u,v,w,z$ by color c.
        # 3. $y$ is connected to $u,v,w,z$ by color c.
        # 4. $u,v$ not connected by color c.
        # 5. $w,z$ not connected by color c.
        # 6. $u$ connected to $w,z$ by color c.
        # 7. $v$ connected to $w,z$ by color c.
        
        # We can iterate over possible $y$ (not connected to x by c).
        # For each $y$, consider the set $S = N_c(x) \cap N_c(y)$.
        # We need to find two disjoint non-edges in color c within $S$, such that all cross-edges are color c.
        # Actually, if we find two vertices $u, v$ in $S$ that are NOT connected by color c, and two vertices $w, z$ in $S$ that are NOT connected by color c, and all cross edges are color c, we have a violation.
        
        # To make it faster, we can precompute for each pair of vertices $(a,b)$ whether they are connected by color 0 or 1.
        # Let's use bitmasks.
        # adj0[i] is bitmask of vertices connected to i by color 0.
        # adj1[i] is bitmask of vertices connected to i by color 1.
        
        # For the new vertex x (index n_old):
        # We define its adj0 and adj1 bitmasks (only bits 0..n_old-1 set).
        
        # Check for color 0 violation:
        # Iterate y in 0..n_old-1.
        # If bit y in adj0[x] is 0 (i.e., x-y is color 1):
        #    S = adj0[x] & adj0[y]
        #    We need to find u,v,w,z in S such that:
        #    - (u,v) is color 1 (not in adj0)
        #    - (w,z) is color 1 (not in adj0)
        #    - cross edges (u,w), (u,z), (v,w), (v,z) are color 0 (in adj0)
        
        # This sub-problem: find two disjoint non-edges in S with complete bipartite connections in adj0.
        # Since |S| is small (max 28), we can just check all pairs in S.
        
        return True # Placeholder, logic below in main loop

    def is_valid_coloring(n, colors):
        # colors is a list of 0s and 1s of length n*(n-1)//2
        # Reconstruct adj lists
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        # Check for K2,2,2 in color 0
        # Iterate over all 6-subsets? No, too slow.
        # Use the incremental check logic or direct check.
        # Direct check for K2,2,2 in color 0:
        # A K2,2,2 in color 0 is a set of 6 vertices {a,b,c,d,e,f} partitioned into 3 pairs 
        # such that all edges between pairs are color 0.
        # This is equivalent to: The subgraph induced by these 6 vertices in color 0 is K_{2,2,2}.
        # K_{2,2,2} is the complement of 3 disjoint edges.
        # So in color 0, the degrees of these 6 vertices within the subset must be 4.
        # And the non-edges must form a perfect matching.
        
        # Efficient check:
        # For each pair of vertices (u,v) that are NOT connected by color 0 (i.e., edge is color 1):
        # Let S = N_0(u) \cap N_0(v). (Neighbors of u and v in color 0, excluding u,v themselves since they are not connected).
        # Actually, u and v are not in each other's N_0.
        # We need to find two other pairs (w,x) and (y,z) in S such that:
        # 1. w,x not connected by color 0.
        # 2. y,z not connected by color 0.
        # 3. All cross edges between {w,x} and {y,z} are color 0.
        # 4. Also, the edges between {u,v} and {w,x,y,z} must be color 0 (satisfied by S definition).
        
        # Wait, if u,v are a pair in the K2,2,2, they are not connected by color 0.
        # The other 4 vertices are in S = N_0(u) \cap N_0(v).
        # We need to partition 4 elements of S into two pairs {w,x} and {y,z} such that:
        # - w,x not connected by color 0.
        # - y,z not connected by color 0.
        # - w connected to y,z by color 0.
        # - x connected to y,z by color 0.
        
        # Iterate u < v:
        # If edge (u,v) is color 1:
        #    S = (adj0[u] & adj0[v])
        #    Remove u,v from S if present (they shouldn't be).
        #    List vertices in S.
        #    If len < 4, continue.
        #    Iterate over all combinations of 4 vertices from S: w,x,y,z.
        #    Check if (w,x) is color 1 and (y,z) is color 1 and cross edges are color 0.
        #    Also check permutations of pairing: {w,y},{x,z} and {w,z},{x,y}.
        
        # This is $O(n^2 \times \binom{n}{4}) \approx 30^2 \times 27000 \approx 2.4M$ operations. Very fast.
        
        for u in range(n):
            for v in range(u+1, n):
                # Check if (u,v) is color 1
                if adj0[u] & (1 << v):
                    continue # Edge is color 0, so u,v cannot be a "non-edge pair" in color 0 K2,2,2
                # Edge is color 1.
                S_mask = adj0[u] & adj0[v]
                # Get list of vertices in S_mask
                S = []
                m = S_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    S.append(idx)
                    m ^= lsb
                if len(S) < 4:
                    continue
                # Check all 4-subsets of S
                for combo in itertools.combinations(S, 4):
                    a, b, c, d = combo
                    # Check pairing 1: (a,b) and (c,d)
                    # (a,b) must be color 1 (not in adj0)
                    if not (adj0[a] & (1 << b)) and not (adj0[c] & (1 << d)):
                        # Check cross edges
                        if (adj0[a] & (1 << c)) and (adj0[a] & (1 << d)) and \
                           (adj0[b] & (1 << c)) and (adj0[b] & (1 << d)):
                            return False
                    # Check pairing 2: (a,c) and (b,d)
                    if not (adj0[a] & (1 << c)) and not (adj0[b] & (1 << d)):
                        if (adj0[a] & (1 << b)) and (adj0[a] & (1 << d)) and \
                           (adj0[c] & (1 << b)) and (adj0[c] & (1 << d)):
                            return False
                    # Check pairing 3: (a,d) and (b,c)
                    if not (adj0[a] & (1 << d)) and not (adj0[b] & (1 << c)):
                        if (adj0[a] & (1 << b)) and (adj0[a] & (1 << c)) and \
                           (adj0[d] & (1 << b)) and (adj0[d] & (1 << c)):
                            return False
        
        # Check for K2,2,2 in color 1
        # Symmetric check
        for u in range(n):
            for v in range(u+1, n):
                # Check if (u,v) is color 0
                if adj1[u] & (1 << v):
                    continue
                S_mask = adj1[u] & adj1[v]
                S = []
                m = S_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    S.append(idx)
                    m ^= lsb
                if len(S) < 4:
                    continue
                for combo in itertools.combinations(S, 4):
                    a, b, c, d = combo
                    if not (adj1[a] & (1 << b)) and not (adj1[c] & (1 << d)):
                        if (adj1[a] & (1 << c)) and (adj1[a] & (1 << d)) and \
                           (adj1[b] & (1 << c)) and (adj1[b] & (1 << d)):
                            return False
                    if not (adj1[a] & (1 << c)) and not (adj1[b] & (1 << d)):
                        if (adj1[a] & (1 << b)) and (adj1[a] & (1 << d)) and \
                           (adj1[c] & (1 << b)) and (adj1[c] & (1 << d)):
                            return False
                    if not (adj1[a] & (1 << d)) and not (adj1[b] & (1 << c)):
                        if (adj1[a] & (1 << b)) and (adj1[a] & (1 << c)) and \
                           (adj1[d] & (1 << b)) and (adj1[d] & (1 << c)):
                            return False
        return True

    # Greedy construction
    def try_construct(target_n, seed):
        random.seed(seed)
        n = 0
        colors = []
        
        # Start with a small valid graph, e.g., K1 or K2
        # Let's start with K1
        # Actually, let's start with K4 or K5 to have some structure?
        # Or just build from scratch.
        
        # We will maintain adj0 and adj1 bitmasks for efficiency during construction.
        # But we need to return the full color list at the end.
        
        # To speed up, we can use the is_valid_coloring check on the full list?
        # No, that's $O(n^2 \binom{n}{4})$. For n=29, it's fast enough to do at each step?
        # 29 steps * 2.4M ops = 70M ops. In Python, this might be ~1-2 seconds.
        # So we can just use the full check after each addition.
        
        # Let's try building from n=1 to target_n.
        
        # Initialize with n=1
        # colors is empty for n=1.
        
        # We will add vertices one by one.
        # For each new vertex, we try random colorings for its edges to previous vertices.
        
        # Optimization: Use symmetry? No, just random.
        
        for new_n in range(2, target_n + 1):
            success = False
            attempts = 0
            while not success and attempts < 500:
                attempts += 1
                if time.time() > deadline:
                    break
                
                # Generate random colors for edges between new vertex (index new_n-1) and 0..new_n-2
                new_colors = [random.randint(0, 1) for _ in range(new_n - 1)]
                
                # Construct full color list
                full_colors = colors + new_colors
                
                # Check validity
                if is_valid_coloring(new_n, full_colors):
                    colors = full_colors
                    success = True
            
            if not success:
                return None
        return {'n': target_n, 'colors': colors}

    # Try to construct 29
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(29, seed)
        if res:
            return res
            
    # If 29 fails, try 28
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(28, seed)
        if res:
            return res

    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

# EVOLVE-BLOCK-END

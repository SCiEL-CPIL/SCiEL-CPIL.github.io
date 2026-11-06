        def greedy_vertex_init(n):
            colors = [0] * (n * (n - 1) // 2)
            # Vertices added 0 to n-1
            # When adding vertex k, color edges (k, i) for i in 0..k-1
            # We process i in random order? Or 0..k-1?
            # Let's try 0..k-1.
            for k in range(1, n):
                for i in range(k):
                    best_c = 0
                    best_score = float('inf')
                    for c in range(4):
                        score = 0
                        for j in range(i):
                            # Triangle (k, i, j)
                            # Edges: (k,i)=c, (k,j)=colors[edge_idx(k,j)], (i,j)=colors[edge_idx(i,j)]
                            c_kj = colors[edge_idx(k, j)]
                            c_ij = colors[edge_idx(i, j)]
                            if c == c_kj and c == c_ij:
                                score += 1
                        if score < best_score:
                            best_score = score
                            best_c = c
                    colors[edge_idx(k, i)] = best_c
            return colors

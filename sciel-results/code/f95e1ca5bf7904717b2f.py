        def get_color_bound(P):
            # Returns a list of vertices sorted by color descending, and the max color index
            # Greedy coloring:
            uncolored = set(P)
            color_classes = []
            while uncolored:
                # Pick a vertex with max degree in uncolored
                v = max(uncolored, key=lambda x: len(adj[x] & uncolored))
                cls = [v]
                # Remove v and neighbors from uncolored
                nbrs = adj[v]
                uncolored -= nbrs
                uncolored.discard(v)
                
                # Try to add more vertices to this class
                # A vertex u is compatible if it is not adjacent to any vertex in cls
                # We can iterate through uncolored and check
                # To be efficient, we can just do one pass.
                # Since n is small, O(n^2) is fine.
                
                # Actually, the standard greedy coloring just assigns one vertex per color 
                # if we do it sequentially. 
                # But "Maximal Independent Set" per color is better.
                
                # Let's just use the number of vertices in P as bound for now 
                # and hope the structure helps.
                # Or use a simple heuristic: 
                # Bound = len(P)
                
                color_classes.append(cls)
                
            # Flatten
            verts = []
            for i, cls in enumerate(color_classes):
                for v in cls:
                    verts.append((v, i))
                    
            verts.sort(key=lambda x: x[1], reverse=True)
            return [v for v, c in verts], len(color_classes) - 1

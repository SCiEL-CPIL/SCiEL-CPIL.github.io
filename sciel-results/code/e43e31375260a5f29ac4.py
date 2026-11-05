    def bron_kerbosch(adj_bit, vertices, time_limit, start_time):
        if not vertices:
            return []
        
        max_clique = []
        
        def bk(R, P, X):
            if time.time() - start_time > time_limit:
                return
            
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = list(R)
                return
            
            P_list = list(P)
            X_list = list(X)
            union = P_list + X_list
            if not union:
                return
            
            mask_P = 0
            for v in P_list:
                mask_P |= (1 << v)
            
            best_u = None
            max_inter = -1
            for u in union:
                inter = bin(adj_bit[u] & mask_P).count('1')
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            if best_u is None:
                return
                
            candidates = [v for v in P if not (adj_bit[best_u] & (1 << v))]
            
            for v in candidates:
                R.add(v)
                P_new = set()
                for w in P:
                    if adj_bit[v] & (1 << w):
                        P_new.add(w)
                X_new = set()
                for w in X:
                    if adj_bit[v] & (1 << w):
                        X_new.add(w)
                
                bk(R, P_new, X_new)
                
                P.remove(v)
                X.add(v)
                R.remove(v)
                
        R = set()
        P = set(vertices)
        X = set()
        bk(R, P, X)
        return max_clique

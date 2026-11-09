import itertools

def construct():
    # Vertices are tuples (i, j, k) where i in 0..3, j in 0..3, k in 0..1
    # Map to 0..31
    # Edges in C4: (0,1), (1,2), (2,3), (3,0)
    # Edges in C2: (0,1)
    
    n = 32
    edges = []
    
    def idx(i, j, k):
        return (i * 4 + j) * 2 + k
        
    def adj_c4(a, b):
        return abs(a - b) == 1 or (a == 0 and b == 3) or (a == 3 and b == 0)
        
    def adj_c2(a, b):
        return a != b
        
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = idx(i, j, k)
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(2):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            v = idx(i2, j2, k2)
                            
                            # Strong product adjacency
                            # 1. i=i2, j=j2, k~k2
                            # 2. i=i2, k=k2, j~j2
                            # 3. j=j2, k=k2, i~i2
                            # 4. j=j2, i~i2, k~k2
                            # 5. k=k2, i~i2, j~j2
                            # 6. i=i2, j~j2, k~k2
                            # 7. i~i2, j~j2, k~k2
                            
                            is_adj = False
                            if i == i2 and j == j2 and adj_c2(k, k2):
                                is_adj = True
                            elif i == i2 and k == k2 and adj_c4(j, j2):
                                is_adj = True
                            elif j == j2 and k == k2 and adj_c4(i, i2):
                                is_adj = True
                            elif j == j2 and adj_c4(i, i2) and adj_c2(k, k2):
                                is_adj = True
                            elif k == k2 and adj_c4(i, i2) and adj_c4(j, j2):
                                is_adj = True
                            elif i == i2 and adj_c4(j, j2) and adj_c2(k, k2):
                                is_adj = True
                            elif adj_c4(i, i2) and adj_c4(j, j2) and adj_c2(k, k2):
                                is_adj = True
                                
                            if is_adj and u < v:
                                edges.append([u, v])
                                
    return {'n': n, 'edges': edges}

    def construct():
        edges = []
        # A = {0, 1, 2}, B = {3, ..., 15}
        
        # 1. Bipartite edges K_{3,13}
        for a in range(3):
            for b in range(3, 16):
                edges.append([a, b])
                
        # 2. Intra A edges (Triangle)
        edges.append([0, 1])
        edges.append([1, 2])
        edges.append([0, 2])
        
        # 3. Intra B edges (Turan T(13,7))
        # Parts:
        # 3,4
        # 5,6
        # 7,8
        # 9,10
        # 11,12
        # 13,14
        # 15
        parts = [
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15]
        ]
        
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                for u in parts[i]:
                    for v in parts[j]:
                        edges.append([u, v])
                        
        return edges
    
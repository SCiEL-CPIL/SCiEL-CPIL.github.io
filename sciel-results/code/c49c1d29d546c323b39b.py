    def _is_not_2_colorable(n, edges):
        if not edges:
            return False
        
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(i)
        
        assignment = [-1] * n
        
        def backtrack():
            # Check for conflicts and propagate
            while True:
                propagated = False
                for i in range(len(edges)):
                    c0 = 0
                    c1 = 0
                    unassigned_v = -1
                    for v in edges[i]:
                        if assignment[v] == 0:
                            c0 += 1
                        elif assignment[v] == 1:
                            c1 += 1
                        else:
                            unassigned_v = v
                            
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and unassigned_v != -1:
                        if assignment[unassigned_v] == 1:
                            return False
                        assignment[unassigned_v] = 1
                        propagated = True
                    elif c1 == 4 and unassigned_v != -1:
                        if assignment[unassigned_v] == 0:
                            return False
                        assignment[unassigned_v] = 0
                        propagated = True
                            
                if not propagated:
                    break
            
            # Find next unassigned vertex
            next_v = -1
            for v in range(n):
                if assignment[v] == -1:
                    next_v = v
                    break
                    
            if next_v == -1:
                return True
            
            # Try 0
            assignment[next_v] = 0
            if backtrack():
                return True
            assignment[next_v] = -1
            
            # Try 1
            assignment[next_v] = 1
            if backtrack():
                return True
            assignment[next_v] = -1
            
            return False
        
        if backtrack():
            return False
        else:
            return True

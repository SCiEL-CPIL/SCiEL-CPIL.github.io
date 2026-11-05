import random

def construct():
    import itertools
    
    # Build Z_16 colouring
    cols = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(i+1, 16):
            c = (i + j) % 16
            cols[i][j] = c
            cols[j][i] = c
            
    k4s = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    k4s.append(((i,j), (i,k), (i,l), (j,k), (j,l), (k,l)))
                    
    def get_k4_colors(k4_edges, curr_colors):
        cs = set()
        for e in k4_edges:
            u, v = e
            cs.add(curr_colors[u][v])
        return cs
        
    best_colors = 16
    best_solution = None
    
    # Try a few random orders of merging
    for _ in range(20):
        curr_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                curr_colors[i][j] = (i+j) % 16
                curr_colors[j][i] = (i+j) % 16
                
        while True:
            active_colors = set(curr_colors[i][j] for i in range(16) for j in range(i+1, 16))
            if len(active_colors) <= 13:
                break
                
            color_list = list(active_colors)
            random.shuffle(color_list)
            
            merged = False
            # Try all pairs in random order
            for idx1 in range(len(color_list)):
                for idx2 in range(idx1+1, len(color_list)):
                    c1 = color_list[idx1]
                    c2 = color_list[idx2]
                    
                    safe = True
                    for k4_edges in k4s:
                        cs = get_k4_colors(k4_edges, curr_colors)
                        if len(cs) == 5 and c1 in cs and c2 in cs:
                            safe = False
                            break
                            
                    if safe:
                        for i in range(16):
                            for j in range(i+1, 16):
                                if curr_colors[i][j] == c2:
                                    curr_colors[i][j] = c1
                                    curr_colors[j][i] = c1
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
                
        final_count = len(set(curr_colors[i][j] for i in range(16) for j in range(i+1, 16)))
        if final_count < best_colors:
            best_colors = final_count
            best_solution = [curr_colors[i][j] for i in range(16) for j in range(i+1, 16)]
            
    return best_solution

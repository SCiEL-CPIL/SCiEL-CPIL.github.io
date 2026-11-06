import time
import random
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit for safety
    
    # Base Gallai coloring: color(i,j) = min(i,j)
    color_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            c = i
            color_matrix[i][j] = c
            color_matrix[j][i] = c
            
    edge_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Helper: check if recoloring edge (i,j) to new_color maintains validity
    def check_edge_recolor(cm, i, j, new_color):
        others = [v for v in range(N) if v != i and v != j]
        for idx_k in range(len(others)):
            for idx_l in range(idx_k + 1, len(others)):
                k = others[idx_k]
                l = others[idx_l]
                # Cycle 1: i-k-j-l-i
                c1 = [cm[i][k], cm[k][j], cm[j][l], cm[l][i]]
                if len(set(c1)) == 4:
                    return False
                # Cycle 2: i-k-l-j-i
                c2 = [cm[i][k], cm[k][l], cm[l][j], cm[j][i]]
                if len(set(c2)) == 4:
                    return False
                # Cycle 3: i-l-k-j-i
                c3 = [cm[i][l], cm[l][k], cm[k][j], cm[j][i]]
                if len(set(c3)) == 4:
                    return False
        return True

    current_colors = set(color_matrix[i][j] for i in range(N) for j in range(i+1, N))
    next_color = max(current_colors) + 1 if current_colors else 0
    
    # Greedy passes with randomization
    # We try to add as many new colors as possible.
    # Randomizing the order of edges can help find different local optima.
    
    best_result = None
    best_count = len(current_colors)
    
    # Save initial state
    initial_matrix = [row[:] for row in color_matrix]
    
    for restart in range(20): # Try a few restarts if time permits
        if time.time() > deadline - 5:
            break
            
        # Reset to initial state for each restart? Or continue from previous best?
        # Continuing is usually better for "maximize colors" as it's monotonic in success.
        # But randomizing the *order* of processing is key.
        
        # Shuffle edges
        random.shuffle(edge_list)
        
        for (i, j) in edge_list:
            if time.time() > deadline - 2:
                break
                
            old_color = color_matrix[i][j]
            new_color = next_color
            
            # Try to assign new_color
            color_matrix[i][j] = new_color
            color_matrix[j][i] = new_color
            
            if check_edge_recolor(color_matrix, i, j, new_color):
                next_color += 1
                # Update best
                if next_color - 1 > best_count:
                    best_count = next_color - 1
                    best_result = [color_matrix[i][j] for i in range(N) for j in range(i+1, N)]
            else:
                # Revert
                color_matrix[i][j] = old_color
                color_matrix[j][i] = old_color
                
    # If we found a better result, return it. Otherwise return current.
    if best_result:
        return best_result
    else:
        return [color_matrix[i][j] for i in range(N) for j in range(i+1, N)]

if __name__ == "__main__":
    # Just to test
    c = construct()
    print(len(set(c)))

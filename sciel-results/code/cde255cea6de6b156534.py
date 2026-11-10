# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54, m=27.
    Strategy: 7x8 grid (56 pts) minus 2 points, using 7 vertical, 8 horizontal, 
    and 12 diagonal lines (mix of slope 1 and -1).
    """
    # Grid dimensions
    W, H = 7, 8
    
    # Generate all 56 points
    all_points = []
    for x in range(W):
        for y in range(H):
            all_points.append((x, y))
            
    # Define candidate lines
    # 7 Verticals: x = i
    verticals = [(1, 0, -x) for x in range(W)]
    # 8 Horizontals: y = i
    horizontals = [(0, 1, -y) for y in range(H)]
    
    # Diagonals
    # Slope 1: y - x = c  =>  -x + y - c = 0  =>  A=-1, B=1, C=-c
    # Slope -1: y + x = c =>  x + y - c = 0   =>  A=1, B=1, C=-c
    
    def get_diag_lines(slope_sign):
        # slope_sign = 1 for y-x=c, -1 for y+x=c
        lines = []
        for x in range(W):
            for y in range(H):
                if slope_sign == 1:
                    c = y - x
                    # Normalize: -x + y - c = 0
                    # A=-1, B=1, C=-c
                    lines.append((-1, 1, -c))
                else:
                    c = y + x
                    # Normalize: x + y - c = 0
                    # A=1, B=1, C=-c
                    lines.append((1, 1, -c))
        
        # Remove duplicates and calculate lengths
        unique_lines = {}
        for L in lines:
            # Canonical form: divide by gcd, make first non-zero positive
            A, B, C = L
            # GCD of A, B, C
            g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A, B, C = A//g, B//g, C//g
            # Make first non-zero positive
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            
            # Count points on this line
            count = 0
            for px, py in all_points:
                if A*px + B*py + C == 0:
                    count += 1
            if key not in unique_lines or unique_lines[key] < count:
                unique_lines[key] = count
                
        return list(unique_lines.keys()), list(unique_lines.values())

    diag_lines_1, diag_counts_1 = get_diag_lines(1)
    diag_lines_neg1, diag_counts_neg1 = get_diag_lines(-1)
    
    # We want to pick 12 lines from the union of these, prioritizing higher counts.
    # Combine them
    all_diags = []
    for i, L in enumerate(diag_lines_1):
        all_diags.append((diag_counts_1[i], L, 1))
    for i, L in enumerate(diag_lines_neg1):
        all_diags.append((diag_counts_neg1[i], L, -1))
        
    # Sort by count descending
    all_diags.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 12
    top_diags = all_diags[:12]
    
    # The 27 lines are: 7 vert + 8 horiz + 12 diag
    # But we need to choose which 2 points to remove from the 56.
    # We evaluate the incidence count for each pair of removed points.
    
    # To speed up, we can pre-calculate which lines each point is on.
    # point_line_mask[i] is a bitmask of lines (0-26) that pass through point i.
    
    lines_final = verticals + horizontals + [d[1] for d in top_diags]
    
    # Validate lines are unique and valid
    # (Assumed valid by construction, but check for A=B=0 which is not possible here)
    
    point_indices = list(range(56))
    best_inc = -1
    best_remove = (0, 1)
    
    # Precompute incidence contribution of each point to each line
    # inc_matrix[point_idx][line_idx] = 1 if point is on line
    
    inc_matrix = np.zeros((56, 27), dtype=np.int8)
    for i, (px, py) in enumerate(all_points):
        for j, (A, B, C) in enumerate(lines_final):
            if A*px + B*py + C == 0:
                inc_matrix[i, j] = 1
                
    # Total incidences if no points removed
    total_full = np.sum(inc_matrix)
    
    # We remove 2 points i, j.
    # New total = total_full - sum(inc_matrix[i]) - sum(inc_matrix[j])
    # We want to minimize the loss, i.e., minimize sum(inc_matrix[i]) + sum(inc_matrix[j])
    # Note: If a line contains both i and j, removing both reduces its count by 2.
    # The formula total_full - row_sum[i] - row_sum[j] is correct for total incidences.
    
    row_sums = np.sum(inc_matrix, axis=1)
    
    # Find pair (i, j) with minimum row_sums[i] + row_sums[j]
    # To be precise, we just need the two smallest row sums.
    # However, we must ensure the resulting points are distinct (they are, indices are distinct).
    
    # Sort indices by row sum
    sorted_indices = np.argsort(row_sums)
    
    # The two smallest sums are at sorted_indices[0] and sorted_indices[1]
    i_rem = sorted_indices[0]
    j_rem = sorted_indices[1]
    
    best_remove = (i_rem, j_rem)
    
    # Construct final points
    points_to_keep = [k for k in range(56) if k != i_rem and k != j_rem]
    final_points = np.array([all_points[k] for k in points_to_keep], dtype=np.int64)
    
    # Construct final lines
    final_lines = np.array(lines_final, dtype=np.int64)
    
    return final_points, final_lines

# EVOLVE-BLOCK-END

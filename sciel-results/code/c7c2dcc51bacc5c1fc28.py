<<<<<<< SEARCH
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
=======
    def gen_cyclic_colors(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                d = min(d, n - d)
                # Map differences to colors 0,1,2,3
                # For n=9: diffs 1,8->0; 2,7->1; 3,6->2; 4,5->3
                if d == 1 or (n > 2 and d == n - 1):
                    c = 0
                elif d == 2 or (n > 3 and d == n - 2):
                    c = 1
                elif d == 3 or (n > 4 and d == n - 3):
                    c = 2
                else:
                    c = 3
                colors.append(c)
        return colors

    best_n = 9
    best_colors = gen_cyclic_colors(9)
>>>>>>> REPLACE

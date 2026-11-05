    for i in range(2, 129):
        for j in range(i, 129):
            gcd_table[(i, j)] = math.gcd(i, j)
            gcd_table[(j, i)] = gcd_table[(i, j)]

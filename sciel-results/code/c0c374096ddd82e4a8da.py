def construct():
    n = 29
    # Calculate quadratic residues modulo 29
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

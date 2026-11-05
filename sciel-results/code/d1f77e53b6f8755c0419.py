def design(problem):
    size = problem["pixels"]
    # Dielectric background
    cell = [[1 for _ in range(size)] for _ in range(size)]
    
    # Rotated square (diamond) air hole
    # Center is at (0.5, 0.5)
    # Radius R controls the size
    R = 0.25
    
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            if abs(x - 0.5) + abs(y - 0.5) < R:
                cell[i][j] = 0
                
    return cell

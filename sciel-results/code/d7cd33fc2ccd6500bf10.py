def design(problem):
    size = problem["pixels"]
    r = 0.2
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y <= r*r:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell

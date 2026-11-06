import math

def design(problem):
    size = problem["pixels"]
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = 2.0 * math.pi * (i + 0.5) / size
            y = 2.0 * math.pi * (j + 0.5) / size
            val = math.cos(x) * math.cos(y)
            row.append(1 if val > 0 else 0)
        cell.append(row)
    return cell

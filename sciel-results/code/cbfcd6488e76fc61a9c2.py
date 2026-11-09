import math

def design(problem):
    size = problem["pixels"]
    cell = []
    # New approach: Single central air hole in a dielectric background
    # 1 = dielectric (epsilon 12), 0 = air (epsilon 1)
    r = 0.25
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < r * r:
                row.append(0) # Air
            else:
                row.append(1) # Dielectric
        cell.append(row)
    return cell

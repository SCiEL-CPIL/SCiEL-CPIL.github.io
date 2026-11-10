def design(problem):
        size = problem["pixels"]
        cell = [[0]*size for _ in range(size)]
        # Square rod centered
        # Side length 0.45
        half = 0.225
        for i in range(size):
            y = (i + 0.5) / size - 0.5
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                if abs(x) < half and abs(y) < half:
                    cell[i][j] = 1
        return cell

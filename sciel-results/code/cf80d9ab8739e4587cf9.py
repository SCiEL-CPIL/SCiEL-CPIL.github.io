    def design(problem):
        size = problem["pixels"]
        cell = [[0 for _ in range(size)] for _ in range(size)]
        center = (size - 1) / 2.0
        # Define a cross shape
        # Vertical arm
        v_width = 12
        h_width = 12
        for i in range(size):
            for j in range(size):
                # Distance from center in x and y
                dx = abs(i - center)
                dy = abs(j - center)
                # If inside the vertical or horizontal bar of the cross
                if dx < v_width / 2.0 or dy < h_width / 2.0:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
        return cell
    
import cv2
import numpy as np
from queue import Queue

def process_maze_image(image_data):
    # Convert image data to numpy array
    img_array = np.frombuffer(image_data.data, dtype=np.uint8)
    img_array = img_array.reshape((image_data.height, image_data.width, 4))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find entrance and exit
    entrance, exit = find_entrance_exit(binary)
    
    # Solve maze
    path = solve_maze(binary, entrance, exit)
    
    # Draw solution
    solution = draw_solution(img_array, path, entrance, exit)
    
    return solution.tolist()

def find_entrance_exit(binary_image):
    height, width = binary_image.shape
    openings = []

    for col in range(width):
        if binary_image[0, col] == 0:
            openings.append((0, col))
        if binary_image[height - 1, col] == 0:
            openings.append((height - 1, col))
    for row in range(height):
        if binary_image[row, 0] == 0:
            openings.append((row, 0))
        if binary_image[row, width - 1] == 0:
            openings.append((row, width - 1))

    if len(openings) < 2:
        openings = [(0, 0), (height - 1, width - 1)]
    elif len(openings) > 2:
        max_distance = 0
        entrance_exit_pair = None
        for i in range(len(openings)):
            for j in range(i + 1, len(openings)):
                distance = ((openings[i][0] - openings[j][0]) ** 2 +
                            (openings[i][1] - openings[j][1]) ** 2) ** 0.5
                if distance > max_distance:
                    max_distance = distance
                    entrance_exit_pair = (openings[i], openings[j])
        openings = entrance_exit_pair

    entrance = min(openings)
    exit = max(openings)

    return entrance, exit

def solve_maze(binary_image, start, end):
    height, width = binary_image.shape
    queue = Queue()
    queue.put(start)
    visited = set([start])
    parent = {start: None}

    while not queue.empty():
        current = queue.get()
        if current == end:
            break

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_point = (current[0] + dx, current[1] + dy)
            if (0 <= next_point[0] < height and
                    0 <= next_point[1] < width and
                    binary_image[next_point] == 0 and
                    next_point not in visited):
                queue.put(next_point)
                visited.add(next_point)
                parent[next_point] = current

    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)
    path.reverse()

    return path

def draw_solution(image, path, start, end):
    solution_image = image.copy()

    for i in range(1, len(path)):
        cv2.line(solution_image,
                 (path[i - 1][1], path[i - 1][0]),
                 (path[i][1], path[i][0]),
                 (0, 0, 255),
                 thickness=3)

    cv2.circle(solution_image, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(solution_image, (end[1], end[0]), 5, (255, 0, 0), -1)

    return solution_image

# The main execution is now handled by the process_maze_image function

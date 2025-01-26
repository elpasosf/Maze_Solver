import cv2
import numpy as np
from queue import Queue
import numpy as np


def load_image(file_path):
    return cv2.imread(file_path)


def apply_thresholding(gray_image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return binary_image



def process_maze_image(file_path):
    image = load_image(file_path)
    gray_image = convert_to_grayscale(image)
    binary_image = apply_thresholding(gray_image)
    edge_image = detect_edges(gray_image)
    return image, gray_image, binary_image, edge_image



def apply_thresholding(gray_image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return binary_image


def process_maze_image(file_path):
    image = load_image(file_path)
    gray_image = convert_to_grayscale(image)
    binary_image = apply_thresholding(gray_image)
    edge_image = detect_edges(gray_image)
    return image, gray_image, binary_image, edge_image


# Make sure to add this import at the top of your script if not already present



def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_thresholding(gray_image):
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image


def detect_edges(gray_image):
    edges = cv2.Canny(gray_image, 100, 200)
    return edges


def process_maze_image(file_path):
    image = load_image(file_path)
    gray_image = convert_to_grayscale(image)
    binary_image = apply_thresholding(gray_image)
    edge_image = detect_edges(gray_image)
    return image, gray_image, binary_image, edge_image


def find_entrance_exit(binary_image):
    height, width = binary_image.shape
    openings = []

    # Check all edges for openings
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
        # If less than 2 openings, use corners as fallback
        openings = [(0, 0), (height - 1, width - 1)]
    elif len(openings) > 2:
        # If more than 2 openings, choose the two most distant ones
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

    # Determine which opening is the entrance (top or left-most)
    entrance = min(openings)
    exit = max(openings)

    return entrance, exit

    def find_entrance_exit(binary_image):
        height, width = binary_image.shape
        openings = []

        # Check all edges for openings
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
            # If less than 2 openings, use corners as fallback
            openings = [(0, 0), (height - 1, width - 1)]
        elif len(openings) > 2:
            # If more than 2 openings, choose the two most distant ones
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

        # Determine which opening is the entrance (top or left-most)
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

    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)
    path.reverse()

    return path


def draw_solution(image, path, start, end):
    solution_image = image.copy()

    # Draw the path with a thicker red line
    for i in range(1, len(path)):
        cv2.line(solution_image,
                 (path[i - 1][1], path[i - 1][0]),  # Start point
                 (path[i][1], path[i][0]),  # End point
                 (0, 0, 255),  # Red color
                 thickness=3)  # Adjust thickness here (was 1, now 3)

    # Draw start and end points
    cv2.circle(solution_image, (start[1], start[0]), 5, (0, 255, 0), -1)  # Green start
    cv2.circle(solution_image, (end[1], end[0]), 5, (255, 0, 0), -1)  # Blue end

    return solution_image


def display_images(images, titles):
    for img, title in zip(images, titles):
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main execution
input_path = r'C:\Users\porte\OneDrive\Desktop\Input IMG\download.jpg'
output_path = r'C:\Users\porte\OneDrive\Desktop\Output IMG\solved_maze.jpg'

original, gray, binary, edges = process_maze_image(input_path)
entrance, exit = find_entrance_exit(binary)
solution_path = solve_maze(binary, entrance, exit)
solved_image = draw_solution(original, solution_path, entrance, exit)

# Save the solved maze
cv2.imwrite(output_path, solved_image)

# Display only the solved maze
cv2.imshow('Solved Maze', solved_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

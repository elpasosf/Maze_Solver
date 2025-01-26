let pyodide;

async function loadPyodide() {
    pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    await pyodide.loadPackage("opencv-python");
    await pyodide.runPythonAsync(`
        import micropip
        await micropip.install('opencv-python')
    `);
}

loadPyodide();

async function solveMaze() {
    const input = document.getElementById('mazeInput');
    const file = input.files[0];
    if (!file) {
        alert('Please select a maze image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = async function(e) {
        const img = new Image();
        img.onload = async function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            const result = await pyodide.runPythonAsync(`
                import cv2
                import numpy as np
                from js import imageData

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

                # Include your other functions here (find_entrance_exit, solve_maze, draw_solution)

                solution = process_maze_image(imageData)
                solution
            `);

            displaySolution(result);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function displaySolution(solutionData) {
    const canvas = document.createElement('canvas');
    canvas.width = solutionData[0].length;
    canvas.height = solutionData.length;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    
    for (let i = 0; i < solutionData.length; i++) {
        for (let j = 0; j < solutionData[0].length; j++) {
            const idx = (i * canvas.width + j) * 4;
            imgData.data[idx] = solutionData[i][j][0];
            imgData.data[idx + 1] = solutionData[i][j][1];
            imgData.data[idx + 2] = solutionData[i][j][2];
            imgData.data[idx + 3] = solutionData[i][j][3];
        }
    }
    
    ctx.putImageData(imgData, 0, 0);
    
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';
    resultDiv.appendChild(canvas);
    
    const downloadLink = document.createElement('a');
    downloadLink.href = canvas.toDataURL('image/png');
    downloadLink.download = 'solved_maze.png';
    downloadLink.textContent = 'Download Solved Maze';
    resultDiv.appendChild(downloadLink);
}

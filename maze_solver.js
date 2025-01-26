let pyodide;

async function initPyodide() {
    console.log('Starting Pyodide loading...');
    try {
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.22.1/full/"
        });
        console.log('Pyodide loaded successfully');
        await pyodide.loadPackage(['numpy', 'opencv-python']);
    } catch (error) {
        console.error('Error during Pyodide loading:', error);
    }
}

window.solveMaze = async function() {
    if (!pyodide) {
        console.error('Pyodide is not loaded yet');
        return;
    }

    const input = document.getElementById('fileInput');
    if (!input) {
        console.error('File input element not found');
        return;
    }
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

            try {
                const result = await pyodide.runPythonAsync(`
    import cv2
    import numpy as np
    from pyodide import create_proxy
    from js import ImageData

    def process_maze_image(image_data):
        # Convert image data to numpy array
        img_array = np.frombuffer(image_data.data.to_py(), dtype=np.uint8)
        img_array = img_array.reshape((image_data.height, image_data.width, 4))
        
        # Rest of your Python code...

    # Include your other functions here (find_entrance_exit, solve_maze, draw_solution)

    image_data_proxy = create_proxy(imageData)
    solution = process_maze_image(image_data_proxy)
    solution
`, { globals: { imageData: imageData } });


                displaySolution(result);
            } catch (error) {
                console.error('Error running Python code:', error);
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
};

function displaySolution(solutionData) {
    // ... (rest of the displaySolution function remains unchanged)
}

// Initialize Pyodide when the script loads
initPyodide();

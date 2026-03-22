import cv2
import numpy as np
import csv

IMAGE_PATH = "Skin Images/Online.png"
grid_points = []
grid_size = 20
threshold = 40

def print_pixel_values(gray_image, output_file="output.csv"):
    print("Saving grayscale pixel values to CSV...")

    h, w = gray_image.shape

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["y", "x", "intensity"])
        
        for y in range(0,h, grid_size):
            for x in range(0, w, grid_size):
                writer.writerow([y, x, int(gray_image[y, x])])
                grid_points.append([y, x, gray_image[y, x]])
        
    print(f"Saved to {output_file}")


def pixelate(image, pixel_size=10):
    h, w = image.shape[:2]
    temp = cv2.resize(image,
                      (w // pixel_size, h // pixel_size),
                      interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp,
                           (w, h),
                           interpolation=cv2.INTER_NEAREST)
    return pixelated

def calculate_grid_averages(gray_image, block_size):
    h, w = gray_image.shape
    grid_intensities = []

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray_image[y:y+block_size, x:x+block_size]
            avg = np.mean(block)
            print(f'{avg}')    
            grid_intensities.append([y, x, avg])
            
    return grid_intensities

def detect_grid_changes(grid_data, threshold=10):
    """
    Detect intensity changes horizontally, vertically, and diagonally.
    grid_data format: [y, x, intensity]
    """

    steep_changes = []

    # Convert list into dictionary for fast lookup
    grid = {}
    for y, x, val in grid_data:
        grid[(y, x)] = val

    for (y, x), val in grid.items():

        neighbors = [
            (y, x + grid_size),          # right
            (y + grid_size, x),          # down
            (y + grid_size, x + grid_size) # diagonal
        ]   

        for ny, nx in neighbors:
            if (ny, nx) in grid:

                diff = abs(val - grid[(ny, nx)])

                if diff > threshold:
                    steep_changes.append((ny, nx, grid[(ny, nx)], diff))
                    print(f"Change at ({ny},{nx}) diff={diff}")

    return steep_changes


# MAIN
def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Error loading image.")
        return

    # Pixelate
    pixelated = pixelate(image, pixel_size=10)
    cv2.imshow("Pixelated Image", pixelated)
    cv2.waitKey(0)


    # Convert to grayscale
    gray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    intensities = calculate_grid_averages(gray, grid_size)
    
    print_pixel_values(gray, "output.csv")
    results = detect_grid_changes(intensities, threshold)
    with open("steep_changes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "intensity", "change_magnitude"])
        writer.writerows(results)

    for pt in results:
        cv2.circle(image, (pt[1], pt[0]), 5, (0, 0, 255), -1)
        

    cv2.imshow("Steep Changes (Red Dots)", image)
    cv2.waitKey(0)
   

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
from PIL import Image
import numpy as np
import os


def average_tiff_images(input_folder_path, save_folder_path, average_n: int):
    print("Functioned called")
    # Get a list of TIFF files in the specified folder
    tiff_files = [
        file
        for file in os.listdir(input_folder_path)
        if file.endswith(".tiff") or file.endswith(".tif")
    ]

    # stop if not enough tiffs in a folder
    if len(tiff_files) < average_n:
        print("There are not enough TIFF files in the folder.")

    tiff_files.sort()
    print(len(tiff_files))

    for i in range(0, len(tiff_files), average_n):
        # Initialize an array to store the pixel values
        total_pixels = np.zeros((512, 512), dtype=np.float16)

        # Loop through the current set of four TIFF files
        for j in range(i, min(i + average_n, len(tiff_files))):
            tiff_file = tiff_files[j]
            file_path = os.path.join(input_folder_path, tiff_file)
            img = Image.open(file_path)

            # Ensure the image has the correct resolution
            if img.size == (512, 512):
                img_array = np.array(img, dtype=np.float16)
                total_pixels += img_array
            else:
                print(f"Skipping {tiff_file} due to incorrect resolution.")

        # Calculate the average pixel values
        average_image = total_pixels / min(average_n, (len(tiff_files) - i))
        # average_image = (average_image - np.min(average_image)) / (
        #     np.max(average_image) - np.min(average_image)
        # )

        # Save the average image
        average_image_path = os.path.join(
            save_folder_path, f"average_image_{i//average_n}.tiff"
        )
        average_image = Image.fromarray(average_image.astype(np.float16))
        average_image.save(average_image_path)

        print(f"Average image {i//average_n} saved at {average_image_path}")


# Running codes
input_folder_path = "/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/PhD projects/analysis/sample_for_analysis/MZ-hpc-prism-M4-102623-1938-z1.7-sample/"
save_folder_path = "/Users/mzhou/Library/CloudStorage/OneDrive-UCSF/PhD projects/analysis/sample_for_analysis/MZ-hpc-prism-M4-102623-1938-z1.7-sample-averaged/"
number_to_average = 5
average_tiff_images(input_folder_path, save_folder_path, number_to_average)

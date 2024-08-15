import matplotlib.pyplot as plt
from PIL import Image


def show_image(image_path):
    """
    Function to display an image from the given path.

    :param image_path: Path to the image file.
    """
    try:
        with Image.open(image_path) as img:
            plt.imshow(img)
            plt.axis("off")
            plt.show()
    except Exception as e:
        print(f"Error opening image: {e}")

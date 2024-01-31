import matplotlib.pyplot as plt
import matplotlib as mpl

def show_binary_image(image, title=None):

    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV

    # Show the image
    plt.figure(figsize=(5, 3))
    plt.imshow(image, cmap=plt.cm.gray)

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()

def show_rgb_image(m):
    plt.figure(figsize=(5, 3))
    plt.imshow(m, cmap=plt.cm.gray)
    plt.title('RGB Image')
    plt.axis('off')
    plt.show()

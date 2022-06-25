import math
import statistics
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

def RGBtoGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width,image_height)
    for i in range(len(greyscale_pixel_array)):
        for x in range(len(greyscale_pixel_array[i])):
            greyscale_pixel_array[i][x] = round(pixel_array_r[i][x] * 0.299 + pixel_array_g[i][x] * 0.587 + pixel_array_b[i][x] * 0.114)
    return  greyscale_pixel_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    fmin = 255
    fmax = 0
    sout = []
    for i in range(len(pixel_array)):
        if (fmin > min(pixel_array[i])):
            fmin = min(pixel_array[i])
        if (fmax < max(pixel_array[i])):
            fmax = max(pixel_array[i])

    for i in range(image_height):
        stemp = []
        for k in range(image_width):
            if (round(fmax - fmin) == 0):
                stemp += [0]
            else:
                stemp += [round((pixel_array[i][k] - fmin) * ((255 - 0) / (fmax - fmin)) + 0)]
        sout += [stemp]
    return sout


def highContrastComp(pixel_array, image_width, image_height):
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(0, image_height - 4):
        for j in range(0, image_width - 4):

            new_array[i + 1][j + 1] = standarddeviation(pixel_array[i][j], pixel_array[i][j + 1], pixel_array[i][j + 2], pixel_array[i][j + 3], pixel_array[i][j + 4],
                                                  pixel_array[i + 1][j], pixel_array[i + 1][j + 1], pixel_array[i + 1][j + 2], pixel_array[i + 1][j + 3], pixel_array[i + 1][j + 4],
                                                  pixel_array[i + 2][j], pixel_array[i + 2][j + 1], pixel_array[i + 2][j + 2], pixel_array[i + 2][j + 3], pixel_array[i + 2][j + 4],
                                                  pixel_array[i + 3][j], pixel_array[i + 3][j + 1], pixel_array[i + 3][j + 2], pixel_array[i + 3][j + 3], pixel_array[i + 3][j + 4],
                                                  pixel_array[i + 4][j], pixel_array[i + 4][j + 1], pixel_array[i + 4][j + 2], pixel_array[i + 4][j + 3], pixel_array[i + 4][j + 4]);
    return new_array


def standarddeviation(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y):
    standard_array = []
    standard_array.append(a)
    standard_array.append(b)
    standard_array.append(c)
    standard_array.append(d)
    standard_array.append(e)
    standard_array.append(f)
    standard_array.append(g)
    standard_array.append(h)
    standard_array.append(i)
    standard_array.append(j)
    standard_array.append(k)
    standard_array.append(l)
    standard_array.append(m)
    standard_array.append(n)
    standard_array.append(o)
    standard_array.append(p)
    standard_array.append(q)
    standard_array.append(r)
    standard_array.append(s)
    standard_array.append(t)
    standard_array.append(u)
    standard_array.append(v)
    standard_array.append(w)
    standard_array.append(x)
    standard_array.append(y)
    mean = sum(standard_array) / len(standard_array)
    var = sum(pow(x - mean, 2) for x in standard_array) / len(standard_array)
    std = math.sqrt(var)

    return std

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    z = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold_value:
                z[i][j] = 0
            else:
                z[i][j] = 255
    return z


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    arr = []
    for i in range(image_height):
        arr.append([])
        for j in range(image_width):
            if ((j == 0) or (j == image_width - 1)) or ((i == 0) or (i == image_height - 1)):

                try:
                    pa1 = pixel_array[i - 1][j - 1]
                except:
                    pa1 = 0

                try:
                    pa2 = pixel_array[i - 1][j]
                except:
                    pa2 = 0

                try:
                    pa3 = pixel_array[i - 1][j + 1]
                except:
                    pa3 = 0

                try:
                    pa4 = pixel_array[i][j - 1]
                except:
                    pa4 = 0

                try:
                    pa5 = pixel_array[i][j]
                except:
                    pa5 = 0

                try:
                    pa6 = pixel_array[i][j + 1]
                except:
                    pa6 = 0

                try:
                    pa7 = pixel_array[i + 1][j - 1]
                except:
                    pa7 = 0

                try:
                    pa8 = pixel_array[i + 1][j]
                except:
                    pa8 = 0

                try:
                    pa9 = pixel_array[i + 1][j + 1]
                except:
                    pa9 = 0

                if pa1 != 0 or pa2 != 0 or pa3 != 0 or pa4 != 0 or pa5 != 0 or pa6 != 0 or pa7 != 0 or pa8 != 0 or pa9 != 0:
                    arr[i].append(1)
                else:
                    arr[i].append(0)
            elif (pixel_array[i - 1][j - 1] == 1) or (pixel_array[i - 1][j] == 1) or (
                    pixel_array[i - 1][j + 1] == 1) or (pixel_array[i][j - 1] == 1) or (pixel_array[i][j] == 1) or (
                    pixel_array[i][j + 1] == 1) or (pixel_array[i + 1][j - 1] == 1) or (pixel_array[i + 1][j] == 1) or (
                    pixel_array[i + 1][j + 1] == 1):
                arr[i].append(1)
            elif (pixel_array[i - 1][j - 1] == 255) or (pixel_array[i - 1][j] == 255) or (
                    pixel_array[i - 1][j + 1] == 255) or (pixel_array[i][j - 1] == 255) or (
                    pixel_array[i][j] == 255) or (pixel_array[i][j + 1] == 255) or (
                    pixel_array[i + 1][j - 1] == 255) or (pixel_array[i + 1][j] == 255) or (
                    pixel_array[i + 1][j + 1] == 255):
                arr[i].append(1)
            else:
                arr[i].append(0)
    return arr


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    arr = []
    for i in range(image_height):
        arr.append([])
        for j in range(image_width):
            if ((j == 0) or (j == image_width - 1)):
                arr[i].append(0)
            elif ((i == 0) or (i == image_height - 1)):
                arr[i].append(0)
            elif (pixel_array[i - 1][j - 1] == 1) and (pixel_array[i - 1][j] == 1) and (
                    pixel_array[i - 1][j + 1] == 1) and (pixel_array[i][j - 1] == 1) and (pixel_array[i][j] == 1) and (
                    pixel_array[i][j + 1] == 1) and (pixel_array[i + 1][j - 1] == 1) and (
                    pixel_array[i + 1][j] == 1) and (pixel_array[i + 1][j + 1] == 1):
                arr[i].append(1)
            elif (pixel_array[i - 1][j - 1] == 255) and (pixel_array[i - 1][j] == 255) and (
                    pixel_array[i - 1][j + 1] == 255) and (pixel_array[i][j - 1] == 255) and (
                    pixel_array[i][j] == 255) and (pixel_array[i][j + 1] == 255) and (
                    pixel_array[i + 1][j - 1] == 255) and (pixel_array[i + 1][j] == 255) and (
                    pixel_array[i + 1][j + 1] == 255):
                arr[i].append(1)
            else:
                arr[i].append(0)
    return arr


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    label = 1
    new_array = [[0 for x in range(image_width)] for y in range(image_height)]
    new_dict = {}
    visited = set()

    for i in range(0, image_height):
        for j in range(0, image_width):
            if ((pixel_array[i][j] != 0) & ((i, j) not in visited)):
                q = Queue()
                q.enqueue((i, j))
                visited.add((i, j))
                value = 0
                while not q.isEmpty():
                    x, y = q.dequeue()
                    value += 1
                    new_array[x][y] = label

                    if (0 <= y - 1) and (pixel_array[x][y - 1] != 0) and ((x, y - 1) not in visited):
                        q.enqueue((x, y - 1))
                        visited.add((x, y - 1))
                    if (y + 1 < image_width) and (pixel_array[x][y + 1] != 0) and ((x, y + 1) not in visited):
                        q.enqueue((x, y + 1))
                        visited.add((x, y + 1))
                    if (0 <= x - 1) and (pixel_array[x - 1][y] != 0) and ((x - 1, y) not in visited):
                        q.enqueue((x - 1, y))
                        visited.add((x - 1, y))
                    if (x + 1 < image_height) and (pixel_array[x + 1][y] != 0) and ((x + 1, y) not in visited):
                        q.enqueue((x + 1, y))
                        visited.add((x + 1, y))

                new_dict[label] = value
                label += 1
    return new_array, new_dict


def computeboxboundry(pixel_array, dic):
    list_k = []
    count = 0
    while count < len(dic.keys()):
        key = 0
        x = 0
        for y in dic.keys():
            if dic[y] > x:
                x = dic[y]
                key = y
        list_k.append(key)
        count += 1
    for key in list_k:
        left = len(pixel_array[0])
        up = len(pixel_array)
        down = 0
        right = 0
        for i in range(len(pixel_array)):
            for j in range(len(pixel_array[i])):
                if pixel_array[i][j] == key and j < left:
                    left = j

                if pixel_array[i][j] == key and i < up:
                    up = i

                if pixel_array[i][j] == key and j > right:
                    right = j

                if pixel_array[i][j] == key and i > down:
                    down = i
        if (right - left) == 0 or (down - up) == 0:
            ratio = 0
        else:
            ratio = ((right - left) / (down - up))
        if 1.5 < ratio < 5:
            return left, right, down, up
    return left, right, down, up



# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate1.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])


    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    # Conversion to Greyscale and Contrast Stretching
    #STEP1
    greyscale_pixel_array = RGBtoGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    #STEP2
    contrast_stretch = scaleTo0And255AndQuantize(greyscale_pixel_array, image_width, image_height)
    #STEP3
    high_contrast_stretch = highContrastComp(contrast_stretch, image_width, image_height)
    contrast_stretch_2 = scaleTo0And255AndQuantize(high_contrast_stretch,image_width,image_height)
    #STEP4
    threshed = computeThresholdGE(contrast_stretch_2,210,image_width,image_height)
    #STEP5
    dilation = computeDilation8Nbh3x3FlatSE(threshed,image_width,image_height)
    dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)
    dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)
    dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)
    dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)
    dilation = computeDilation8Nbh3x3FlatSE(dilation, image_width, image_height)



    erosion = computeErosion8Nbh3x3FlatSE(dilation,image_width,image_height)
    erosion = computeErosion8Nbh3x3FlatSE(erosion, image_width, image_height)
    erosion = computeErosion8Nbh3x3FlatSE(erosion, image_width, image_height)
    erosion = computeErosion8Nbh3x3FlatSE(erosion, image_width, image_height)
    erosion = computeErosion8Nbh3x3FlatSE(erosion, image_width, image_height)
    erosion = computeErosion8Nbh3x3FlatSE(erosion, image_width, image_height)








    #STEP6
    connected = computeConnectedComponentLabeling(dilation,image_width,image_height)
    #STEP7

    px_array = px_array_r

    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # center_x = image_width / 2.0
    # center_y = image_height / 2.0
    # bbox_min_x = center_x - image_width / 4.0
    # bbox_max_x = center_x + image_width / 4.0
    # bbox_min_y = center_y - image_height / 4.0
    # bbox_max_y = center_y + image_height / 4.0

    numbers = computeboxboundry(connected[0],connected[1])

    bbox_min_x = numbers[0]
    bbox_max_x = numbers[1]
    bbox_min_y = numbers[3]
    bbox_max_y = numbers[2]




    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)



    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
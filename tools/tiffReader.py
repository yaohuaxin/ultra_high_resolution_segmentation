import argparse
import skimage.io as io

parser = argparse.ArgumentParser(description='Input image file.')
parser.add_argument('imageFile',
                    help='Input image file.')

args = parser.parse_args()

imageFilePath = args.imageFile

mask = io.imread(imageFilePath)
print("shape of narray: ", mask.shape)
io.imshow(mask)
io.show()
'''
for row in range(mask.shape[0]):
    for col in range(mask.shape[1]):
        print(row, col)
        if mask[row][col] != 0:            
            print(mask[row][col])
'''
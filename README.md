# Image-Rotation-Detection-using-Transfer-Learning-
I used pretrained VGG-16 to carry out transfer learning to detect rotation in images.

I will update the README with further details very soon. And add a sample collection of images, trained model file. The actual images I've used are not for public use.

# Overview
1. I had a collection of images, around 40k of different sizes.
2. I randomly rotated these images using cv2 module. The rotations are 90 degrees clockwise, 90 degrees anticlockwise, and 180 degrees. A general 360 degree rotated version can also be made.
3. The dataset consists of images and the labels which are how the image has been rotated. This is also according to cv2. Specifically, cv2.ROTATE_90_CLOCKWISE=0, cv2.ROTATE_180=1, and cv2.ROTATE_90_COUNTERCLOCKWISE=2.
4. All the images are stored in a directory. They're read and rotated, then stored in another directory which has subdirectory named with the rotation label. So, images rotated 90 degree clockwise are stored in 0/ and so on.
5. Keras is used to load pretrained-VGG16 and add further layers on top, creating a network, loading train, validation, and test images.

# Directory structure
split_data/
  train/
    -1/
     0/
     1/
     2/
  test/
    -1/
     0/
     1/
     2/
  val/
    -1/
     0/
     1/
     2/

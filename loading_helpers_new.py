import cv2
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import ImageGrid

def get_image_slices(img):
    
    dim = 2560
    imgs = np.zeros([4,640,640,3])
    imgs[0] = img[0:int(dim/4), 0:int(dim/4), :]
    imgs[1] = img[int(dim/4):int(dim/2), int(dim/4):int(dim/2), :]
    imgs[2] = img[int(dim/2):int(3*dim/4), int(dim/2):int(3*dim/4), :]
    imgs[3] = img[int(3*dim/4):int(dim), int(3*dim/4):int(dim), :]
    
    return np.asarray(imgs)

def slice_xy_from_paths_array(img_paths):
    raw_img = cv2.imread(img_paths[0])
    raw_label = cv2.imread(img_paths[1])
    imgs = get_image_slices(raw_img)
    labels = get_image_slices(raw_label)
  
    return imgs, labels

def cvtImages(img, label=False):

    if not label:
        img = img[:, :, 0]
        img = img.reshape(640,640,1)
    else:
        img = img[:, :, 0:2]
    img = img/255
    return img

def visual_test_n_loaded_imgs(n, ready_im, ready_lab):
    print("The grid is being compiled of %d images"%(n))
    print(ready_im.shape)
    #plt.imshow(ready_im[1,:,:])
    #plt.imshow(ready_lab[1,:,:,1])

    fig = plt.figure(figsize = (100, 200))
    grids = ImageGrid(fig, 111, nrows_ncols= (n, 2), axes_pad = 0.1)

    z = 0
    
    for i in range(n):  
        test_im=ready_im[i, :, :, 0]
        original = ready_lab[i, :, :, 1]
        z = 2*i
        #print(test_im)  
        grid1 = grids[z]
        grid2 = grids[z+1]
        grid1.imshow(original, cmap="gray")
        grid2.imshow(test_im, cmap = "gray")
        
def visual_test_n_preds(n, pred, ready_lab, lab_layer, pred_layer):
    print("The grid is being compiled of %d images"%(n))
    #print(ready_im.shape)
    #plt.imshow(ready_im[1,:,:])
    #plt.imshow(ready_lab[1,:,:,1])

    fig = plt.figure(figsize = (100, 200))
    grids = ImageGrid(fig, 111, nrows_ncols= (n, 2), axes_pad = 0.1)

    z = 0
    
    for i in range(n):  
        test_im=pred[i, :, :, pred_layer]
        original = ready_lab[i, :, :, lab_layer]
        z = 2*i
        #print(test_im)  
        grid1 = grids[z]
        grid2 = grids[z+1]
        grid1.imshow(original, cmap="gray")
        grid2.imshow(test_im, cmap = "gray")
    #plt.figsave("./from_func.pdf")
    return fig

def load_imgs_parallel(training_img_paths, num_slices=10):
    
    new_pool = mp.Pool(mp.cpu_count())
    
    feed_arr = [training_img_paths["x"], training_img_paths["y"]]
    feed_arr = np.asarray(feed_arr)
    feed_arr = feed_arr.T
    
    num_training_images = len(training_img_paths["x"])
    num_slices = num_slices
    load_batch_size = int(num_training_images/num_slices)
    
    processed_imgs = np.zeros([0,640,640, 1])
    
    processed_labs = np.zeros([0,640,640, 2])
    print("feed array's shape:"+str(feed_arr.shape))
    
    for i in range(num_slices):
        print("%4d slice is being processed out of %4d slice."%((i+1), num_slices))
        cur_start = int(i*load_batch_size)
        cur_end = int((i+1)*load_batch_size)
        cur_slice = feed_arr[cur_start:cur_end]
        
        loaded_imgs = [new_pool.map(slice_xy_from_paths_array, [pair for pair in cur_slice])]
        
        squeezed = np.squeeze(loaded_imgs)
        squeezed = squeezed.swapaxes(0,1)
        squeezed = np.reshape(squeezed, (2, load_batch_size*4, 640, 640, 3))
        print(squeezed.shape)

        temp_imgs =np.asarray([cvtImages(img) for img in squeezed[0]])
        temp_labs =np.asarray([cvtImages(img, label = True) for img in squeezed[1]])
        
        processed_imgs = np.vstack([processed_imgs, temp_imgs])
        processed_labs = np.vstack([processed_labs, temp_labs])
    
    new_pool.close()
    
    return processed_imgs, processed_labs
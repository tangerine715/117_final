import matplotlib.pyplot as plt
import numpy as np
from rembg import remove
from PIL import Image
import cv2

#taken from selectpoints
class SelectPoints:
    """
    Class that encapsulates allowing the user to click on a matplotlib
    axis and renders the points they clicked on
    """
    def __init__(self,ax,npoints):
        self.ax = ax
        self.pts, = ax.plot([0],[0],'r.')
        self.npoints = npoints
        self.xs = list()
        self.ys = list()
        self.cid = self.pts.figure.canvas.mpl_connect('button_press_event',self)

    def __call__(self, event):      
        #ignore clicks outside the figure axis
        if event.inaxes!=self.pts.axes: 
            return
        
         #otherwise record the click location and draw point on the plot
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.pts.set_data(self.xs,self.ys)
        self.ax.text(event.xdata+150,event.ydata,('%s'%len(self.xs)),bbox=dict(facecolor='red',alpha=0.3),verticalalignment='top')
        self.pts.figure.canvas.draw()
        
        #once we have npoints clicked, stop listening for
        #click events so that we don't accidentally add more 
        if (len(self.xs) >= self.npoints):
            self.pts.figure.canvas.mpl_disconnect(self.cid)

def select_k_points(ax,npoints):
    """
    Function to allow for interactive selection of points, displaying
    a number along side each point you click in order to make it easier
    to maintain correspondence.
    
    Parameters
    ----------
    ax : matplotlib figure axis
        Indicates the axis where you want to allow the user to select points
    npoints : int
        How many points are needed from the user
        
    Returns
    -------
    SelectPoints object
        Returns an object with fields xs and ys that contain the point 
        coordinates once the user clicks
        
    """

    ax.set_title(('click to select %d points' % npoints))
    selectpoints = SelectPoints(ax,npoints)
    #plt.show()
    return selectpoints

def get_correspondences(imnames,baseim,npoints=4):
    
    """
    The function loads in a set of images which are going to be assembled
    in to a mosaic. It displays the central (base) image along with each
    peripheral image and allows the user to click to to select corresponding
    points between the base image and the peripheral image. It returns the
    images along with the user input. If the original image files were in 
    color this function converts them to grayscale. 
    
    Parameters
    ----------
    imnames : list of str
        Filenames of image files that are going in to the mosaic
    
    baseim : int
        An index which specifies which of the image files is the base image
        
    npoints : int
        How many points are required from the user. Defaults to 4
        
    Returns
    -------
    imgs : list of 2D float arrays
        The arrays for the corresponding images given in imnames. These 
        are gray scale images represented as floats.   
     
    pointmatches : list of SelectPoints objects
        Returns an object whose fields xs and ys contain the point 
        coordinates once the user has clicked  (see selectpoints.py)
        
    """

    nimages = len(imnames)

    #loop over images and load in each one and convert to grayscale
    imgs = list()
    for fname in imnames:  
        print('loading...',fname)
        I = fname #modified this since i dont have the name :)
        #convert to float data and from color to grayscale if necessary
        #optional: downsample the image to 1/4 resolution just to make things run quickly
        #zoom(I, zoom=0.25, order = 4)
        ## your code here
        
        # converting to greyscale
        I = I.astype(np.float32) / 255.0
        I = np.mean(I, axis=I.shape[2]-1)
        
        
        #finally, store the array in our list of images
        imgs.append(I)
 

    #loop over each pair of overlapping images and have the user 
    #click to specify corresponding points
    pointmatches = list()
    for i in range(nimages):
        if (i==baseim):
            continue
        
        fig = plt.figure()
    
        #select points in base image
        ax1 = fig.add_subplot(2,1,1)
        ax1.imshow(imgs[baseim],cmap=plt.cm.gray)
 
        #corresponding points in overlapping image
        ax2 = fig.add_subplot(2,1,2)
        ax2.imshow(imgs[i],cmap=plt.cm.gray)

        sp1 = select_k_points(ax1,npoints)
        sp2 = select_k_points(ax2,npoints)
 
        pointmatches.append((sp1,sp2))
        fig.tight_layout()

    return imgs,pointmatches

#my own

#remove background of an input image, subject must be in focus
def remove_bg(input_path, output_path):
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    output_image.save(output_path)

#split video into multiple frames 50 ms apart
def split_vid(vidname):
    vidcap = cv2.VideoCapture(f'videos/{vidname}.MOV')
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(f'frames/{vidname}/frame%d.jpg' % count, image)      
      success,image = vidcap.read()
      count += 50

#load image with bg removed and convert to gray
def load_fg(path):
    img_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2GRAY)
    mask = img_rgba[:, :, 3]
    return gray, mask
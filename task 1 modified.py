import cv2
import numpy as np
x=input("Enter address of image(image should be png file)")#reading the required image , copy the path from properties and paste make sure
#to write the image name.png as well,make sure image is in png form
#sample input is C:\\Users\\Aravind\\Desktop\\Python\\flower.png
#also possible input is C:\Users\Aravind\Desktop\Python\time.png
sampl =cv2.imread(x,0)
#sample=cv2.GaussianBlur(sampl,(5,5),0)#applying gaussian blur

def conv_transform(image): #function to convolute, convolution rotates the kernel by 180 degrees
    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j]=image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy
#cv2.imshow("originage",conv_transform(sample))
def convolution(image, kernel):
    kernel=conv_transform(kernel)
    image_h=image.shape[0] #gives us the height of image
    image_w=image.shape[1]#gives us the width of image

    kernel_h=kernel.shape[0] #gives us the height of the kernel
    kernel_w=kernel.shape[1]  #gives us the width of the kernel

    h=kernel_h//2 #this step is necessary because of the way in which the kernel is multiplied to the image
    w=kernel_w//2
    #while convoluting the image and kernel we need to make sure that it is done in the correct manner and in the same order
    image_conv=np.zeros(image.shape)

    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum=0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum+=kernel[m][n]*image[i-h+m][j-w+n]
            image_conv[i][j]=sum
    return image_conv
def convolution1(image, kernel):
    kernel=conv_transform(kernel)
    image_h=image.shape[0] #gives us the height of image
    image_w=image.shape[1]#gives us the width of image

    kernel_h=kernel.shape[0] #gives us the height of the kernel
    kernel_w=kernel.shape[1]  #gives us the width of the kernel

    h=kernel_h//2 #this step is necessary because of the way in which the kernel is multiplied to the image
    w=kernel_w//2
    #while convoluting the image and kernel we need to make sure that it is done in the correct manner and in the same order
    image_conv=np.zeros(image.shape)

    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum=0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum+=kernel[m][n]*image[i-h+m][j-w+n]
            image_conv[i][j]=sum/273
    return image_conv
def norm(img1,img2):
    img_copy=np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q=(img1[i][j]**2 + img2[i][j]**2)**(1/2)
            if(q>55):
                img_copy[i][j]=255
            else:
                img_copy[i][j]=0
    return img_copy
kernelg=np.zeros([5,5])#kernel for gaussian filter
kernelg[0,0]=1
kernelg[0,1]=4
kernelg[0,2]=7
kernelg[0,3]=4
kernelg[0,4]=1
kernelg[1,0]=4
kernelg[1,1]=16
kernelg[1,2]=26
kernelg[1,3]=16
kernelg[1,4]=4
kernelg[2,0]=7
kernelg[2,1]=26
kernelg[2,2]=41
kernelg[2,3]=26
kernelg[2,4]=7
kernelg[3,0]=4
kernelg[3,1]=16
kernelg[3,2]=26
kernelg[3,3]=16
kernelg[3,4]=4
kernelg[4,0]=1
kernelg[4,1]=4
kernelg[4,2]=7
kernelg[4,3]=4
kernelg[4,4]=1
sample=convolution1(sampl,kernelg)

#kernel = np.zeros(shape=(3,3))
kernel=np.zeros([3,3]) #creating kernel as 3X3 matrix initialising all values as zero
#creating kernel of sobel y 
kernel[0,0]=-1
kernel[0,1]=-2
kernel[0,2]=-1
kernel[1,0]=0
kernel[1,1]=0
kernel[1,2]=0
kernel[2,0]=1
kernel[2,1]=2
kernel[2,2]=1
##kernel[0,0]=-3
##kernel[0,1]=-10
##kernel[0,2]=-3
##kernel[1,0]=0
##kernel[1,1]=0
##kernel[1,2]=0
##kernel[2,0]=3
##kernel[2,1]=10
##kernel[2,2]=3
gy = convolution(sample,kernel)
#cv2.imshow("gradient_y",gy)

#creating kernel of sobel x
kernel[0,0]=-1
kernel[0,1]=0
kernel[0,2]=1
kernel[1,0]=-1
kernel[1,1]=0
kernel[1,2]=1
kernel[2,0]=-2
kernel[2,1]=0
kernel[2,2]=2
##kernel[0,0]=-3
##kernel[0,1]=0
##kernel[0,2]=3
##kernel[1,0]=-10
##kernel[1,1]=0
##kernel[1,2]=10
##kernel[2,0]=-3
##kernel[2,1]=0
##kernel[2,2]=3
gx = convolution(sample,kernel)
#cv2.imshow("gradient_x",gx)
g_sobel=norm(gx,gy)

cv2.imshow("Sobel_edge",g_sobel)
#cv2.imshow("original image",sample)

cv2.waitKey(0)
cv2.destroyAllWindows()

    


    

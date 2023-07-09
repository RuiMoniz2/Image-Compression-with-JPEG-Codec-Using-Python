import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from bitarray import bitarray
import math
from PIL import Image
import time
import scipy.fftpack as fft
import numpy as np


colors_red = [(0, 0, 0), (1, 0, 0)]
colors_green = [(0, 0, 0), (0, 1, 0)]
colors_blue = [(0, 0, 0), (0, 0, 1)]
colors_grayscale = [(0,0,0), (1, 1, 1)]

def main():
    image=readImage()
    h,w =image[:,:,0].shape
   #ex7_1(image)
    Y,Cb,Cr, Q_CbCr, Q_Y =encoder(image,50,"4:2:0",64)
    R,G,B = decoder(Y,Cb,Cr,50,"4:2:0",64,w,h,Q_CbCr,Q_Y)
    img = np.empty(image.shape, dtype='uint8')
    img[:,:,0] = R
    img[:,:,1] = G
    img[:,:,2] = B
    #image=encoder(Y,Cb,Cr,50,"4:2:2")
    #ex3(image)
    #decoder(Y,Cb,Cr,factor,sampType,BS,originalWidth,originalHeight,Q_CbCr,Q_Y):
    
    #ycbcr=convertYCbCr(image)
    #showImage(image,"Logo.bmp")
    #ex5(image)
    #ex4(image)
    
    
    #ex4(image)
    #R = image[:,:,0]
    #G = image[:,:,1]
    #B = image[:,:,2]
    #padding(image)
    #Y, Cb, Cr =  convertYCbCr(R, G, B)
    #print(Y.shape)
    #print(Cb.shape)
    #print(Cr.shape)
    #H,W = Cb.shape
    #showImageDCT(Cb,"Canal R")
    #Y = dct_blocks(Y,8)
    #Cb = dct_blocks(Cb,8)
    #Cr = dct_blocks(Cr,8)
    #print(Y.shape)
    #print(Cb.shape)
    #print(Cr.shape)
    #showImageDCT(Cb,"DCT do canal R")
    
    #Y, Cb, Cr =  quantization(Y, Cb, Cr,50)
    #Y = idct_blocks(Y,8,H,W)
    #showImageDCT(Cb,"Quantized")
    #Y, Cb, Cr =  idcpm(Y, Cb, Cr)
    #Y = idct_blocks(Y,8,H,W)
   
    # showImageDCT(Y,"IDCPM")
    
def encoder(image,factor,sampType,BS):
        
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    Y, Cb, Cr = convertYCbCr(R, G, B)
    
    Y, Cb, Cr = downsampling(Y, Cb, Cr, sampType)
    print(np.size(Y))
    print(np.size(Cb))
    print(np.size(Cr))
    #showImageCM(colors_grayscale, 'gray', Y, 'Y channel')
    #showImageCM(colors_grayscale, 'gray', Cb, 'Cb channel')
    #showImageCM(colors_grayscale, 'gray', Cr, 'Cr channel')
    
    Y,HY,WY = dct_blocks(Y, BS)
    Cb,HC,WC = dct_blocks(Cb, BS)
    Cr,HC,WC = dct_blocks(Cr, BS)
    showImageCM(colors_grayscale, 'gray', Y, 'Y channel')
    showImageCM(colors_grayscale, 'gray', Cb, 'Cb channel')
    showImageCM(colors_grayscale, 'gray', Cr, 'Cr channel')
    
        
    Y, Cb, Cr, Q_CbCr, Q_Y = quantization(Y, Cb, Cr, factor)
    
        
    Y , Cb, Cr = dcpm(Y, Cb, Cr)

        
    
    return Y, Cb ,Cr , Q_CbCr , Q_Y

def decoder(Y,Cb,Cr,factor,sampType,BS,originalWidth,originalHeight,Q_CbCr,Q_Y):
    if(sampType == '4:2:2'):
        Cb_height = originalHeight
        Cr_height = originalHeight
        Cb_width = int(originalWidth/2) + (originalWidth % 2)
        Cr_width = int(originalWidth/2) + (originalWidth % 2)
    elif(sampType == '4:2:0'):
        Cb_height = int(originalHeight/2) + (originalHeight % 2)
        Cr_height = int(originalHeight/2) + (originalHeight % 2)
        Cb_width = int(originalWidth/2) + (originalWidth % 2)
        Cr_width = int(originalWidth/2) + (originalWidth % 2)
    elif(sampType == '4:4:4'):
        Cb_height = originalHeight
        Cr_height = originalHeight
        Cb_width = originalWidth
        Cr_width = originalWidth
    
    Y, Cb, Cr = idcpm(Y, Cb, Cr)
    Y, Cb, Cr = iquantization(Y, Cb, Cr, Q_CbCr, Q_Y)
    Y = idct_blocks(Y, originalHeight, originalWidth, BS)
    Cb = idct_blocks(Cb, Cb_height, Cb_width, BS)
    Cr = idct_blocks(Cr, Cr_height, Cr_width, BS)
    Y, Cb, Cr = upsampling(Y, Cb, Cr, sampType)
    print(Y.shape)
    print(Cb.shape)
    R, G, B = convertRGB(Y, Cb, Cr)
    
    
    
    
    return R, G, B



def readImage():
    img = plt.imread('C:/Users/ruipm/Desktop/TP1 Solução/imagens/peppers.bmp')
    return img
    
def showImageCM(colors,name,image,title):
    colormap = clr.LinearSegmentedColormap.from_list(name, colors, N=256)
    plt.figure()
    plt.title(title)
    plt.imshow(image,cmap = colormap)
    
    
def showImageDCT(image,title):
    colors= [(0,0,0), (1,0.5,0.5)]
    colormap = clr.LinearSegmentedColormap.from_list("dct YCBCR", colors, N=256)
    plt.figure()
    plt.title(title)
    plt.imshow(np.log(abs(image) + 0.0001), cmap = colormap)
    
    
def convertYCbCr(R,G,B):
    height, width = R.shape
    Y = np.empty((height, width), dtype = np.uint8)
    Cb = np.empty((height, width), dtype = np.uint8)
    Cr = np.empty((height, width), dtype = np.uint8)

    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = (B - Y)/1.772 + 128
    Cr = (R - Y)/1.402 + 128

    return Y, Cb, Cr


def convertRGB(Y,Cb,Cr):
    height, width = Y.shape
    R = np.empty((height, width), dtype = np.uint8)
    G = np.empty((height, width), dtype = np.uint8)
    B = np.empty((height, width), dtype = np.uint8)
    
    R = Y + 1.402*(Cr - 128)
    G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    B = Y + 1.771*(Cb - 128)
    
    
    R[R>255]=255
    R[R<0]=0

    G[G>255]=255
    G[G<0]=0

    B[B>255]=255
    B[B<0]=0

    R= np.round(R).astype(np.uint8)
    G= np.round(G).astype(np.uint8)
    B= np.round(B).astype(np.uint8)

    return R, G, B

    
 


def dct2(img):
    X_dct =fft.dct(fft.dct(img, norm="ortho").T, norm="ortho").T
    return X_dct

def idct2(img):
    X_idct =fft.idct(fft.idct(img, norm="ortho").T, norm="ortho").T
    return X_idct

def dct_blocks(data,BS):
    height, width = data.shape
    
    nrColumn_padding = (BS - width % BS) % BS
    nrLine_padding = (BS - height % BS) % BS
    
    
    lastColumn = data[:, width - 1].reshape(height, 1)    
    extraColumns = np.repeat(lastColumn, nrColumn_padding, axis=1)
    data = np.concatenate((data, extraColumns), axis=1)  

    lastLine = data[height - 1, :].reshape(1, width + nrColumn_padding) 
    extraLines = np.repeat(lastLine, nrLine_padding , axis=0) 
    data = np.concatenate((data, extraLines), axis=0)

    
    nrBlocks_height = (height + nrColumn_padding) / BS
    nrBlocks_width = (width + nrLine_padding) / BS
    for i in range(int(nrBlocks_height)):
        for j in range(int(nrBlocks_width)):
            data[i*BS:i*BS + BS, j*BS :j*BS + BS] = dct2(data[i*BS:i*BS + BS, j*BS :j*BS + BS])
    
    return data,height+nrLine_padding,width+nrColumn_padding

def idct_blocks(data,BS,originalHeight,originalWidth):
    height,width = data.shape
    nrBlocks_height = height / BS
    nrBlocks_width = width / BS
    
    for i in range(int(nrBlocks_height)):
        for j in range(int(nrBlocks_width)):
            data[i*BS:i*BS + BS, j*BS :j*BS + BS] = idct2(data[i*BS:i*BS + BS, j*BS :j*BS + BS])

    data = data[0:originalHeight, 0:originalWidth]

    return data
    
    
    
    
   
    
    
    
def dcpm(Y,Cb,Cr):
    Yheight, Ywidth = Y.shape
    Cheight, Cwidth = Cb.shape
            
    Y_Final = np.copy(Y)
    Cb_Final = np.copy(Cb)
    Cr_Final = np.copy(Cr)
            
    for i in range(8,Ywidth,8):
        Y_Final[0, i] = Y[0, i] - Y[0, i - 8]
        if(i < Cwidth):
            Cb_Final[0, i] = Cb[0, i] - Cb[0, i - 8]
            Cr_Final[0, i] = Cr[0, i] - Cr[0, i - 8]
    for i in range(8, Yheight, 8):
        Y_Final[i, 0] = Y[i, 0] - Y[i - 8, Ywidth - 8]
        if(i < Cheight):
            Cb_Final[i, 0] = Cb[i, 0] - Cb[i - 8, Cwidth - 8]
            Cr_Final[i, 0] = Cr[i, 0] - Cr[i - 8, Cwidth - 8]
        for j in range(8, Ywidth, 8):
            Y_Final[i, j] = Y[i, j] - Y[i, j - 8]
            if(i < Cheight and j < Cwidth):
                Cb_Final[i, j] = Cb[i, j] - Cb[i, j - 8]
                Cr_Final[i, j] = Cr[i, j] - Cr[i, j - 8]		
    return Y_Final, Cb_Final, Cr_Final

def idcpm(Y,Cb,Cr):
    Yheight, Ywidth = Y.shape
    Cheight, Cwidth = Cb.shape
        
    Y_Final = np.copy(Y)
    Cb_Final = np.copy(Cb)
    Cr_Final = np.copy(Cr)
    
    for i in range(8, Ywidth, 8):
        Y_Final[0, i] = Y[0, i] + Y_Final[0, i - 8]
        if(i < Cwidth):
            Cb_Final[0, i] = Cb[0, i] + Cb_Final[0, i - 8]
            Cr_Final[0, i] = Cr[0, i] + Cr_Final[0, i - 8]
    
    for i in range(8, Yheight, 8):
        Y_Final[i, 0] = Y[i, 0] + Y_Final[i - 8, Ywidth - 8]
        if(i < Cheight):
            Cb_Final[i, 0] = Cb[i, 0] + Cb_Final[i - 8, Cwidth - 8]
            Cr_Final[i, 0] = Cr[i, 0] + Cr_Final[i - 8, Cwidth - 8]
        for j in range(8, Ywidth, 8):
            Y_Final[i, j] = Y[i, j] + Y_Final[i, j - 8];
            if(i < Cheight and j < Cwidth):
                Cb_Final[i, j] = Cb[i, j] + Cb_Final[i, j - 8]
                Cr_Final[i, j] = Cr[i, j] + Cr_Final[i, j - 8]
    	            
    return Y_Final, Cb_Final, Cr_Final

def iquantization(Y, Cb, Cr, Q_CbCr, Q_Y):
    height_Y, width_Y = Y.shape
    height_C, width_C = Cb.shape
    YFinal = np.empty((height_Y,width_Y))
    
    CbFinal = np.empty((height_C,width_C))
    CrFinal = np.empty((height_C,width_C))
    
    for i in range(0,height_Y,8):
      for j in range(0,width_Y,8):
         YFinal[i:i+8, j:j+8] = Y[i:i+8, j:j+8] * Q_Y
         if(i < height_C and j < width_C):
              CbFinal[i:i+8, j:j+8] = CbFinal[i:i+8,j:j+8] * Q_CbCr
              CrFinal[i:i+8, j:j+8] = CrFinal[i:i+8,j:j+8] * Q_CbCr
   
    return YFinal, CbFinal, CrFinal
def quantization(Y,Cb,Cr,factor):
    Q_CbCr = np.array([17,18,24,47,99,99,99,99,\
                18,21,26,66,99,99,99,99,    \
                24,26,56,99,99,99,99,99,    \
                47,66,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99]).reshape(8,8)
    
    Q_Y = np.array([16,11,10,16,24,40,51,61,\
                12,12,14,19,26,58,60,55,	\
                14,13,16,24,40,57,69,56,	\
                14,17,22,29,51,87,80,62,	\
                18,22,37,56,68,109,103,77,	\
                24,35,55,64,81,104,113,92,	\
                49,64,78,87,103,121,120,101,\
                72,92,95,98,112,100,103,99]).reshape(8,8)
    if (factor >= 50):
        factorF = (100 - factor) / 50
    else:
        factorF = 50 / factor

    if (factorF  == 0 ):
        Q_CbCr= np.round(Q_CbCr/Q_CbCr)
        Q_Y= np.round(Q_Y/Q_Y)
    else:
        Q_CbCr= np.round(Q_CbCr*factorF)
        Q_Y = np.round(Q_Y*factorF)
    
    Q_CbCr[Q_CbCr>255]=255
    Q_Y[Q_Y>255]=255
    
    Yheight, Ywidth = Y.shape
    Cheight, Cwidth = Cb.shape
    
    Y_Final = np.zeros((Yheight,Ywidth), dtype='int16')
    Cb_Final = np.zeros((Cheight,Cwidth), dtype='int16')
    Cr_Final = np.zeros((Cheight,Cwidth), dtype='int16')
    
    for i in range(0,Yheight,8):
        for j in range(0,Ywidth,8):
            Y_Final[i:i+8, j:j+8] = np.round(Y[i:i+8, j:j+8] / Q_Y)
            if(i < Cheight and j < Cwidth):
                Cb_Final[i:i+8, j:j+8] = np.round(Cb[i:i+8,j:j+8] / Q_CbCr)
                Cr_Final[i:i+8, j:j+8] = np.round(Cr[i:i+8,j:j+8] / Q_CbCr)
    
    return Y_Final, Cb_Final, Cr_Final, Q_CbCr, Q_Y
    
    
    
            



def ex3(image):
    R  = image[:,:,0]
    G  = image[:,:,1]
    B  = image[:,:,2]
    showImageCM(colors_red, 'red', R, 'Red channel')
    showImageCM(colors_green, 'green', G, 'Green channel')
    showImageCM(colors_blue, 'blue', B, 'Blue channel')

    plt.show()

def ex5(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    Y, Cb, Cr = convertYCbCr(R, G, B)
    showImageCM(colors_red, 'red', R, 'Red channel')
    showImageCM(colors_green, 'green', G, 'Green channel')
    showImageCM(colors_blue, 'blue', B, 'Blue channel')
    showImageCM(colors_grayscale, 'gray', Y, 'Y channel')
    showImageCM(colors_grayscale, 'gray', Cb, 'Cb channel')
    showImageCM(colors_grayscale, 'gray', Cr, 'Cr channel')
    plt.show()  

def ex7_1(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    Y, Cb, Cr =  convertYCbCr(R, G, B)
    Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
    Y_dct = dct2(Y)
    Cb_dct = dct2(Cb_ds)
    Cr_dct = dct2(Cr_ds)
    showImageDCT(Y_dct, 'Y_dct ')
    showImageDCT(Cb_dct, 'Cb_dct ')
    showImageDCT(Cr_dct, 'Cr_dct ')
    plt.show()


def unpadding(image,line,column):
    unpaddedImage = image[:line,:column,:]
    return unpaddedImage
    
def padding(image):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    line,column = R.shape
    rest=line%16
    
    if rest!=0:
        paddedR = np.tile(R[len(R)-1],((16-rest),1))
        R = np.vstack((R, paddedR))

        paddedG = np.tile(G[len(G)-1],((16-rest),1))
        G = np.vstack((G, paddedG))

        paddedB = np.tile(B[len(B)-1],((16-rest),1))
        B = np.vstack((B, paddedB))
        
    line,column = R.shape
    
    paddedImage = np.zeros((line, column, 3))
    paddedImage[:, :, 0] = R
    paddedImage[:, :, 1] = G
    paddedImage[:, :, 2] = B
    
    return paddedImage
    
    

def ex4(image):
    line,column = image[:,:,0].shape
    plt.figure()
    plt.title("Original Image")
    print(image.shape)
    plt.imshow(image)
    image=padding(image)
    print(image.shape)
    plt.figure()
    plt.title("Padded Image")
    plt.imshow(image.astype('uint8'))
    image=unpadding(image,line,column)
    print(image.shape)
    plt.figure()
    plt.title("Unpadded Image")
    plt.imshow(image.astype('uint8'))


    
    
"""def upsampling(Y, Cb, Cr, var):
	height, width = Y.shape
	if var == '4:2:0':
		Cb_us = np.repeat(Cb, 2, axis=1)
		Cb_us = np.repeat(Cb_us, 2, axis=0)
		Cr_us = np.repeat(Cr, 2, axis=1)
		Cr_us = np.repeat(Cr_us, 2, axis=0)
	elif d == '4:2:2':
		Cb_us = np.repeat(Cb, 2, axis=1)
		Cr_us = np.repeat(Cr, 2, axis=1)
	

	return Y, Cb_us[0:height, 0:width], Cr_us[0:height, 0:width]    
"""
def upsampling(C1, C2, C3, d):
    height, width = C1.shape
    if d == '4:2:0':
        C2_us = np.repeat(C2, 2, axis=1)
        C2_us = np.repeat(C2_us, 2, axis=0)
        C3_us = np.repeat(C3, 2, axis=1)
        C3_us = np.repeat(C3_us, 2, axis=0)
    elif d == '4:2:2':
        C2_us = np.repeat(C2, 2, axis=1)
        C3_us = np.repeat(C3, 2, axis=1)
    
    print(C1.shape)
    print(C2_us[0:height, 0:width].shape)
    return C1, C2_us[0:height, 0:width], C3_us[0:height, 0:width]

def downsampling(Y, Cb, Cr, var):
    height, width = Y.shape
    if var == '4:2:0':
        Cb = Cb[::2, ::2]
        Cr = Cr[::2, ::2]
    elif var == '4:2:2':
        Cb = Cb[:, ::2]
        Cr = Cr[:, ::2]
    

    return Y, Cb, Cr


    

    
   




    
    

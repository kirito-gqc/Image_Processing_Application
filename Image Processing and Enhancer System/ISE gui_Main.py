from tkinter import CENTER
from turtle import position
import PySimpleGUI as sg
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageFont, ImageDraw, ImageTk ,ImageOps, ImageFilter, ImageEnhance

import cv2

import numpy as np

import textwrap

from matplotlib import pyplot as plt

from guidedfilter import guided_filter

# Preprocessing Message for waiting to open GUI
print("The Image Enhancer GUI will be open within 1 minute")
#Image 1 pre-processing

# Adjust Color
# Reading the image
original_image = cv2.imread('1-7.jpg')

# convert to hsv colorspace
hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

# adjust the hsv and merge together
newH = cv2.add(h, 125)
newS = cv2.add(s, -100)
newV = cv2.add(v, -10)
newHSV = cv2.merge([newH, newS, newV])

# convert back to color
Color_Changed_Image = cv2.cvtColor(newHSV, cv2.COLOR_HSV2RGB)

# save image
cv2.imwrite("1-7 Color Changing.jpg", Color_Changed_Image)


# Add image
Water = Image.open("1-7.jpg").convert("RGBA")
Fish = Image.open("Nemo.png").resize((700,600))
Coral = Image.open("Coral.png").resize((2800,2800))

#study the image
print(Water.format, Water.size, Water.mode)
print(Fish.format, Fish.size, Fish.mode)
print(Coral.format, Coral.size, Coral.mode)

#merge the image
Add_Fish_Image = Water.copy()
Add_Fish_Image.paste(Fish, (int(Water.width*0.20), int(Water.height*0.45)), Fish)
Add_Fish_Coral_Image = Add_Fish_Image.copy()
Add_Fish_Coral_Image.paste(Coral, (int(Water.width*0.55), int(Water.height*0.1)), Coral)

# save image
Add_Fish_Coral_Image.save("1-7 Original Water Add Image.png")

# Different Water Add Image
Water = Image.open("1-7 Color Changing.jpg")
Fish = Image.open("Dead Nemo.png").resize((700,600))
Coral = Image.open("Coral.png").resize((2800,2800))

#study the image
print(Water.format, Water.size, Water.mode)
print(Fish.format, Fish.size, Fish.mode)
print(Coral.format, Coral.size, Coral.mode)

#merge the image
Add_Fish_Image2 = Water.copy()
Add_Fish_Image2.paste(Fish, (int(Water.width*0.20), int(Water.height*0.45)), Fish)
Add_Fish_Coral_Image2 = Add_Fish_Image2.copy()
Add_Fish_Coral_Image2.paste(Coral, (int(Water.width*0.65), int(Water.height*0.15)), Coral)

# save image
Add_Fish_Coral_Image2.save("1-7 Changed Water Add Image.png")


# add title
Add_Title_Image = Add_Fish_Coral_Image
print(Add_Title_Image.format, Add_Title_Image.size, Add_Title_Image.mode)

#adding text 
title_font = ImageFont.truetype("BD_Cartoon_Shout.ttf", 250)
poster_text = "Find The Nemo"

poster = ImageDraw.Draw(Add_Title_Image)

#shadow
poster.text(((Add_Title_Image.width*0.1)+10,(Add_Title_Image.height*0.15)-10), poster_text, (0, 0, 0), font=title_font)
#Text
poster.text((Add_Title_Image.width*0.1,Add_Title_Image.height*0.15), poster_text, (0, 125, 255), font=title_font)

# save image
Add_Title_Image.save("1-7 Original Water Add Title.png")

# changed water add title
Add_Title_Image2 = Add_Fish_Coral_Image2
print(Add_Title_Image2.format, Add_Title_Image2.size, Add_Title_Image2.mode)

#adding text 
title_font = ImageFont.truetype("BD_Cartoon_Shout.ttf", 250)
poster_text = "Save The Nemo"

poster = ImageDraw.Draw(Add_Title_Image2)

#shadow
poster.text(((Add_Title_Image2.width*0.1)+10,(Add_Title_Image2.height*0.15)-10), poster_text, (0, 0, 0), font=title_font)
#Text
poster.text((Add_Title_Image2.width*0.1,Add_Title_Image2.height*0.15), poster_text, (0, 125, 255), font=title_font)

# save image
Add_Title_Image2.save("1-7 Changed Water Add Title.png")

# Enhance image
enhancer = ImageEnhance.Contrast(Add_Title_Image)
factor = 1.5 #gives original image
Enhanced_Image = enhancer.enhance(factor)
Enhanced_Image.save("1-7 Enhanced Image.png")

#==========================================================================================================================
#Image 2 pre-processing

# read image
original_image = cv2.imread("007.jpg")

# reduce noise and save image
Reduced_Noise = cv2.fastNlMeansDenoisingColored(original_image,None,7,10,7,21)
cv2.imwrite("007 Reduced Noise.jpg", Reduced_Noise)

# read image
Reduced_Noise_Image = Image.open("007 Reduced Noise.jpg")

# median filter
x = 1
while x <= 3 :
    Reduced_Noise_Image = Reduced_Noise_Image.filter(ImageFilter.MedianFilter(5))
    x = x + 1

# Add frame
Frame = Image.open("Painting Frame.png").convert("RGBA").resize((880,650))
Paint = Reduced_Noise_Image.convert("RGBA")

#study the image
print(Frame.format, Frame.size, Frame.mode)
print(Paint.format, Paint.size, Paint.mode)

#merge the image
Painting = Frame.copy()
Painting.paste(Paint, (int(Frame.width*0.1), int(Frame.height*0.13)), Paint)

# save image
Painting.save("007 Painting.png")

# Add author
Painting = Image.open("007 Painting.png").convert("RGBA")
Author = Image.open("Author.jpeg").convert("RGBA").resize((200,200))

#study the image
print(Painting.format, Painting.size, Painting.mode)
print(Author.format, Author.size, Author.mode)

#draw on the existing image
mask = Image.new("L", Author.size, 0)
draw = ImageDraw.Draw(mask)

x1 = Author.width*0.05
y1 = Author.height*0.05
x2 = Author.width*0.9
y2 = Author.height*0.9

draw.ellipse((x1,y1,x2,y2), fill=255, outline=150, width=10)

Add_Author_Image = Painting.copy()
#merge the image
Add_Author_Image.paste(Author, (int(Painting.width*0.05), int(Painting.height*0.65)), mask)

# add signature
Add_Signature_Image = Add_Author_Image
print(Add_Signature_Image.format, Add_Signature_Image.size, Add_Signature_Image.mode)

#adding text 
title_font = ImageFont.truetype("Sabrina AT.otf", 30)
poster_text = "Leonardo da Vinci"

poster = ImageDraw.Draw(Add_Signature_Image)

#Text
poster.text((Add_Signature_Image.width*0.15,Add_Signature_Image.height*0.3), poster_text, (0, 0, 0), font=title_font)

# save image
Add_Signature_Image.save("007 Add Author.png")

#========================================================================================================================== 
#Image 3 pre-processing
 #g.png
img_boat= Image.open('anime.png').convert("RGB")

original3 = cv2.imread("g.png")
hsv = cv2.cvtColor(original3, cv2.COLOR_BGR2HSV)
h = hsv[: ,: , 0]
s = hsv[: ,: , 1]
v = hsv[: ,: , 2]

# shift the hue
# cv2 will clip automatically to avoid color wrap - around
huechange = 170 # 0 is no change;
schange = 30
vchange = 20
0 <= huechange <= 180
hnew = cv2.add(h, huechange)
snew = cv2.add(s, schange)
vnew = cv2.add(v, vchange)

# combine new hue with s and v
hsvnew = cv2.merge([hnew, snew, vnew])

# convert from HSV to BGR
result = cv2.cvtColor(hsvnew, cv2.COLOR_HSV2BGR)

cv2.imwrite('g_changed.png', result)

img = Image.open('g_changed.png').convert("RGB") 
img_flip = ImageOps.mirror(img) 

new_img3 = Image.new('RGB', (img.width*2,img.height)) 
w,h = img_boat.width,img_boat.height

shape = [(60,40), (w-10,h-10)]
mask_im = Image.new("L", img_boat.size, 0)
draw = ImageDraw.Draw(mask_im)
draw.ellipse(shape, fill=255)
mask_im.save('mask_g.jpg', quality=95)
mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(10))
mask_im_blur.save('mask_circle_blur.jpg', quality=95)

new_img3.paste(img, (0, 0)) 

new_img3.paste(img_flip, (img.width, 0)) 

new_img3.save('g_enhance.jpg',quality=95)
word_img3 = new_img3.copy()
astr = "Welcome to the river"
para = textwrap.wrap(astr, width=15)

MAX_W, MAX_H = 1280,940
poster = ImageDraw.Draw(word_img3)
font = ImageFont.truetype(
    'ALGER.ttf', 120)

current_h, pad = 200,10
for line in para:
    w, h = draw.textsize(line, font=font)
    poster.text((((MAX_W - w) / 2)+10,current_h-10), line, ((0, 85, 100)), font=font) 
    poster.text(((MAX_W - w) / 2, current_h), line,(121, 218, 110), font=font)
    current_h += h + pad

word_img3.save('g_text.jpg',quality=95)

back_im = new_img3.copy()
back_im.paste(img_boat,(800,700),mask_im_blur)
back_im.save('g_effect.jpg',quality = 95)
astr = "Welcome to the river"
para = textwrap.wrap(astr, width=15)

MAX_W, MAX_H = 1280,940
poster = ImageDraw.Draw(back_im)
font = ImageFont.truetype(
    'ALGER.ttf', 120)

current_h, pad = 200,10
for line in para:
    w, h = draw.textsize(line, font=font)
    poster.text((((MAX_W - w) / 2)+10,current_h-10), line, ((0, 85, 100)), font=font) 
    poster.text(((MAX_W - w) / 2, current_h), line,(121, 218, 110), font=font)
    current_h += h + pad

back_im.save('g_effectntext.jpg',quality=95)

#==========================================================================================================================

# Image 4 preprocessing
# 3-7.jpg
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(img4, clip_hist_percent=1):
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.ylim([0,100000])
    plt.savefig('img4_plot.png', dpi = 300, bbox_inches = 'tight')

    auto_result = cv2.convertScaleAbs(img4, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

img4 = cv2.imread('3-7.jpg')
auto_result, alpha, beta = automatic_brightness_and_contrast(img4)

cv2.imwrite('enhanced_4.jpg', auto_result)
img4 = Image.open('enhanced_4.jpg')
title_font = ImageFont.truetype("Saber.ttf", 300) 

poster_text = "So bright" 

 

poster = ImageDraw.Draw(img4) 

 

#3d effect
for index in range(10,0,-1):
     poster.text(((img4.width*0.1)+index,(img4.height*0.15)-index), poster_text, (0, 0, 0), font=title_font) 

#Text 

poster.text((img4.width*0.1,img4.height*0.15), poster_text, (218, 243, 0), font=title_font) 

img4.save('img4_text.jpg',quality=95)

#==========================================================================================================================

#Image 5 pre-processing

def get_illumination_channel(I, w):
    #image dimensions are stored in the variable m and n 
    M, N, _ = I.shape 
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge') 
    
    # Padding of half kernel size is applied to ensure image sizes are still same.
    darkChannel = np.zeros((M, N))
    brightChannel = np.zeros((M, N))

    for i, j in np.ndindex(darkChannel.shape):
        # dark channel obtained using np.min to get lowest pixel value 
        darkChannel[i, j] = np.min(padded[i:i + w, j:j + w, :]) 
        # bright channel using np.max to get highest pixel value
        brightChannel[i, j] = np.max(padded[i:i + w, j:j + w, :]) 

    return darkChannel, brightChannel

def get_atmosphere(I, brightChannel, p=0.1):
    # Step 2 : 
    #getting the atmophere lighting 
    #is obtained by taking mean of top 10% intensity of bright channel value
    M, N = brightChannel.shape
    flatI = I.reshape(M*N, 3)  # reshaping the image array
    flatbright = brightChannel.ravel() # flattening image array

    # sorting and slicing
    # slicing to include only 10% of pixels then the mean of sliced taken
    searchidx = (-flatbright).argsort()[:int(M*N*p)]
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

def get_initial_transmission(A, brightChannel):
    # Step 3 :
    # finding transmission map
    # the portion of light that is not scattered and reach the camera
    # using the bright channel
    A_c = np.max(A)
    InitialTransMap = (brightChannel-A_c)/(1.-A_c) # finding initial transmission map
    return (InitialTransMap - np.min(InitialTransMap))/(np.max(InitialTransMap) - np.min(InitialTransMap))

def get_correctedTransMapransmission(I, A, darkChannel, brightChannel, InitialTransMap, alpha, omega, w):
    #Step 4
    # using dark channel to estimate corrected transmission map
    # to corrected potential erroneous transmission that obtain from bright channel
    
    im3 = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im3[:, :, ind] = I[:, :, ind] / A[ind] #divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel(im3, w) # dark channel transmission map
    dark_t = 1 - omega*dark_c # corrected dark transmission map
    correctedTransMap = InitialTransMap # initializing corrected transmission map with initial transmission map
    differenceChannel = brightChannel - darkChannel # difference between transmission maps

    for i in range(differenceChannel.shape[0]):
        for j in range(differenceChannel.shape[1]):
            if(differenceChannel[i, j] < alpha):
                correctedTransMap[i, j] = dark_t[i, j] * InitialTransMap[i, j]

    return np.abs(correctedTransMap)

def get_final_image(I, A, refined_t, tmin):
    
    # Step 5 : Smoothing Transmission Map using guided filter 
    # Step 6 : Calc the resultant image
    
    # duplicating the channel of 2D refined map to 3 channels
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
    # finding result
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A
    # normalized image
    return (J - np.min(J))/(np.max(J) - np.min(J))

def dehaze(I, tmin, w, alpha, omega, p, eps, reduce=False):
    
    # Step 7 : Combine all the techniques and pass it as an image
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    InitialTransMap = get_initial_transmission(A, Ibright) 
    if reduce:
        InitialTransMap = reduce_InitialTransMap(InitialTransMap)
    correctedTransMap = get_correctedTransMapransmission(I, A, Idark, Ibright, InitialTransMap, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, correctedTransMap, w, eps)
    J_refined = get_final_image(I, A, refined_t, tmin)
    
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

def reduce_InitialTransMap(InitialTransMap):
    InitialTransMap = (InitialTransMap*255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    InitialTransMap = cv2.LUT(InitialTransMap, table)
    InitialTransMap = InitialTransMap.astype(np.float64)/255
    return InitialTransMap

im = cv2.imread('lowlight.png')
orig = im.copy()

tmin = 0.1   # minimum value for t to make J image
w = 15       # window size, which determine the corseness of prior images
alpha = 0.4  # threshold for transmission correction
omega = 0.75 # this is for dark channel prior
p = 0.1      # percentage to consider for atmosphere
eps = 1e-3   # for J image

I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
I = I[:, :, :3] / 255

f_enhanced = dehaze(I, tmin, w, alpha, omega, p, eps)
f_enhanced2 = dehaze(I, tmin, w, alpha, omega, p, eps, True)
cv2.imwrite('original.png', orig)
cv2.imwrite('lowlight_v1.png', f_enhanced)
cv2.imwrite('lowlight_v2.png', f_enhanced2)

#adding text
imgOri = Image.open('lowlight_v2.png')
titlefont = ImageFont.truetype("BRITANIC.ttf", 30) 
poster1 = ImageDraw.Draw(imgOri) 
poster1.text((80, 30), "Enchanced Image", font=titlefont, fill =(255, 255, 255)) 
imgOri.save('Text_original.png',quality=95)


#adding image
Image1 = Image.open('original.png')
Image1copy = Image1.copy()
Image2 = Image.open('moon.png')
Image2copy = Image2.copy() 
Image1copy.paste(Image2copy, (10, 10))
Image1copy.save('PicImage.png')

#==========================================================================================================================
#GUI
#Image 1 GUI
def Image1():
    seg1 = [[sg.Button("<< Back to Menu", key = "back1")],
            [sg.Text("Image 1")], 
            [sg.Button("Load Image")], 
            [sg.Text("Select an Operation:\n")], 
            [sg.Button('Change Water Colour',  key="-colour1-")], 
            [sg.Button('Add Image', key="-addimage1-")],
            [sg.Button('Add Title', key ="-title1-")],
            [sg.Button('All with Original Water', key ="-all1-")],
            [sg.Button('Enhance contrast', key ="-contrast1-")]
            ] 
    
    seg2 = [[sg.Text("OUTPUT")], 
            [sg.Text("Image 1:")], 
            [sg.Image(key="-IMAGE-")]
    ]

    layout2 = [[ 

        sg.Column(seg1), 

        sg.VSeperator(), 

        sg.Column(seg2), 

    ]] 
  
    window = sg.Window("Image 1", layout2,modal = True, size=(800, 400)) #.read() 
    
    while True: 

         event, values = window.read() 
         
         if event == "back1" or event == sg.WIN_CLOSED:
                break
     
         elif event == "Load Image": 

                image = Image.open('1-7.jpg') 

                image.thumbnail((500, 279)) 

         elif event == "-colour1-":

                image = Image.open('1-7 Color Changing.jpg')
                image.thumbnail((500, 279)) 
                
         elif event == "-addimage1-":

                image = Image.open('1-7 Changed Water Add Image.png')
                image.thumbnail((500, 279)) 

         elif event == "-title1-": 

                image = Image.open('1-7 Changed Water Add Title.png')
                image.thumbnail((500, 279)) 
                
         elif event == "-all1-": 

                image = Image.open('1-7 Original Water Add Title.png')
                image.thumbnail((500, 279)) 
        
         elif event == "-contrast1-": 

                image = Image.open('1-7 Enhanced Image.png')
                image.thumbnail((500, 279)) 
                
         else:
                break

         photo_img = ImageTk.PhotoImage(image)       

         window["-IMAGE-"].update(data=photo_img) 
    
    window.close()

#Image 2 GUI
def Image2():
    seg1 = [[sg.Button("<< Back to Menu", key = "back2")],
            [sg.Text("Image 2")], 
            [sg.Button("Load Image")], 
            [sg.Text("Select an Operation:\n")], 
            [sg.Button('Noise Reduction',  key="-noise2-")], 
            [sg.Button('Become a Painting', key="-painting2-")],
            [sg.Button('Add Author', key ="-author2-")]
            ] 
    
    seg2 = [[sg.Text("OUTPUT")], 
            [sg.Text("Image 2:")], 
            [sg.Image(key="-IMAGE-")]
    ]

    layout2 = [[ 

        sg.Column(seg1), 

        sg.VSeperator(), 

        sg.Column(seg2), 

    ]] 
  
    window = sg.Window("Image 2", layout2,modal = True, size=(800, 400)) #.read() 
    
    while True: 

         event, values = window.read() 
         
         if event == "back2" or event == sg.WIN_CLOSED:
                break
     
         elif event == "Load Image": 

                image = Image.open('007.jpg') 

                image.thumbnail((500, 279)) 

               

         elif event == "-noise2-":

                image = Image.open('007 Reduced Noise.jpg')
                image.thumbnail((500, 279)) 
                
         elif event == "-painting2-":

                image = Image.open('007 Painting.png')
                image.thumbnail((500, 279)) 

         elif event == "-author2-": 

                image = Image.open('007 Add Author.png')
                image.thumbnail((500, 279)) 
         else:
                break

         photo_img = ImageTk.PhotoImage(image)       

         window["-IMAGE-"].update(data=photo_img) 
    
    window.close()
    

# Image 3 GUI
def Image3():
    
    seg1 = [[sg.Button("<< Back to Menu", key = "back3")],
            [sg.Text("Image 3")], 
            [sg.Button("Load Image")], 
            [sg.Text("Select an Operation:\n")], 
            [sg.Button('Image enhance',  key="-enhance3-"), 
             sg.Button('Image add effect', key="-effect3-")],
            [sg.Button('Image add word', key ="-word3-"),
             sg. Button('Image add effect and word',  key="-mix3-")] 
            ] 
    
    seg2 = [[sg.Text("OUTPUT")], 
            [sg.Text("Image 3:")], 
            [sg.Image(key="-IMAGE-")]
    ]

    layout3 = [[ 

        sg.Column(seg1), 

        sg.VSeperator(), 

        sg.Column(seg2), 

    ]] 
  
    window = sg.Window("Image 3", layout3,modal = True, size=(800, 400)) #.read() 
    
    while True: 

         event, values = window.read() 
         
         if event == "back3" or event == sg.WIN_CLOSED:
                break
     
         elif event == "Load Image": 

                image = Image.open('g.png') 

                image.thumbnail((400, 300)) 

               

         elif event == "-enhance3-":

                image = Image.open('g_enhance.jpg')
                image.thumbnail((400, 300)) 
                
         elif event == "-effect3-":

                image = Image.open('g_effect.jpg')
                image.thumbnail((400, 300)) 

         elif event == "-word3-": 

                image = Image.open('g_text.jpg')
                image.thumbnail((400, 300)) 
                
         elif event == "-mix3-": 

                image = Image.open('g_effectntext.jpg')
                image.thumbnail((400, 300)) 
         else:
                break

         photo_img = ImageTk.PhotoImage(image)       

         window["-IMAGE-"].update(data=photo_img) 
    
    window.close()

#Image 4 GUI
    
def Image4():
        
    seg1 = [[sg.Button("<< Back to Menu", key = "back4")],
            [sg.Text("Image 4")], 
            [sg.Button("Load Image")], 
            [sg.Text("Select an Operation:\n")], 
            [sg.Button('Image enhance',  key="-enhance4-"), 
             sg.Button('Image add effect', key="-effect4-")],
            [sg.Text("Histogram normalization:\n")],
             [sg. Button('Show Histogram ',  key="-hist4-")] ,
             [sg.Text("alpha(gain): "), sg.Text(key="alpha")],
             [sg.Text("beta(bias): "), sg.Text(key="beta")]
            ] 
    
    seg2 = [[sg.Text("OUTPUT")], 
            [sg.Text("Image 4:")], 
            [sg.Image(key="-IMAGE-")]
    ]

    layout4 = [[ 

        sg.Column(seg1), 

        sg.VSeperator(), 

        sg.Column(seg2), 

    ]] 
  
    window = sg.Window("Image 4", layout4,modal = True, size=(800, 400)) #.read() 
    
    while True: 

         event, values = window.read() 
         
         if event == "back4" or event == sg.WIN_CLOSED:
                break
     
         elif event == "Load Image": 

                image = Image.open('3-7.jpg') 

                image.thumbnail((450, 300)) 

               

         elif event == "-enhance4-":

                image = Image.open('enhanced_4.jpg')
                image.thumbnail((450, 300)) 
                
         elif event == "-effect4-":

                image = Image.open('img4_text.jpg')
                image.thumbnail((450, 300)) 
                
         elif event == "-hist4-": 
                window['alpha'].Update(alpha)
                window['beta'].Update(beta)

                image = Image.open('img4_plot.png')
                image.thumbnail((450, 300)) 
         else:
                break

         photo_img = ImageTk.PhotoImage(image)       

         window["-IMAGE-"].update(data=photo_img) 
    
    window.close()

#Image 5 GUI
def Image5():
    section1 = [[sg.Button("<< Back to Menu", key = "back4")],
            [sg.Text("Image 5")], 
            [sg.Button("Load Image")], 
            [sg.Text("Select an Operation:\n")], 
            [sg.Button('Original Image',  key="-origImage-"), 
             sg.Button('Corrected transmission map', key="-afterTransMap-")],
             [sg.Button('Final Image ', key="-Afterfilter-"),
             sg.Button('Add Text ', key="-text-"),
             sg.Button('Add Image ', key="-picture-")]
            
            ] 
    
    section2 = [[sg.Text("OUTPUT")], 
            [sg.Text("Image 5:")], 
            [sg.Image(key="-IMAGE-")]
    ]

    layout01 = [[ 

        sg.Column(section1), 

        sg.VSeperator(), 

        sg.Column(section2), 

    ]] 
  
    window = sg.Window("Image 5", layout01,modal = True, size=(800, 400)) #.read() 
    
    while True: 

         event, values = window.read() 
         
         if event == "back4" or event == sg.WIN_CLOSED:
                break
     
         elif event == "Load Image": 

                image = Image.open('lowlight.png') 

                image.thumbnail((450, 300)) 

               

         elif event == "-origImage-":

                image = Image.open('lowlight.png')
                image.thumbnail((450, 300)) 
                
         elif event == "-afterTransMap-":

                image = Image.open('lowlight_v1.png')
                image.thumbnail((450, 300)) 
                
         elif event == "-Afterfilter-": 

                image = Image.open('lowlight_v2.png')
                image.thumbnail((450, 300)) 
         elif event == "-text-": 
                image = Image.open('Text_original.png')
                image.thumbnail((450, 300))
         elif event == "-picture-": 
                image = Image.open('PicImage.png')
                image.thumbnail((450, 300))                                   
         else:
                break

         photo_img = ImageTk.PhotoImage(image)       

         window["-IMAGE-"].update(data=photo_img) 
    
    window.close()


#Main Menu GUI
sg.theme('SystemDefault')

layout = [  [sg.Text('This is the main screen Menu')],
            [sg.Button('Image 1', font=('Arial',12), key="img1")],
            [sg.Button('Image 2', font=('Arial',12), key="img2")],
            [sg.Button('Image 3', font=('Arial',12), key="img3")],
            [sg.Button('Image 4', font=('Arial',12), key="img4")],
            [sg.Button('Image 5', font  = ('Arial',12), key="img5")],
             [sg.Button('Exit',key='exit')],
            ]
window = sg.Window('Image Enhancer', layout, size=(400, 400))

#event
while True:             
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'img1':
        window.hide()
        Image1()
        window.un_hide()
    elif event == 'exit':
        break
    elif event == 'img2':
        window.hide()
        Image2()
        window.un_hide()
    elif event =='img3':
        window.hide()
        Image3()
        window.un_hide()
    elif event == 'img4':
        window.hide()
        Image4()
        window.un_hide()
    elif event == 'img5':
        window.hide()
        Image5()
        window.un_hide()
        
window.close()



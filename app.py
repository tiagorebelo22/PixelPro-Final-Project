# importing the library

import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from  PIL import Image
from datetime import datetime
import os
import time

from streamlit_image_select import image_select
from streamlit_image_comparison import image_comparison
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="PixelPro",
    page_icon=":frame_with_picture:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """
    Defining the main page of the app
    """
    with st.sidebar:
        selected = option_menu("Navigation Menu", ["Picture Generator", "Picture Gallery"], 
            icons=["gear", "images"], menu_icon="cast", default_index=0)
        st.markdown("<p style='text-align: center;font-size:15px'><b>© 2023 PixelPro </b></p>", unsafe_allow_html=True)

    if selected == "Picture Gallery":
        picture_gallery()
    else:
        picture_generator()

def model(start_time,scale,uploaded_file):
    """
    Funtion that outputs a high resolution picture, using the input scale and uploaded picture.
    The start time is used to define the directory in which the generated files will be saved.
    """

    def PSNR(super_resolution, high_resolution):
        """
        Compute the peak signal-to-noise ratio, measures quality of image.
        """
        # Max value of pixel is 255
        psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
        return psnr_value
    
    def predict_step(model, x):
        """
        Predicts the high resolution image from the low resolution image 
        """

        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        # This step is necessary because the model was trained with 4D tensors (batch, height, width, color)
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        # Passing low resolution image to model
        super_resolution_img = model.predict(x)
        # Clips the tensor from min(0) to max(255)
        # If the RGB code is below 0 or above 255, it will be set as 0 or 255, respectively 
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # Rounds the values of a tensor to the nearest integer, because RGB codes are integers
        super_resolution_img = tf.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor (to remove the dummy dimension) and converts to uint8
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0)
        return super_resolution_img

    # Considering the scale, it loads the corresponding model
    model = tf.keras.models.load_model(f"Models/EDSR_{scale}x.tf", 
                                       custom_objects={"PSNR":PSNR})
    
    # Directory in which the files will be saved
    path = "Picture_Gallery/"+start_time.strftime("%Y%m%d_%H%M%S")

    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the original/uploaded picture
    file_in_bytes = uploaded_file.getvalue()

    with open(path+"/original.png", 'wb') as f: 
        f.write(file_in_bytes)

    # Convert uploaded picture to tensor format
    lowres = tf.io.decode_image(file_in_bytes, channels=3, dtype=tf.dtypes.uint8)
    
    # Predict high resolution picture
    preds = predict_step(model,lowres)

    # Save high resolution picture
    tf.keras.utils.save_img(path+"/x"+str(scale)+".png",preds)


def picture_generator():
    """
    Defining the Picture Generator page of the app
    """    

    # PixelPro & Logo
    col1,col2 = st.columns([0.9,1])

    with col1: 
        st.markdown("<p style='text-align: left;font-family:PixelSplitter;font-size:80px'><b>PixelPro</b></p>", unsafe_allow_html=True)

    with col2:
        st.write("")
        st.image("Logo.png", width=100)

    # Uploading a picture

    st.subheader("Please upload a picture...")

    col3,_,col4 = st.columns([3,0.5,1])

    with col3:
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

    with col4:
        upscale_factor = st.radio("**Scaling factor**",["x2","x4","x8","All"])
        if upscale_factor == "x2":
            scale_list = [2]
        elif upscale_factor == "x4":
            scale_list = [4]
        elif upscale_factor == "x8":
            scale_list = [8]
        else:
            scale_list = [2,4,8]

    st.write("---")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col5, col6, col7 = st.columns([2, 1.5,0.5])

        # Picture preview
        with col5:
            st.subheader("Uploaded picture")
            st.image(image,width=400)

        # Picture specifications
        with col6:
            st.subheader("Current specifications:")
            w, h = image.size
            st.write("Size: ",w,"x",h)
            st.subheader("Future specifications:")
            for scale in scale_list: 
                scale_string = "(x"+str(scale)+"):"
                st.write("Size",scale_string,w*scale,"x",h*scale)

        # Button Generate
        with col7:
            button_pixelup = st.button("Generate!")

        if button_pixelup == 1:

            # Start time is used for the name of the directory in which the generated files are saved
            start_time = datetime.now()

            with st.spinner(text="Generating your pictures... :hourglass_flowing_sand:"):
                # Defining progress bar
                my_bar = st.progress(0)
                percent_complete = 0
                
                # Generate pictures for each scale
                for scale in scale_list:
                    my_bar.progress(percent_complete)
                    model(start_time,scale,uploaded_file)
                    percent_complete =+ int(100/len(scale_list))

            # Display 100% completion and success message
            my_bar.progress(100)
            time.sleep(0.5)
            my_bar.empty()
            st.success("Success! Please check the Picture Gallery.", icon="✅")
            


def crop_image(path_lowres_img, path_highres_img, snippet_size, scale, pixel_lowres_width=0, pixel_lowres_height=0):
    """
    Cropping images considering the center of the cropped area and the snippet size.
    The center of the cropped area will have pixel_lowres_width and pixel_lowres_height as coordinates.
    """

    # Converting the low resolution image to tensor format
    lowres_img = tf.io.read_file(path_lowres_img)
    lowres_img = tf.io.decode_image(lowres_img, channels=3, dtype=tf.dtypes.uint8)

    # Converting the high resolution image to tensor format
    highres_img = tf.io.read_file(path_highres_img)
    highres_img = tf.io.decode_image(highres_img, channels=3, dtype=tf.dtypes.uint8)

    # Defining width and height of the low resolution image
    lowres_img_width = tf.shape(lowres_img)[1]
    lowres_img_height = tf.shape(lowres_img)[0]

    # Calculate half of snippet size, as the coordinates will be the center of the cropped area
    half_size = int(snippet_size/2)

    # Dealing with out of bounds situations (the coordinates are too close to the edges of the image)
    if pixel_lowres_width + half_size > lowres_img_width: # too close to the right edge
        pixel_lowres_width = lowres_img_width - half_size
    elif pixel_lowres_width - half_size < 0: # too close to the left edge
        pixel_lowres_width = half_size

    if pixel_lowres_height + half_size > lowres_img_height: # too close to the bottom edge
        pixel_lowres_height = lowres_img_height - half_size
    elif pixel_lowres_height - half_size < 0: # too close to the top edge
        pixel_lowres_height = half_size

    highres_width = pixel_lowres_width * scale # Defining the starting pixel to slice the width for the highres image
    highres_height = pixel_lowres_height * scale # Defining the starting pixel to slice the height for the highres image

    # Cropping the images
    lowres_img_cropped = lowres_img[
        pixel_lowres_height - half_size : pixel_lowres_height + half_size, # Slicing the height
        pixel_lowres_width - half_size: pixel_lowres_width + half_size # Slicing the width
    ]

    highres_img_cropped = highres_img[
        highres_height - half_size*scale : highres_height + half_size*scale, # Slicing the height
        highres_width - half_size*scale : highres_width + half_size*scale # Slicing the width
    ]

    # Converting tensors to images 
    lowres_img_cropped = tf.keras.preprocessing.image.array_to_img(lowres_img_cropped)
    highres_img_cropped = tf.keras.preprocessing.image.array_to_img(highres_img_cropped)

    return lowres_img_cropped, highres_img_cropped


def picture_gallery():
    """
    Defining the Picture Gallery page of the app
    """

    st.title("Picture Gallery")

    # Creates list of timestamps, if at least one .png file was generated
    directory="Picture_Gallery"
    timestamps = [img_folder.name for img_folder in os.scandir(directory) if 
                  (os.path.isdir(img_folder) and any(file.path.endswith(".png") for file in os.scandir(img_folder.path)))]
    
    # Orders list of timestamps by most recent
    timestamps = reversed(sorted(timestamps))

    # Creates a dictionary with timestamps in the format "YY-MM-dd HH:mm:ss" as keys and the original timestamps as values 
    timestamps = {(datetime.strptime(timestamp,"%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")):timestamp for timestamp in timestamps}

    # Creates select box with available timestamps
    selected_timestamp = st.selectbox("**Choose a timestamp**", timestamps.keys())
    selected_files_dir = os.path.join(directory,timestamps[selected_timestamp])
    selected_files = os.scandir(selected_files_dir)

    # Obtains the images corresponding to the selected timestamp
    files = [file.path for file in selected_files if file.path.endswith(".png")]
    files = sorted(files)
    
    col1,col2 = st.columns([5,1.2])

    # Defines the image selection section
    with col1:
        with st.spinner(text="In progress..."):
            img = image_select(
                label="",
                images=files,
                captions=[os.path.basename(file) for file in files],
                use_container_width=False
                )
    
    # Defines the download button (triggers the download of the selected image)
    with col2:
        st.subheader("") # blank line
        st.subheader("") # blank line 
        st.subheader("") # blank line 
        st.subheader("") # blank line 
        with open(img, "rb") as file:
            btn = st.download_button(
                 label="Download picture",
                 data = file,
                 file_name = os.path.basename(img),
                 mime="image/png"
                 )


    if img is not None:

        # Define scale of selected image
        if os.path.basename(img) == "original.png":
            scale=1
        elif os.path.basename(img) == "x2.png":
            scale=2
        elif os.path.basename(img) == "x4.png":
            scale=4
        else:
            scale=8
        
        col3,_,col4 = st.columns([3,0.5,1])

        # Defines the zoom area coordinate system, zoom percentage and image specifications
        with col4:
            st.subheader("") # blank line
            st.subheader("") # blank line 
            st.markdown("<p style='text-align: center;'><b>Zoom Area<br>(click on image)</b></p>", unsafe_allow_html=True)

            # Defines the zoom area coordinates system
            path_original_img = os.path.join(selected_files_dir,"original.png") # Define the path of the original image in the selected timestamp directory
            original_img = Image.open(path_original_img) # Opens the original image

            ratio = 190 / original_img.width # The column has a width of 190px
            newsize = (190, int(original_img.height*ratio))
            img_resized = original_img.resize(newsize) # Resizes the original image to have a width of 190px
            coordinates = streamlit_image_coordinates(img_resized, key="pil") # Creates the coordinate system image

            # The default coordinates are (0,0)
            if coordinates is None:
                coordinates = {"x":0,"y":0}

            # Defines the zoom percentage and snippet size
            st.markdown("<p style='text-align: center;'><b>Zoom</b></p>", unsafe_allow_html=True)
            zoom = st.slider("**Zoom**",0,100,50,5, format="%d%%", label_visibility="collapsed") # Zoom goes from 0% to 100%, in steps of 5%, and the default value is 50% 
            
            max_size = min(original_img.width,original_img.height) # Define the maximum size of the image comparison, which has a square shape
            snippet_size = int(max_size*(1-zoom/100))

            if snippet_size < 2:
                snippet_size = 2 # Minimum of 2 pixels of width and height 

            # Defines the picture specifications
            st.markdown("<p style='text-align: center;'><b>Specifications</b></p>", unsafe_allow_html=True)

            w, h = original_img.size
            st.write("Size (original): ",w,"x",h)
            if scale != 1:
                scale_string = "(x"+str(scale)+"):"
                st.write("Size",scale_string,w*scale,"x",h*scale)

        # Defines the picture comparison section
        with col3:
            st.subheader("Picture Comparison")

            path_img2 = img # Defines the path of the selected image
            center_pixel_width = int(coordinates["x"]/ratio) # Defines the width center pixel for the cropping area, considering that the coordinate system image had been resized with a factor equal to 'ratio'
            center_pixel_height = int(coordinates["y"]/ratio) # Defines the height center pixel for the cropping area, considering that the coordinate system image had been resized with a factor equal to 'ratio'

            img1, img2 = crop_image(path_original_img, path_img2, snippet_size, scale, center_pixel_width, center_pixel_height) # Crop image

            # Create the picture comparison
            image_comparison(
                img1=img1,
                img2=img2,
                label1="original.png",
                label2=os.path.basename(img)
                )

# Run the app
if __name__ == '__main__':
    main()



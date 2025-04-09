import streamlit as st
from PIL import Image
import numpy as np

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import construct_mip as mip
from ultralytics import YOLO
import torch
import cv2
from PIL import Image, ImageDraw
from model import TResUnet

from io import BytesIO

path_to_frontal ="/Users/ghazalfallahpour/ghazal-ui/frontal_best_XL_epoch100_batch-1_patience50_auto.pt"
path_to_sagittal = "/Users/ghazalfallahpour/ghazal-ui/sag_best_XL_epoch100_batch-1_patience50_auto.pt"
path_to_seg_weghts = "/Users/ghazalfallahpour/ghazal-ui/segmentation-weights.pth"


def mip_to_normalized_slice_frontal(frontal_mip):
    
    mip_rgb = mip_array_to_rgb(frontal_mip)

    mip_rgb.save("temp_mip_frontal.png")
    model_frontal = YOLO(path_to_frontal)


    yolo_results = model_frontal("temp_mip_frontal.png")

    # Y-Level from Bounding Boxes
    boxes = yolo_results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        st.error("No L3 detection found.")
        st.stop()

    # Calculate predicted y-slice index from YOLO boxes
    sum_y = 0
    count = 0

    # Copy MIP image for drawing boxes
    mip_predicted = mip_rgb.copy()
    draw = ImageDraw.Draw(mip_predicted)

    for box in boxes:
        cls = int(box.cls.tolist()[0])
        xywh = box.xywh.tolist()[0]
        x_center, y_center, w, h = xywh
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Choose color by class
        if cls == 0:
            color = "#E66100"  # Orange
            sum_y += y_center + (h / 2)
            count += 1
        elif cls == 1:
            color = "#5D3A9B"  # Purple
            sum_y += y_center - (h / 2)
            count += 1
        else:
            color = "white"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Draw dashed horizontal line at predicted L3 location
    if count > 0:
        pred_y_pixel = sum_y / count
        line_y = int(pred_y_pixel)
        image_width = mip_predicted.width
        dash_length = 10
        gap = 5
        for x in range(0, image_width, dash_length + gap):
            draw.line([(x, line_y), (x + dash_length, line_y)], fill="#0072ed", width=2)

        # Convert y-pixel to CT slice index
        mip_height = frontal_mip.shape[0]
        ct_depth = ct_array.shape[0]
        predicted_slice_index = int(pred_y_pixel / mip_height * ct_depth)
    else:
        st.error("Could not determine predicted slice index from YOLO outputs.")
        st.stop()

    # Assign updated prediction image
    frontal_mip_prediction = mip_predicted
    
    # === Step 3: Extract slice ===
    l3_slice = extract_single_l3_slice("temp_ct.nii.gz", predicted_slice_index)

    # Normalize to 0–255 for display
    l3_slice_norm = ((l3_slice - l3_slice.min()) / (l3_slice.max() - l3_slice.min()) * 255).astype(np.uint8)
    
    return l3_slice_norm, frontal_mip_prediction

def mip_to_normalized_slice_sagittal(saggital_mip):
    mip_rgb = mip_array_to_rgb(saggital_mip) 
    
    model_sagittal = YOLO(path_to_sagittal)
    mip_rgb.save("temp_mip_sagittal.png")

    # Run YOLO prediction
    yolo_results = model_sagittal("temp_mip_sagittal.png")

    boxes = yolo_results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        st.error("No vertebrae detected.")
        st.stop()

    # Copy MIP image for drawing boxes
    mip_predicted = mip_rgb.copy()
    draw = ImageDraw.Draw(mip_predicted)

    y_centers = []
    for box in boxes:
        xywh = box.xywh.tolist()[0]
        x_center, y_center, w, h = xywh
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        y_centers.append(y_center)
        draw.rectangle([x1, y1, x2, y2], outline="white", width=3)

    if len(y_centers) < 3:
        st.error("Not enough vertebrae detected to identify L3.")
        st.stop()

    # Sort y-centers from bottom to top (larger y = lower in image)
    y_centers_sorted = sorted(y_centers, reverse=True)
    pred_y_pixel = y_centers_sorted[3]  # 3rd from bottom

    # Draw dashed horizontal line at predicted L3 location
    line_y = int(pred_y_pixel)
    image_width = mip_predicted.width
    dash_length = 10
    gap = 5
    for x in range(0, image_width, dash_length + gap):
        draw.line([(x, line_y), (x + dash_length, line_y)], fill="#0072ed", width=2)

    # Convert y-pixel to CT slice index
    mip_height = saggital_mip.shape[0]  # was incorrectly using frontal_mip
    ct_depth = ct_array.shape[0]        # ensure ct_array is defined in scope
    predicted_slice_index = int(pred_y_pixel / mip_height * ct_depth)

    # === Step 3: Extract slice ===
    l3_slice = extract_single_l3_slice("temp_ct.nii.gz", predicted_slice_index)

    # Normalize to 0–255 for display
    l3_slice_norm = ((l3_slice - l3_slice.min()) / (l3_slice.max() - l3_slice.min()) * 255).astype(np.uint8)
    
    return l3_slice_norm, mip_predicted


def window_and_normalize(slice_2d, level, window):
    min_val = level - window // 2
    max_val = level + window // 2
    windowed = np.clip(slice_2d, min_val, max_val)
    normalized = ((windowed - windowed.min()) / (windowed.max() - windowed.min()) * 255).astype(np.uint8)
    return Image.fromarray(normalized)

def image_to_bytes(img):
    """Convert an image (PIL Image or numpy array) to bytes for download"""
    # Convert NumPy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        # Convert BGR to RGB if it's a color image (OpenCV often returns BGR)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ensure data type is uint8
        if img.dtype != np.uint8:
            img = (img * 255 if img.max() <= 1.0 else img).astype(np.uint8)
            
        # Create PIL Image
        img = Image.fromarray(img)
    
    # Now img should be a PIL Image, proceed with saving
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def display_prediction_block(title, images, file_prefix="output"):
    """
    Display labeled images in a tabbed block with optional download buttons.
    
    Parameters:
        title (str): Section title (e.g. "Frontal Predictions")
        images (dict): Dictionary of {'label': PIL.Image or numpy array}
        file_prefix (str): Prefix for downloaded file names
    """
    st.header(title)
    with st.expander(f"View {title}", expanded=True):
        tabs = st.tabs(list(images.keys()))
        for i, (label, img) in enumerate(images.items()):
            # For displaying in Streamlit, convert appropriately
            # Streamlit's image function can handle both PIL Images and NumPy arrays
            tabs[i].image(img, caption=label, use_container_width=True)
            
            try:
                # For downloading, make sure we convert properly
                bytes_data = image_to_bytes(img)
                tabs[i].download_button(
                    label="Download",
                    data=bytes_data,
                    file_name=f"{file_prefix}_{label.replace(' ', '_').lower()}.png"
                )
            except Exception as e:
                tabs[i].warning(f"Unable to create download button: {str(e)}")

def mip_array_to_rgb(mip_array):
    # Normalize to 0–255
    mip_norm = ((mip_array - mip_array.min()) / (mip_array.max() - mip_array.min()) * 255).astype(np.uint8)
    mip_image = Image.fromarray(mip_norm).convert("RGB")  # Convert to 3-channel RGB
    return mip_image


# === Function to extract and window L3 slice ===
def extract_single_l3_slice(ct_path, l3_slice_num, level=50, window=250):
    def apply_window(slice_2d, level, window):
        max_val = level + window / 2
        min_val = level - window / 2
        return np.clip(slice_2d, min_val, max_val)

    sitk_image = sitk.ReadImage(ct_path)
    volume = sitk.GetArrayFromImage(sitk_image)
    slice_image = volume[l3_slice_num, :, :]
    return apply_window(slice_image, level, window)

def mip_array_to_image(mip_array):
    norm = ((mip_array - mip_array.min()) / (mip_array.max() - mip_array.min()) * 255).astype(np.uint8)
    return Image.fromarray(norm)



# === Streamlit App ===
st.title("CT to L3 Segmentation")

is_upside_down = st.checkbox("Is the CT scan upside down? (Is the sacrum at the top of the image?)")

uploaded_file = st.file_uploader("Upload CT Scan (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:

    st.write("Running pipeline...")

    # Save uploaded file temporarily
    with open("temp_ct.nii.gz", "wb") as f:
        f.write(uploaded_file.read())

    sitk_image = sitk.ReadImage("temp_ct.nii.gz")
    
    spacing = sitk_image.GetSpacing()  
    x_spacing, y_spacing = spacing[0], spacing[1]
    mm2_per_pixel = x_spacing * y_spacing
    
    ct_array = sitk.GetArrayFromImage(sitk_image)  # shape: (Z, H, W)

    st.subheader("View CT Scan")

    # View selection
    view = st.selectbox("View", ["Axial", "Coronal", "Sagittal"])

    # Get shape depending on view
    if view == "Axial":
        num_slices = ct_array.shape[0]
    elif view == "Coronal":
        num_slices = ct_array.shape[1]
    elif view == "Sagittal":
        num_slices = ct_array.shape[2]

    # Slice selector
    slice_idx = st.slider(f"{view} slice index", 0, num_slices - 1, num_slices // 2)

    # Windowing controls
    level = st.slider("Window Level (center)", -1000, 1000, 50)
    window = st.slider("Window Width", 1, 2000, 350)

    # Extract the slice
    if view == "Axial":
        slice_2d = ct_array[slice_idx, :, :]
    elif view == "Coronal":
        slice_2d = ct_array[:, slice_idx, :]
    elif view == "Sagittal":
        slice_2d = ct_array[:, :, slice_idx]

    # Display
    img = window_and_normalize(slice_2d, level, window)
    
    st.image(img, caption=f"{view} slice {slice_idx}", use_container_width=True)
    
    
    # ==== This is where MIP starts ======

    mipped_image_frontal = mip.preprocess_sitk_image_for_slice_detection_frontal(sitk_image)

    frontal_mip = np.squeeze(mipped_image_frontal[0])
    if is_upside_down:
        frontal_mip = np.flipud(frontal_mip)

    
    mipped_image_sag = mip.preprocess_sitk_image_for_slice_detection_sagittal(sitk_image)

    sag_mip = np.squeeze(mipped_image_sag[0]) 
    if is_upside_down:
        sag_mip = np.flipud(sag_mip)


    # ==== Here the input is the mipped image, the output is the yolo predictions on the mip image and the normalized slice, ready for input to transformer ====  
    
    l3_slice_norm_frontal, mip_frontal = mip_to_normalized_slice_frontal(frontal_mip)
    
    l3_slice_norm_sag, mip_saggital = mip_to_normalized_slice_sagittal(sag_mip)


    
    # === Step 5: Segmentation ===
    # Load your model
    def load_model():
        model = TResUnet(patch_size=4)
        model.load_state_dict(torch.load(path_to_seg_weghts, map_location=torch.device('cpu')))
        model.eval()
        return model

    segment_model = load_model()

    # Run inference on the L3 slice
    def run_inference(image):
        # Image size
        size = (256, 256)
        
        # Preprocess image
        image = cv2.resize(image, size)
        save_img = image.copy()
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to the format expected by your model
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        
        with torch.no_grad():
            # Run inference with heatmap
            heatmap, y_pred = segment_model(image, heatmap=True)
            y_pred = torch.sigmoid(y_pred)
            
            # Convert prediction to binary mask
            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.uint8) * 255
            
                        # Binary mask (1 where segmented, 0 elsewhere)
            binary_mask = y_pred > 0

            # Count number of segmented pixels
            pixel_count = np.sum(binary_mask)

            # Convert to mm² (1 pixel = 1 mm²)
            area_mm2 = pixel_count * mm2_per_pixel  # this is redundant but clear

            # Optionally: convert to cm² or m²
            area_cm2 = area_mm2 / 100

            
            # Convert grayscale mask to RGB for display
            mask_rgb = np.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=np.uint8)
            mask_rgb[:,:,1] = y_pred  # Green channel for visibility
            
            # Prepare heatmap for visualization
            # Check if heatmap is a tensor or numpy array
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap[0].cpu().numpy()
            else:
                # If already numpy, just take the first item if it's batched
                if heatmap.ndim > 3:
                    heatmap = heatmap[0]
                    
            heatmap = cv2.resize(heatmap, size)
            # Normalize heatmap to 0-255 range
            heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Create overlay visualization
            if len(save_img.shape) == 2:
                save_img_rgb = cv2.cvtColor(save_img, cv2.COLOR_GRAY2RGB)
            else:
                save_img_rgb = save_img
                
            # Create a visualization with all results
            alpha = 0.5
            overlay = cv2.addWeighted(save_img_rgb, 1, mask_rgb, alpha, 0)
            
            
            return overlay, mask_rgb, heatmap_colored, area_cm2
        
    overlay_frontal, mask_frontal, heatmap_frontal, area_frontal  = run_inference(l3_slice_norm_frontal)
    overlay_sagittal, mask_sagittal, heatmap_sagittal, area_sagittal  = run_inference(l3_slice_norm_sag)




        # === UI Toggles ===
    show_frontal = st.checkbox("Show Frontal Predictions", value=True)
    show_sagittal = st.checkbox("Show Sagittal Predictions", value=True)

    # For overlay_frontal, mask_frontal, transformer_heat_mask:
    # Make sure these are properly formatted as uint8 numpy arrays
    overlay_frontal = overlay_frontal.astype(np.uint8)
    mask_frontal = mask_frontal.astype(np.uint8)
    heatmap_frontal = heatmap_frontal.astype(np.uint8)
    
    frontal_mip_prediction = mip_frontal              
    tranverse_ct_image = overlay_frontal
    segmentation_mask = mask_frontal
    transformer_heat_mask = heatmap_frontal
    
    # For overlay_frontal, mask_frontal, transformer_heat_mask:
    # Make sure these are properly formatted as uint8 numpy arrays
    overlay_sagittal = overlay_sagittal.astype(np.uint8)
    mask_sagittal = mask_sagittal.astype(np.uint8)
    heatmap_sagittal = heatmap_sagittal.astype(np.uint8)
    
    sagittal_mip_prediction = mip_saggital              
    tranverse_ct_image_sag = overlay_sagittal
    segmentation_mask_sag = mask_sagittal
    transformer_heat_mask_sag = heatmap_sagittal


    # === Frontal Section ===
    if show_frontal:
        frontal_images = {
            "Frontal MIP Prediction": frontal_mip_prediction,
            "Transverse CT Image": tranverse_ct_image,
            "Segmentation Mask": segmentation_mask,
            "Transformer Heat Mask": transformer_heat_mask
        }
        display_prediction_block("Frontal Predictions", frontal_images, file_prefix="frontal")

    # === Sagittal Section ===
    if show_sagittal:
        sagittal_images = {
            "Sagittal MIP Prediction": sagittal_mip_prediction,
            "Transverse CT Image (Sagittal)": tranverse_ct_image_sag,
            "Segmentation Mask (Sagittal)": segmentation_mask_sag,
            "Transformer Heat Mask (Sagittal)": transformer_heat_mask_sag
        }
        display_prediction_block("Sagittal Predictions", sagittal_images, file_prefix="sagittal")
        
    st.markdown(f"**Frontal segmentation area:**  ({area_frontal:.2f} cm²)")
    st.markdown(f"**Sagittal segmentation area:**  ({area_sagittal:.2f} cm²)")



    


    
    
    
    
    
    

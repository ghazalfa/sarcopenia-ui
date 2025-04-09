from ultralytics import YOLO
from PIL import Image

model = YOLO("/Users/ghazalfallahpour/capstone_gui/frontal_best_XL_epoch100_batch-1_patience50_auto.pt")

# Save uploaded file temporarily
with open("temp_ct.nii.gz", "wb") as f:
    f.write(uploaded_file.read())

# Load the CT scan
sitk_image = sitk.ReadImage("temp_ct.nii.gz")

# Generate MIPs (tuple of frontal, sagittal)
mipped_image = mip.preprocess_sitk_image_for_slice_detection(sitk_image)

# Use frontal MIP (index 0)
frontal_mip = np.squeeze(mipped_image[0])  # ensure shape is 2D

# Convert to RGB image for YOLO
mip_rgb = mip_array_to_rgb(frontal_mip)

# Save MIP to disk (YOLO prefers file paths in this context)
mip_rgb.save("temp_mip.png")

# Run YOLO prediction
results = model.predict(source="temp_mip.png", conf=0.25)

print(results)

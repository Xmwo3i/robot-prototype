# import OpenCV
#

"""
Data Transfer - Jason
Resize the image to match the input size expected by the VLA vision encoder
Compress the image!! (e.g., JPEG/WebP) so it can be efficiently stored or transmitted
Test implementing object detection to send as metadata with YOLO

Typical VLA image size: 224x224
"""

import cv2
import argparse
import os
import json
from ultralytics import YOLO




def resize_for_vla(input_directory, input_file, output_directory, output_file_type, resized_width, resized_height):
    name, _ = os.path.splitext(input_file)
    img = cv2.imread(f"{input_directory}/{input_file}")
    h,w = img.shape[:2]
    print(f"Height:{h}\t Width: {w}")

    resized_image = cv2.resize(img, (resized_width, resized_height))

    if output_file_type in ["jpg", "jpeg"]:
        quality = [cv2.IMWRITE_JPEG_QUALITY, 50]
    elif output_file_type == "webp":
        quality = [cv2.IMWRITE_WEBP_QUALITY, 50]

    cv2.imwrite(f"{output_directory}/{name}_resized.{output_file_type}", resized_image, quality)

    img = cv2.imread(f"{output_directory}/{name}_resized.{output_file_type}")
    h2,w2 = img.shape[:2]
    print(f"New Height:{h2}\t New Width: {w2}")


#Saves the results with at least 25% confidence into a json file
def predict_image(model, img_dir, file_type, name):
    
    results = model.predict(
        source=f"{img_dir}/{name}_resized.{file_type}",
        conf=0.25,
        verbose=False
    )

    result = results[0] 

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": confidence,
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    metadata = {
        "image": f"{name}_resized.{file_type}",
        "num_detections": len(detections),
        "detections": detections
    }
    output_file_path = f"{name}_metadata.json"
    with open(output_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_dir", type = str, help = "Images folder", default = "./images")
    parser.add_argument("-i", "--input_file", type = str, help = "Input image")
    parser.add_argument("-od", "--out_dir", type = str, help = "Output folder", default = "./output")
    parser.add_argument("-ft", "--out_file_type", type = str.lower, help = "Output file type", choices=["jpeg","webp", "jpg"], default = "jpeg")
    parser.add_argument("-x", "--vla_xinput", type = int, help = "VLA X Input Size", default = 224)
    parser.add_argument("-y", "--vla_yinput", type = int, help = "VLA Y Input Size", default = 224)
    args = parser.parse_args()
    img_out_dir = args.out_dir
    name, _ = os.path.splitext(args.input_file)

    if os.path.exists(img_out_dir):
        print("Using directory")
    else: 
        print("Creating directory")
        os.makedirs(img_out_dir, exist_ok=True)
        
    resize_for_vla(args.input_dir, args.input_file, args.out_dir, args.out_file_type, args.vla_xinput, args.vla_yinput)


    model = YOLO("yolov8n.pt")

    predict_image(model,args.out_dir, args.out_file_type, name)





if __name__ == "__main__":
    main()
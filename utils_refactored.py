
import numpy as np
import cv2
import pickle
from typing import Dict, List, Tuple, Union
from PIL import Image

class FileManager:
    @staticmethod
    def load_pickle(path: str) -> Dict:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def read_txt(path: str) -> List[str]:
        with open(path, 'r') as file:
            return file.read().splitlines()

class EllipseParser:
    @staticmethod
    def parse_doc(content: List[str], just_names: List[str]) -> Dict:
        sorted_dict = {
            filename: content[i+2:i+2+int(content[i+1])]
            for i, filename in enumerate(content) if filename in just_names
        }
        return sorted_dict

    @staticmethod
    def convert_line_to_values(line: str) -> Tuple[float, float, float, float, float, float]:
        values = line.split()
        outs = [float(i) for i in values if i.replace('.', '', 1).isdigit()]
        return tuple(outs)

class BoundingBox:
    @staticmethod
    def ellipse_to_bbox(major_axis_radius: float, minor_axis_radius: float, angle: float, center_x: float, center_y: float, detection_score: float) -> Tuple[float, float, float, float, float]:
        angle_rad = np.deg2rad(angle)
        half_width = abs(major_axis_radius * np.sin(angle_rad)) + abs(minor_axis_radius * np.cos(angle_rad))
        half_height = abs(major_axis_radius * np.cos(angle_rad)) + abs(minor_axis_radius * np.sin(angle_rad))
        left_x = center_x - half_width
        top_y = center_y - half_height
        width = 2 * half_width
        height = 2 * half_height
        return left_x, top_y, width, height, detection_score

    @staticmethod
    def plot_bbox_on_image(image: np.ndarray, left_x: float, top_y: float, width: float, height: float, detection_score: float) -> np.ndarray:
        image_copy = image.copy()
        left_x, top_y, width, height = map(int, [left_x, top_y, width, height])
        cv2.rectangle(image_copy, (left_x, top_y), (left_x+width, top_y+height), (0, 255, 0), 2)
        cv2.putText(image_copy, str(detection_score), (left_x, top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image_copy

def convert_grayscale_to_rgb(img: Image, debug: bool = False) -> Image:
    if img.mode == 'L':
        if debug:
            print('Image was grayscale. Converted to RGB.')
        return img.convert('RGB')
    else:
        if debug:
            print('Image is not grayscale. No conversion was performed.')
        return img

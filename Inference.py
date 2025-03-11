import numpy as np

import cv2
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
import os
import csv
import random
import glob
from skimage.draw import polygon
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
from Generator import PointGenerator



class SAMInference:
    def __init__(self,checkpoint="sam_vit_b_01ec64.pth",model_type="vit_b",output_path='test'):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.output_path  = output_path
        os.makedirs(self.output_path, exist_ok=True)
        device = "cuda"
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        map_file = open('color_mapping.csv','r')
        reader = csv.reader(map_file)
        self.color_map ={}
        self.color_list=[]
        for row in reader:
            class_name,hex_color = row
            color = mcolors.hex2color(hex_color)
            rgb_color = tuple(int(c * 255) for c in color)
            self.color_map[class_name] = rgb_color
            self.color_list.append(rgb_color)


    def ReadPoint(self,csv_path):
        csv_file = open(csv_path,mode='r')
        csv_reader = csv.reader(csv_file)
        points=[]
        class_list=[]

        for row in csv_reader:
            cl,x,y = row
            x = int(float(x))
            y = int(float(y))
            points.append([x,y])
            class_list.append(cl)
        points =np.array(points)
        class_list =np.array(class_list)
        return points,class_list



    def MaskIOU(self,sam_mask,binary_hull_mask):

        intersection = np.logical_and(sam_mask, binary_hull_mask)
        union = np.logical_or(sam_mask, binary_hull_mask)
        # Compute IoU
        IOU = np.sum(intersection) / np.sum(union)
        return IOU

    
    def Inference(self,txt):
        image_path =txt.replace('.csv','.jpg').replace('annotations','images')
        print(image_path)
        if not os.path.exists(image_path):
            return None
        points,labels =self.ReadPoint(txt)
        unique_labels = np.unique(labels)
        masks_total = []
        scores_total = []
        color_total = []
        ori_image = cv2.imread(image_path)
        draw_img= Image.open(image_path)
        draw = ImageDraw.Draw(draw_img)
        for pt in points:
            x,y = pt
            x, y = map(int, (x,y))
            draw.ellipse((x - 10, y - 10, x + 10, y + 10),fill=(255,255,0))
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        for label in unique_labels:
            if label=='chain'or label=='scale':
                mode = 'points'
                IOU_t = 0.1
            else:
                mode = 'polygon'
                IOU_t = 0.5
            class_points = points[labels == label]
            # label_index = int(label)
            binary_hull_mask = None
            class_points=np.unique(class_points, axis=0)
            point_list = None
            try:
              
                hull = ConvexHull(class_points)
                hull_vertices = class_points[hull.vertices]
                polygon_hull = Polygon(hull_vertices)
                polygon_points = [tuple(point) for point in hull_vertices]
                
                draw.polygon(polygon_points, outline="green", width=13) 

                mask_shape = image.shape[:2]
                binary_hull_mask = np.zeros(mask_shape, dtype=np.uint8)
                polygon_coords = np.array(polygon_hull.exterior.coords)
                rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], shape=mask_shape)

                # Mark the polygon area in the binary mask
                binary_hull_mask[rr, cc] = 1

     

                point_list = PointGenerator(polygon_hull,class_points,mode=mode)
            except:
                if(len(class_points)<3):
                    continue
                point_list = random.sample(class_points, 3)  
            
            if point_list is None:
                continue
            
            label_list = [1]*len(point_list)
            input_point = np.array(point_list)
            input_label = np.array(label_list)
            sam_mask, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,)
            
            if binary_hull_mask is not None:
              IOU = self.MaskIOU(sam_mask,binary_hull_mask)
              print(f'IOU:{IOU:.2f}')

              if IOU>IOU_t:
                kernel = np.ones((5, 5), np.uint8)
                dilated_mask = cv2.dilate(binary_hull_mask, kernel, iterations=50)
                sam_mask = np.logical_and(sam_mask, dilated_mask)
          
                masks_total.append(sam_mask)
                scores_total.append(scores)
             
                current_color = self.color_map[label.lstrip('0')]
                color_total.append(current_color)
                colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
                sam_mask = sam_mask.squeeze()
                
 
                for c in range(3):  # For each channel in RGB
                    colored_mask[:, :, c] += (sam_mask.astype(np.uint8) * current_color[c])
   
   
            else:
              masks_total.append(sam_mask)
              scores_total.append(scores)
              current_color = self.color_map[label.lstrip('0')]
              color_total.append(current_color)
              print(current_color)
              colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)
              sam_mask = sam_mask.squeeze()
              num_labels, labels_m, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
              min_size = 20000
              filtered_mask = np.zeros_like(sam_mask)
              for label_m in range(1, num_labels): 
                if stats[label_m, cv2.CC_STAT_AREA] >= min_size:
                    filtered_mask[labels_m == label_m] = 255
              for i in range(0,50,3):
                    kernel = np.ones((i, i), np.uint8)  
                    filled_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

              for c in range(3):  # For each channel in RGB
                colored_mask[:, :, c] += (filled_mask.astype(np.uint8) * current_color[c])
    
        if len(masks_total) > 0:
          mask_shape = masks_total[0].squeeze().shape  # Get the height and width of a single mask
          colored_mask = np.zeros((*mask_shape, 3), dtype=np.uint8)  # RGB mask with shape (682, 1024, 3)

          # Combine masks and apply colors
          for i, mask in enumerate(masks_total):
                mask = mask.squeeze()  # Ensure mask is 2D (height x width)
                color = color_total[i]
                # print(color)
                # Apply the mask to each color channel (R, G, B)s
                num_labels, labels_m, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                min_size = 20000
                filtered_mask = np.zeros_like(mask)
                for label_m in range(1, num_labels): 
                    if stats[label_m, cv2.CC_STAT_AREA] >= min_size:
                        filtered_mask[labels_m == label_m] = 255
                for i in range(0,50,3):
                    kernel = np.ones((i, i), np.uint8)  
                    filled_mask = cv2.morphologyEx(filtered_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                
                for c in range(3):  # For each channel in RGB
                    colored_mask[:, :, c] += (filled_mask.astype(np.uint8) * color[c])

          output_name = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','.png'))
          cv2.imwrite(output_name,colored_mask)
          alpha = 0.6  # transparency factor for the mask overlay
          overlayed_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

          # Plotting the original image with the mask overlay
          output_name_mask = os.path.join(self.output_path,os.path.basename(image_path).replace('.jpg','_mask.jpg'))
          cv2.imwrite(output_name_mask,overlayed_image)

    

if __name__ == "__main__":
    Infer = SAMInference()
    
    txts = ['./annotations_plot1/IMG_8638.csv']
    #txts = glob.glob('./annotations_plot3/*.csv')
    print(f'total:{len(txts)}')
    
    for txt in txts:
        print(txt)
        Infer.Inference(txt)

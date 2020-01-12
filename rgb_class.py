import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
import json
import torchvision.transforms as transforms
from random import randrange
    
                
class ARID_Dataset(object):
    def __init__(self, root, transforms=None):
        
        self.root = root
        self.transforms = transforms
        
        #######################################################################
        ##                                                                   ##
        ##   Read all JSONs and build a DICTIONARY with all useful features  ##
        ##                                                                   ##
        #######################################################################
        
        image_id = 0
        
        self.dict_scenes = dict()
        self.dict_scenes['boxes'] = list()
        self.dict_scenes['area'] = list()
        self.dict_scenes['labels'] = list()
        self.dict_scenes['iscrowd'] = list() 
        self.dict_scenes['image_id'] = list()
        self.dict_scenes['path_img'] = list()
        
        data_list = list()

        
        
        for direc in os.listdir(self.root):
            for subdirec in os.listdir(osp.join(self.root+direc)):
                if subdirec == 'depth' or subdirec == 'col_surf' or subdirec =='rgbd':
                    continue
                else:
                    if subdirec == 'rgb':
                        images = os.listdir(osp.join(self.root+direc,subdirec))
                        for img in images:
                            if img != 'crops' and img != 'crops_processed':
                                path = osp.join(self.root+direc,osp.join(subdirec,img))
                                self.dict_scenes['path_img'].append(path)
                    else: #json
                        with open(osp.join(self.root+direc,direc+'_labels.json')) as f:
                            data = json.load(f)
                        data_list.append(data)
                    
        
        
        for k in range(len(data_list)):
            
            data = data_list[k]
            
            for i in range(len(data)):
                boxes_list = list()
                area_list = list()
                labels_list = list()
                iscrowd_list = list()
                
                for j in range(len(data[i]['annotations'])):
                    x_min = data[i]['annotations'][j]['x']              
                    y_max = data[i]['annotations'][j]['y'] 
                    x_max = x_min + data[i]['annotations'][j]['width']
                    y_min = y_max - data[i]['annotations'][j]['height']
                    boxes_list.append([x_min,y_min,x_max,y_max])
                    area_list.append((x_max-x_min)*(y_max-y_min))
                    lab = data[i]['annotations'][j]['id']
                    if lab is not None:
                        labels_list.append(lab[0:len(lab)-2])
                    else:
                        labels_list.append(lab)
                    iscrowd_list.append(False) #if True it won't be evaluated                
                
                self.dict_scenes['boxes'].append(boxes_list)
                self.dict_scenes['area'].append(area_list)
                self.dict_scenes['labels'].append(labels_list)
                self.dict_scenes['iscrowd'].append(iscrowd_list)
                self.dict_scenes['image_id'].append(image_id)
                image_id += 1 # univoque identifier     
                
        #######################################################################
        ##                                                                   ##
        ##   Build a mapping str to int since label are requested to be int  ##
        ##                                                                   ##
        #######################################################################
        
        labels = self.dict_scenes['labels']

        classes = list()
        for i in range(len(labels)):
            for j in labels[i]:
                if j not in classes and j is not None:
                    classes.append(j)
        
        self.classes = sorted(classes)
        
        i = 0
        self.labels_str2int = dict()
        for c in self.classes:
            self.labels_str2int[c] = i
            i += 1

    def __getitem__(self, idx):
        
        # load images
        img_path = self.dict_scenes['path_img'][idx]
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)

        # get bounding box coordinates for each mask
        num_objs = len(self.dict_scenes['labels'][idx])
        
        boxes = self.dict_scenes['boxes'][idx]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = list()
        for l in self.dict_scenes['labels'][idx]:
            if l is not None:
                labels.append(self.labels_str2int[l])
            else: #11 labels are None, we consider it negligible
                labels.append(randrange(51))
        # convert everything into a torch.Tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        
        area = self.dict_scenes['area'][idx]
        # convert everything into a torch.Tensor
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        
        return len(self.dict_scenes['image_id'])
    

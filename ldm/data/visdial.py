from PIL import Image
import os
import json
import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VisDialBase(Dataset): 
    def __init__(self, config=None, size=None, interpolation="bicubic", random_crop=False, crop_size=None):
        self.split = self.get_split()
        self.root_dir = "/hub_data1/inho/data/visdial"

        self.data = self.load_data()

        self.size = size
        if crop_size is None:
            self.crop_size = size if size is not None else None
        else:
            self.crop_size = crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
        if self.crop_size is not None:
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper

    def load_data(self):
        
        data = []

        if self.split == 'train':
            image_dir_cocotrain = os.path.join(self.root_dir, 'images', 'VisualDialog_train2018', 'train2014')
            image_dir_cocoval = os.path.join(self.root_dir, 'images', 'VisualDialog_train2018', 'val2014')
            
            dialog_dir = os.path.join(self.root_dir, 'dialogs')
            dialog_file = os.path.join(dialog_dir, 'train_imgid_dialogs.json') 
            f = open(dialog_file)
            dialog_data = json.load(f)
            image_id_list = list(dialog_data.keys())
            
            image_file_list = os.listdir(image_dir_cocotrain) + os.listdir(image_dir_cocoval)
            imgid_file_map = {}
            for image_file in image_file_list:
                image_id = str(int(image_file.split('_')[-1].split('.')[0]))
                imgid_file_map[image_id] = image_file
            
            for image_id in image_id_list:
                image_file = imgid_file_map[image_id]
                if 'train' in image_file:
                    image_path = os.path.join(image_dir_cocotrain, imgid_file_map[image_id]) 
                else:
                    image_path = os.path.join(image_dir_cocoval, imgid_file_map[image_id]) 
                data.append({'image_path': image_path, 'dialog_data': dialog_data[image_id]})

        elif self.split == 'val':
            image_dir = os.path.join(self.root_dir, 'images', 'VisualDialog_val2018')
            dialog_dir = os.path.join(self.root_dir, 'dialogs')
            dialog_file = os.path.join(dialog_dir, 'val_imgid_dialogs.json')    
            f = open(dialog_file)
            dialog_data = json.load(f)
            
            for image_file in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_file)
                image_id = str(int(image_file.split('_')[-1].split('.')[0]))

                data.append({'image_path': image_path, 'dialog_data': dialog_data[image_id]})

        return data  
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = dict((k, self.data[i][k]) for k in self.data[i]) 
        image_path = example['image_path']
        dialog_data = example['dialog_data']
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        if self.size is not None:
            image = self.preprocessor(image=image)["image"]        

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        dialog = ""
        for i in range(10):
            dialog = dialog + dialog_data['dialogs'][i]['question'] + " " + dialog_data['dialogs'][i]['answer'] + ". "
        example["caption"] = dialog
        return example


class VisDialTrain(VisDialBase):
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(config=config, size=size,
                          interpolation=interpolation)

    def get_split(self):
        return "train"


class VisDialValidation(VisDialBase):
    def get_split(self):
        return "val"
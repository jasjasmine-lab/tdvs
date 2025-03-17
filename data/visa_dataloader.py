import cv2
import csv
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from data.nsa import patch_ex


mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def VisADataset_cad(type, root, setting=4):

    if setting == 1 or setting == 2:
        task_num = 2
    elif setting == 3:
        task_num = 5
    else:
        assert "setting not exist"

    Dataset_list = []
    for task_id in range(task_num):
        Dataset_list.append(VisADataset_task(type, root, setting, task_id + 1))

    return Dataset_list, task_num

class VisADataset_task(Dataset):
    def __init__(self,type, root, setting, task_id):
        self.data = []
        with open(f'./data/VisA/{setting}/{task_id}.csv', 'rt') as f:
            render = csv.reader(f, delimiter=',')
            header = next(render)
            i = 0
            for row in render:
                if row[1] == type:
                    i += 1
                    data_dict = {'object':row[0],'split':row[1],'label':row[2],'image':row[3],'mask':row[4]}
                    self.data.append(data_dict)
                    
        self.label_to_idx = {'candle': '0', 'capsules': '1', 'cashew': '2', 'chewinggum': '3', 'fryum': '4', 'macaroni1': '5',
                             'macaroni2': '6', 'pcb1': '7', 'pcb2': '8', 'pcb3': '9', 'pcb4': '10',
                             'pipe_fryum': '11',}
        
        self.image_size = (256,256)
        self.root = root
        self.type = type

    def __len__(self):
        return len(self.data)

    def find_idx(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise ValueError("Index out of bounds")

        target_label = self.data[idx]['object']

        possible_indices = [i for i in range(len(self.data)) if self.data[i]['object'] == target_label and i != idx]

        if not possible_indices:
            raise ValueError("no possible")

        return random.choice(possible_indices)

    def get_nsa_args(self, clsname):
        return {'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                'num_patches': 4,
                }

    def __getitem__(self, idx):
        item = self.data[idx]

        if idx % 3 != 0 and self.type == 'train':
            nsa_idx = self.find_idx(idx)
            type = 'nsa'
        else:
            type = '_nsa'
            nsa_idx = idx

        nsa_item = self.data[nsa_idx]

        source_filename = item['image']
        nsa_filename = nsa_item['image']

        transform_fn = transforms.Resize(self.image_size)
        transform_gt = transforms.Resize(self.image_size, interpolation=Image.NEAREST)

        target = cv2.imread(os.path.join(self.root, nsa_filename))
        target = cv2.cvtColor(target, 4)
        target = Image.fromarray(target, "RGB")
        target = transform_fn(target)

        source = cv2.imread(os.path.join(self.root, source_filename))
        source = cv2.cvtColor(source, 4)
        source = Image.fromarray(source, "RGB")
        source = transform_fn(source)

        if type == 'nsa':
            source, mask = patch_ex(np.asarray(target), np.asarray(source), **self.get_nsa_args(item['object']))
            mask = (mask[:, :, 0] * 255.0).astype(np.uint8)

        else:
            if item.get("mask", None):
                mask = cv2.imread(os.path.join(self.root, item['mask']), cv2.IMREAD_GRAYSCALE)
                # mask1 = cv2.imread(self.root + item['mask'], cv2.IMREAD_GRAYSCALE)
                mask[mask > 0] = 255.0
            else:
                if item['label'] == 'normal': # good
                    mask = np.zeros(self.image_size).astype(np.uint8)
                elif item['label'] == 'anomaly':  # defective
                    mask = (np.ones(self.image_size) * 255.0).astype(np.uint8)
                else:
                    raise ValueError("Labels must be [None, 0, 1]!")

        target = transforms.ToTensor()(target)
        source = transforms.ToTensor()(source)

        prompt = ""

        mask = Image.fromarray(mask, "L")
        mask = transform_gt(mask)
        mask = transforms.ToTensor()(mask)

        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        clsname = item["object"]
        image_idx = self.label_to_idx[clsname]

        return dict(jpg=target, txt=prompt, hint=source, mask=mask, filename=source_filename, clsname=clsname, label=int(image_idx))





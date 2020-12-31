import os
import xmltodict
import cv2
import numpy as np

from autoagent.datasets.dataset import Dataset


class PascalVoc(Dataset):
    def __init__(self,
                 folder=os.path.join(os.path.dirname(__file__), "../../data/voc/"),
                 train=True,
                 load_on_ram=True):

        self.main_folder = folder
        self.load_on_ram = load_on_ram

        if train:
            # 2007trainval+2012trainval
            kind = 'trainval'
            years = ['2007', '2012']
        else:
            # 2007test
            kind = 'test'
            years = ['2007']

        self.names = []
        for y in years:
            target_folder = os.path.join(folder, f"{y}{kind}")
            with open(os.path.join((target_folder), f"ImageSets/Main/{kind}.txt"), mode='r', encoding='utf-8') as f:
                self.names.extend((f"{y}{kind}", (x.strip())) for x in f.readlines())

        if load_on_ram:
            self.imgs = [None for _ in range(len(self.names))]
            self.annotations = [None for _ in range(len(self.names))]

            print(f"Loading {__class__.__name__} on RAM...")

            for i in range(len(self.names)):
                print(f"Loading img {i}/{len(self.names)}", end="\r")
                self.imgs[i] = cv2.imread(os.path.join(self.main_folder, f"{self.names[i][0]}/JPEGImages/{self.names[i][1]}.jpg"))

                with open(os.path.join(self.main_folder, f"{self.names[i][0]}/Annotations/{self.names[i][1]}.xml"), mode='r', encoding='utf-8') as f:
                    self.annotations[i] = xmltodict.parse(
                        f.read()
                    )

    @property
    def size(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.load_on_ram:
            img = self.imgs[idx]
            annotations = self.annotations[idx]
        else:
            name = self.names[idx]
            sample = os.path.join(self.main_folder, f"{name[0]}/JPEGImages/{name[1]}.jpg")
            target = os.path.join(self.main_folder, f"{name[0]}/Annotations/{name[1]}.xml")
            img = cv2.imread(sample)

            with open(target, mode='r', encoding='utf-8') as f:
                annotations = xmltodict.parse(f.read())

        h, w, _ = img.shape

        new_annotations = {
                'objects': []
        }

        objects = annotations['annotation']['object']
        if type(objects) is not list:
            objects = [objects]

        for obj in objects:
            if int(obj['difficult']) != 1:
                # Remove 1 as voc indices start from 1
                xmin = str(int(obj['bndbox']['xmin'])-1)
                ymin = str(int(obj['bndbox']['ymin'])-1)
                xmax = str(int(obj['bndbox']['xmax'])-1)
                ymax = str(int(obj['bndbox']['ymax'])-1)

                new_annotations['objects'].append({
                    'name': obj['name'],
                    'bndbox': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    }

                })

        return img, new_annotations


if __name__ == "__main__":
    p = PascalVoc(train=False)
    print(p.size)
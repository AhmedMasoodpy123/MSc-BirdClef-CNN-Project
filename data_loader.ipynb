{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1223,
     "status": "ok",
     "timestamp": 1598976380459,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "ByGCldeIvPOu"
   },
   "outputs": [],
   "source": [
    "import random\n",
    "import os\n",
    "import numpy as np\n",
    "\n",
    "from PIL import Image\n",
    "import torch\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torchvision.datasets import ImageFolder\n",
    "from torchvision.datasets import *\n",
    "from torchvision.datasets.folder import *\n",
    "import torchvision.transforms as transforms\n",
    "from torchvision import transforms\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1861,
     "status": "ok",
     "timestamp": 1598976381143,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "wwJCGEu8Riog"
   },
   "outputs": [],
   "source": [
    "def pil_image_loader(path):\n",
    "    with open(path, 'rb') as f:\n",
    "        with Image.open(f) as img:\n",
    "            return img.convert('RGB')\n",
    "\n",
    "\n",
    "def accimage_loader(path):\n",
    "    import accimage\n",
    "    try:\n",
    "        return accimage.Image(path)\n",
    "    except IOError:\n",
    "        return pil_image_loader(path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1834,
     "status": "ok",
     "timestamp": 1598976381146,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "R7f74IIOv3ze"
   },
   "outputs": [],
   "source": [
    "def default_loader(path):\n",
    "    from torchvision import get_image_backend\n",
    "    if get_image_backend() == 'accimage':\n",
    "        return accimage_loader(path)\n",
    "    else:\n",
    "        return pil_image_loader(path)\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1806,
     "status": "ok",
     "timestamp": 1598976381149,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "09bUfcxdz-M4"
   },
   "outputs": [],
   "source": [
    "def make_dataset(dir, class_to_idx):\n",
    "    images = []\n",
    "    dir = os.path.expanduser(dir)\n",
    "    for target in sorted(os.listdir(dir)):\n",
    "        d = os.path.join(dir, target)\n",
    "        if not os.path.isdir(d):\n",
    "            continue\n",
    "\n",
    "        for root, _, fnames in sorted(os.walk(d)):\n",
    "            for fname in sorted(fnames):\n",
    "                if is_image_file(fname):\n",
    "                    path = os.path.join(root, fname)\n",
    "                    item = (path, class_to_idx[target])\n",
    "                    images.append(item)\n",
    "\n",
    "    return images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1778,
     "status": "ok",
     "timestamp": 1598976381151,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "9GYM-qEoUiTm"
   },
   "outputs": [],
   "source": [
    "def find_classes(dir):\n",
    "    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]\n",
    "    classes.sort()\n",
    "    class_to_idx = {classes[i]: i for i in range(len(classes))}\n",
    "    return classes, class_to_idx\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1754,
     "status": "ok",
     "timestamp": 1598976381152,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "zTJEzwbQz2YF"
   },
   "outputs": [],
   "source": [
    "# A bird data set Class\n",
    "class BirdDataset(Dataset):\n",
    "\n",
    "    def __init__(self, root, transform=None, target_transform=None,loader=default_loader):\n",
    "\n",
    "        \n",
    "        classes, class_to_idx = find_classes(root)\n",
    "        imgs       = make_dataset( root, class_to_idx)\n",
    "\n",
    "        if len(imgs) == 0:\n",
    "            raise(RuntimeError(\"No image found: \" + root + \"\\n\"))\n",
    "\n",
    "        self.root = root\n",
    "        self.imgs = imgs\n",
    "        self.classes = classes\n",
    "        self.class_to_idx = class_to_idx\n",
    "        self.transform = transform\n",
    "        self.target_transform = target_transform\n",
    "        self.loader = loader\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.imgs) \n",
    "        \n",
    "    def __getitem__(self, index):\n",
    "\n",
    "        path, target = self.imgs[index]\n",
    "        img = self.loader(path)\n",
    "        if self.transform is not None:\n",
    "            img = self.transform(img)\n",
    "        if self.target_transform is not None:\n",
    "            target = self.target_transform(target)\n",
    "\n",
    "        return img, target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1736,
     "status": "ok",
     "timestamp": 1598976381153,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "YYZ0YxsKR0Dw"
   },
   "outputs": [],
   "source": [
    "class Convert(object):\n",
    "    def __call__(self, img):\n",
    "        return torch.unsqueeze(torch.from_numpy(np.array(img)), 0).float()\n",
    "\n",
    "class OneHot(object):\n",
    "    def __call__(self, label):\n",
    "\n",
    "        return label\n",
    "\n",
    "class Flatten(object):\n",
    "    def __call__(self, img):\n",
    "        return img.view(28*28)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1722,
     "status": "ok",
     "timestamp": 1598976381154,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "mbB9sz6U0eHK"
   },
   "outputs": [],
   "source": [
    "\n",
    "def fetch_dataloader(types, data_dir, params, **kwargs):\n",
    "\n",
    "    dataloaders = {}\n",
    "    \n",
    "    normMean = [0.49139968, 0.48215827, 0.44653124]\n",
    "    normStd = [0.24703233, 0.24348505, 0.26158768]\n",
    "    normTransform = transforms.Normalize(normMean, normStd)\n",
    "    trainTransform = transforms.Compose([\n",
    "        transforms.RandomCrop(32, padding=4),\n",
    "        transforms.RandomHorizontalFlip(),\n",
    "        transforms.ToTensor(),\n",
    "        normTransform\n",
    "    ])\n",
    "    testTransform = transforms.Compose([\n",
    "        transforms.ToTensor(),\n",
    "        normTransform\n",
    "    ])\n",
    "        # A data transform which crops image from center and creates a 128*128 image\n",
    "    train_transformer = transforms.Compose([\n",
    "        transforms.CenterCrop((128, params.width)), \n",
    "        transforms.ToTensor()])           \n",
    "\n",
    "\n",
    "\n",
    "    for split in ['train', 'val', 'test']:\n",
    "        if split in types:\n",
    "            path       = os.path.join(data_dir, \"{}\".format(split))\n",
    "            if split == 'train':\n",
    "                dl = DataLoader(BirdDataset(path,transform=train_transformer),\n",
    "                                batch_size=params.batch_size, shuffle=True,\n",
    "                                num_workers=params.num_workers)\n",
    "            elif split == 'val':\n",
    "                dl = DataLoader(BirdDataset(path,transform=train_transformer), \n",
    "                                batch_size=params.batch_size, shuffle=False,\n",
    "                                num_workers=params.num_workers)\n",
    "            else: # test\n",
    "                dl = DataLoader(BirdDataset(path,transform=train_transformer), \n",
    "                                batch_size=params.batch_size, shuffle=False,\n",
    "                                num_workers=params.num_workers)\n",
    "\n",
    "            dataloaders[split] = dl\n",
    "\n",
    "    return dataloaders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1710,
     "status": "ok",
     "timestamp": 1598976381155,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "sjNV3j9D_eQj"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "data_loader.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

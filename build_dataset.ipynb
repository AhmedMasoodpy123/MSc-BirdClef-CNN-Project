{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Fsolk2YvYyHj"
   },
   "outputs": [],
   "source": [
    "#from __future__ import print_function\n",
    "import argparse\n",
    "import random\n",
    "import os\n",
    "from shutil import copy as cp\n",
    "import subprocess\n",
    "import librosa\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import csv\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "import pandas as pd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 2489,
     "status": "ok",
     "timestamp": 1598726613936,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "S49EB6ekbN9D",
    "outputId": "53f5609d-a911-4525-e698-fa3a8c1c054c"
   },
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount(\"mnt\", force_remount=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "byuABYJWMzPw"
   },
   "outputs": [],
   "source": [
    "#Create Spectogram Image for each file\n",
    "def create_spectogram(file,output_folder):\n",
    "  cmap = plt.get_cmap('inferno')\n",
    "  plt.figure(figsize=(8,8))\n",
    "  y, sr = librosa.load(file)\n",
    "  plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');\n",
    "  plt.axis('off');\n",
    "  plt.savefig(output_folder[:-3].replace(\".\", \"\")+\".png\")\n",
    "  plt.clf()\n",
    "  plt.close(1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "95kqyPsIOV-Y"
   },
   "outputs": [],
   "source": [
    "def extract_features_csv(filename,label,ouput_path,input):\n",
    "  y, sr = librosa.load(input)\n",
    "  rmse = librosa.feature.rmse(y=y)\n",
    "  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)\n",
    "  spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)\n",
    "  spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)\n",
    "  rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)\n",
    "  zcr = librosa.feature.zero_crossing_rate(y)\n",
    "  mfcc = librosa.feature.mfcc(y=y, sr=sr)  \n",
    "  to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '    \n",
    "  for m in mfcc:\n",
    "    to_append += f'{np.mean(m)} '\n",
    "  to_append += label\n",
    "  file = open(ouput_path+'dataset.csv', 'a', newline='')\n",
    "  with file:\n",
    "    writer = csv.writer(file)\n",
    "    writer.writerow(to_append.split())\n",
    "  file.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Q25O4rKEo540"
   },
   "outputs": [],
   "source": [
    "def splitFiles(input_path,output_path,file_list,training_percentage):\n",
    "  header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'\n",
    "  for i in range(1, 21):\n",
    "    header += f' mfcc{i}'\n",
    "  header += ' label'\n",
    "  print(header)\n",
    "  header = header.split()\n",
    "  file = open(output_path+'dataset.csv', 'w', newline='')\n",
    "  with file:\n",
    "    writer = csv.writer(file)\n",
    "    writer.writerow(header)\n",
    "  split = int(training_percentage/100. * len(file_list))\n",
    "  print(\"Count of Training\",split)\n",
    "  print(\"Count of Validation\",int((len(file_list)+split)/2)-split)\n",
    "  print(\"Count of Test\",len(file_list)-int((len(file_list)+split)/2))\n",
    "  train_filenames = file_list[:split]\n",
    "  val_filenames   = file_list[split:int((len(file_list)+split)/2)]\n",
    "  test_filenames  = file_list[int((len(file_list)+split)/2):]\n",
    "  filenames = {'train': train_filenames,\n",
    "                 'val'  : val_filenames,\n",
    "                 'test' : test_filenames}\n",
    "  print(filenames)\n",
    "  if not os.path.exists(output_path):\n",
    "    os.mkdir(output_path)\n",
    "  else:\n",
    "    print(\"Warning: output dir {} already exists\".format(output_path))\n",
    "    \n",
    "    # Preprocess train, val and test\n",
    "  for split in ['train', 'val', 'test']:\n",
    "    output_dir_split = os.path.join(output_path, '{}'.format(split))\n",
    "    output_dir_split_img=os.path.join(output_path, '{}'.format(split))\n",
    "    print(output_dir_split)\n",
    "    if not os.path.exists(output_dir_split):\n",
    "      os.mkdir(output_dir_split)\n",
    "    else:\n",
    "      print(\"Warning: dir {} already exists\".format(output_dir_split))\n",
    "   \n",
    "    if not os.path.exists(output_dir_split_img):\n",
    "      os.mkdir(output_dir_split_img)\n",
    "    else:\n",
    "      print(\"Warning: dir {} already exists\".format(output_dir_split_img))\n",
    "       \n",
    "    print(\"Copying preprocessed data to {} ...\".format(split, output_dir_split))\n",
    "    for filename in filenames[split]:\n",
    "      if not os.path.exists(output_dir_split_img+\"/\"+filename[0]):\n",
    "        print(\"Warning: dir {} created \".format(output_dir_split_img+\"/\"+filename[0]))\n",
    "        os.mkdir(output_dir_split_img+\"/\"+filename[0])\n",
    "        \n",
    "      create_spectogram(input_path+filename[0]+\"/\"+filename[1],output_dir_split_img+\"/\"+filename[0]+\"/\"+filename[1])\n",
    "      extract_features_csv(filename[1],filename[0],output_path,input_path+filename[0]+\"/\"+filename[1])\n",
    "      cp(input_path+filename[0]+\"/\"+filename[1],output_dir_split)\n",
    "          \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "dOMvwgM1mg-H"
   },
   "outputs": [],
   "source": [
    "def build_dataset(output_path,input_path):\n",
    "  csv_files=[]\n",
    "  filenames = []\n",
    "  # define the ls command\n",
    "  ls = subprocess.Popen([\"ls\", input_path],stdout=subprocess.PIPE,)\n",
    "  # define the grep command\n",
    "  grep = subprocess.Popen([\"grep\", \"-v\", \"$/\"],\n",
    "                        stdin=ls.stdout,\n",
    "                        stdout=subprocess.PIPE,\n",
    "                        )\n",
    "\n",
    "  # read from the end of the pipe (stdout)\n",
    "  endOfPipe = grep.stdout\n",
    "  classlist=[]\n",
    "  # output the files line by line\n",
    "  for line in endOfPipe:\n",
    "    classlist.append(line.decode('ascii').rstrip(\"\\n\"))\n",
    "  # Now we have all possible ClassList\n",
    "  i=0\n",
    "  for folder in classlist:\n",
    "    endOfPipe=subprocess.Popen([\"ls\", input_path+folder],\n",
    "                        stdout=subprocess.PIPE,\n",
    "                         )\n",
    "      # read from the end of the pipe (stdout)\n",
    "    files = endOfPipe.stdout\n",
    "    # output the files line by line\n",
    "\n",
    "    for file in files:\n",
    "      i=i+1\n",
    "      csv_files.append([folder,file.decode('ascii').rstrip(\"\\n\")])\n",
    "  random.seed(1240)\n",
    "  random.shuffle(csv_files)\n",
    "  print(\"File Count is:\",i)\n",
    "  print(\"Total No of Classes\",len(classlist))\n",
    "    # We have shuffled all files\n",
    "  splitFiles(input_path,output_path,csv_files,80)\n",
    "  return csv_files"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 2889618,
     "status": "ok",
     "timestamp": 1598733810176,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "rEr0xU31o4Na",
    "outputId": "e0fe6be2-321a-498c-c2e2-ca3cb3cb14f7"
   },
   "outputs": [],
   "source": [
    "csv_dataset=build_dataset(\"/content/mnt/My Drive/Dataset_bk/Spectograms/\",\"/content/mnt/My Drive/Filtered Sound/\")\n",
    "print(csv_dataset)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "iwvOniICjhk1"
   },
   "outputs": [],
   "source": [
    "def datasplit(filename,data):\n",
    "  file = open(\"/content/mnt/My Drive/Dataset_bk/img/dataset_img.csv\", 'w', newline='')\n",
    "  with file:\n",
    "    writer = csv.writer(file)\n",
    "    for row in data:\n",
    "      writer.writerow(row)\n",
    "  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 68
    },
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 1134,
     "status": "ok",
     "timestamp": 1598723645056,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "2QEx53DdkGCV",
    "outputId": "6aa55651-813d-4a16-ffe2-d31c2342f791"
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('/content/mnt/My Drive/Dataset_bk/dataset_bk.csv')\n",
    "data.head()# Dropping unneccesary columns\n",
    "data = data.drop(['filename'],axis=1)#Encoding the Labels\n",
    "\n",
    "labels = data.iloc[:, -1]\n",
    "encoder = LabelEncoder()\n",
    "encoder.fit(labels)\n",
    "y = encoder.transform(labels)\n",
    "#Scaling the Feature columns\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))\n",
    "data=[]\n",
    "i=0\n",
    "\n",
    "for row in X:\n",
    "  r=row.tolist()\n",
    "  r.append(y[i])\n",
    "  data.append(r)\n",
    "  i=i+1\n",
    "split = int(80/100. * len(data))\n",
    "print(\"Count of Training\",split)\n",
    "print(\"Count of Validation\",int((len(data)+split)/2)-split)\n",
    "print(\"Count of Test\",len(data)-int((len(data)+split)/2))\n",
    "traindata = data[:split]\n",
    "valdata   = data[split:int((len(data)+split)/2)]\n",
    "testdata  = data[int((len(data)+split)/2):]\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/traindataset.csv',traindata)\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/valdataset.csv',valdata)\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/testdataset.csv',testdata)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "colab_type": "code",
    "executionInfo": {
     "elapsed": 869,
     "status": "ok",
     "timestamp": 1598726514762,
     "user": {
      "displayName": "Daniyal Ahmed",
      "photoUrl": "",
      "userId": "08861004385506740697"
     },
     "user_tz": -300
    },
    "id": "eZneh99klUv6",
    "outputId": "b1feb27c-1f94-4bde-8727-1ef6f8cd8660"
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('/content/mnt/My Drive/Dataset_bk/dataset_bk.csv')\n",
    "data.head()# Dropping unneccesary columns\n",
    "#data = data.drop(['filename'],axis=1)#Encoding the Labels\n",
    "X=data.iloc[:,0].values\n",
    "labels = data.iloc[:, -1]\n",
    "y=labels\n",
    "\n",
    "data=[]\n",
    "i=0\n",
    "print(X)\n",
    "for row in X:\n",
    "  r=row\n",
    "  data.append([r,y[i]])\n",
    "  i=i+1\n",
    "split = int(80/100. * len(data))\n",
    "print(\"Count of Training\",split)\n",
    "print(\"Count of Validation\",int((len(data)+split)/2)-split)\n",
    "print(\"Count of Test\",len(data)-int((len(data)+split)/2))\n",
    "traindata = data[:split]\n",
    "valdata   = data[split:int((len(data)+split)/2)]\n",
    "testdata  = data[int((len(data)+split)/2):]\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/dataset_img.csv',data)\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/valdataset.csv',valdata)\n",
    "datasplit('/content/mnt/My Drive/Dataset_bk/testdataset.csv',testdata)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ndlqXPaXVoSx"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "build_dataset.ipynb",
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

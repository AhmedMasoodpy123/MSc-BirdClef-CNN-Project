{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "EaSXeD2CI7Qk"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import operator\n",
    "import time\n",
    "import numpy as np\n",
    "from sklearn.utils import shuffle\n",
    "from sklearn.metrics import average_precision_score\n",
    "import torch\n",
    "import torch.optim as optim\n",
    "from torch.optim.lr_scheduler import StepLR\n",
    "from torch.autograd import Variable\n",
    "import torch.nn.functional as F\n",
    "from tqdm import tqdm\n",
    "import logging\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "XnIJKXTEJq-J"
   },
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount(\"mnt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "FdIVHsmwv8HD"
   },
   "outputs": [],
   "source": [
    "%cd \"mnt/My Drive/Code\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "XYkrPYcnu2ZE"
   },
   "outputs": [],
   "source": [
    "!pip3 install import_ipynb\n",
    "import import_ipynb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Rn2mS88_v_SX"
   },
   "outputs": [],
   "source": [
    "!pip install utils\n",
    "import utils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "K0xOytpZKE0-"
   },
   "outputs": [],
   "source": [
    "import net\n",
    "import data_loader\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "OdU0CuvIMw_e"
   },
   "outputs": [],
   "source": [
    "# Helper Functions i.e. json_loader\"\n",
    "import helper_functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "MUSWKZSlURnO"
   },
   "outputs": [],
   "source": [
    "def load_checkpoint(filename, model, optimizer=None):\n",
    "    \"\"\"Loads model parameters (state_dict) from a reload file. \n",
    "    Args:\n",
    "        filename: (string) filename which contains the previous weights\n",
    "        model: (torch.nn.Module) model to which the weights are being loaded\n",
    "        optimizer: (torch.optim) the optimizer being used\n",
    "    \"\"\"\n",
    "    if not os.path.exists(filename):\n",
    "        raise(\"File doesn't exist {}\".format(filename))\n",
    "    filename = torch.load(filename)\n",
    "    model.load_state_dict(filename['state_dict'])\n",
    "\n",
    "    if optimizer:\n",
    "        optimizer.load_state_dict(filename['optim_dict'])\n",
    "\n",
    "    return filename"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "hWaCZPQlWNQk"
   },
   "outputs": [],
   "source": [
    "\n",
    "def save_checkpoint(state, outputpath,is_best=False):\n",
    "    \"\"\"Saves the model as a checpoint\n",
    "    Args:\n",
    "        state: (dict) Models state dict\n",
    "        outputpath: (string) Output Path\n",
    "        is_best: (bool) Check if the model is best\n",
    "    \"\"\"\n",
    "    filepath = os.path.join(outputpath+\"model_weights.tar\")\n",
    "    if is_best:\n",
    "      filepath=filepath+\".best\"\n",
    "    if not os.path.exists(outputpath):\n",
    "        print(\"Checkpoint Directory does not exist! Making directory {}\".format(outputpath))\n",
    "        os.mkdir(outputpath)\n",
    "    else:\n",
    "        print(\"Checkpoint Directory exists! {}\",filepath)\n",
    "    torch.save(state, filepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "kbt6yIgnY3Rw"
   },
   "outputs": [],
   "source": [
    "def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch):\n",
    "    # set model to training mode\n",
    "    model.train()\n",
    "    loss = 0\n",
    "    correct = 0\n",
    "    test_loss=0\n",
    "    # summary for current training loop and a running average object for loss\n",
    "    summ = []\n",
    "    nProcessed = 0\n",
    "    loss_avg = helper_functions.RunningAverage()\n",
    "    nTrain = len(dataloader.dataset)\n",
    "    # Use tqdm for progress bar\n",
    "    \n",
    "    with tqdm(total=len(dataloader)) as t:\n",
    "        for i, (train_batch, labels_batch) in enumerate(dataloader):\n",
    "            #train_batch = train_batch.view(train_batch.shape[0], -1)\n",
    "            # move to GPU if available\n",
    "            if params.cuda:\n",
    "                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)\n",
    "\n",
    "            # convert to torch Variables\n",
    "            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)\n",
    "            # compute model output and loss\n",
    "            \n",
    "            # clear previous gradients, compute gradients of all variables wrt loss\n",
    "            optimizer.zero_grad()\n",
    "            output_batch = model(train_batch)\n",
    "\n",
    "            #loss = F.nll_loss(output_batch, labels_batch)\n",
    "            loss += loss_fn(output_batch, labels_batch)\n",
    "\n",
    "            #loss.backward()\n",
    "\n",
    "            # performs updates using calculated gradients\n",
    "            optimizer.step()\n",
    "            partialEpoch = epoch + i / len(dataloader) - 1\n",
    "\n",
    "            nProcessed += len(train_batch)\n",
    "            pred = output_batch.data.max(1)[1] # get the index of the max log-probability\n",
    "            correct += pred.eq(labels_batch.data).cpu().sum()\n",
    "    test_loss = loss\n",
    "    test_loss /= len(dataloader) # loss function already averages over batch size\n",
    "    nTotal = len(dataloader.dataset)\n",
    "    acc = 100.*correct/nTotal\n",
    "    print('\\nTrain set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\\n'.format(\n",
    "          test_loss, correct, nTotal, acc))\n",
    "    return acc\n",
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
    "id": "ocluW9jIZXxL"
   },
   "outputs": [],
   "source": [
    "def evaluate(model, loss_fn, dataloader, metrics, params, num_classes, epoch , eva=False):\n",
    "\n",
    "    # set model to evaluation mode\n",
    "    model.eval()\n",
    "\n",
    "    # summary for current eval loop\n",
    "    summ = []\n",
    "    loss = 0\n",
    "    correct = 0\n",
    "    test_loss=0\n",
    "    # compute metrics over the dataset\n",
    "    cm = []\n",
    "    for i, (data_batch, labels_batch) in enumerate(dataloader):\n",
    "        # move to GPU if available\n",
    "        if params.cuda:\n",
    "            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)\n",
    "        # fetch the next evaluation batch\n",
    "        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)\n",
    "        \n",
    "        # compute model output\n",
    "        output_batch = model(data_batch)\n",
    "        loss += loss_fn(output_batch, labels_batch).item()\n",
    "        #loss += F.nll_loss(output_batch, labels_batch).item()\n",
    "        pred = output_batch.data.max(1)[1] # get the index of the max log-probability\n",
    "        correct += pred.eq(labels_batch.data).cpu().sum()\n",
    "\n",
    "    test_loss = loss\n",
    "    test_loss /= len(dataloader) # loss function already averages over batch size\n",
    "    nTotal = len(dataloader.dataset)\n",
    "    acc = 100.*correct/nTotal\n",
    "    print('\\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\\n'.format(\n",
    "        test_loss, correct, nTotal, acc))\n",
    "    return acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "qfsxIn9ui1Xz"
   },
   "outputs": [],
   "source": [
    "def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,\n",
    "                       restore_file=None):\n",
    "\n",
    "\n",
    "    # reload weights\n",
    "    if restore_file is not None:\n",
    "        restore_path = os.path.join(model_dir+restore_file)\n",
    "        logging.info(\"Restoring parameters from {}\".format(restore_path))\n",
    "        load_checkpoint(restore_path, model, optimizer)\n",
    "    \n",
    "    scheduler = None\n",
    "    if hasattr(params,'lr_decay_gamma'):\n",
    "        scheduler = StepLR(optimizer, step_size=params.lr_decay_step, gamma=params.lr_decay_gamma)\n",
    "    best_acc= 0.0\n",
    "    map_train=[]\n",
    "    map_test=[]\n",
    "    for epoch in range(params.num_epochs):\n",
    "        # Run one epoch\n",
    "        logging.info(\"Epoch {}/{}\".format(epoch + 1, params.num_epochs))\n",
    "\n",
    "        if scheduler is not None:\n",
    "            scheduler.step()\n",
    "        print(\" I have finally goten to training\")\n",
    "        # compute number of batches in one epoch (one full pass over the training set)\n",
    "        if epoch>40:\n",
    "          map_train.append(train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch))\n",
    "        # Evaluate for one epoch on validation set\n",
    "          map_test.append(evaluate(model, loss_fn, val_dataloader, metrics, params, 11, epoch))\n",
    "        else:\n",
    "          train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch)\n",
    "          evaluate(model, loss_fn, val_dataloader, metrics, params, 11, epoch)\n",
    "\n",
    "\n",
    "        is_best=False\n",
    "        if is_best==True:\n",
    "          save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optim_dict' : optimizer.state_dict()},\n",
    "                               filename=model_dir,is_best=True)\n",
    "        else:\n",
    "          save_checkpoint({'epoch': epoch + 1,\n",
    "                               'state_dict': model.state_dict(),\n",
    "                               'optim_dict' : optimizer.state_dict()},\n",
    "                               model_dir)\n",
    "\n",
    "        # If best_eval, best_save_path\n",
    "        if is_best:\n",
    "            logging.info(\"- Found new best accuracy\")\n",
    "            #best_val_met = val_acc\n",
    "\n",
    "            # Save best val metrics in a json file in the model directory\n",
    "            best_json_path = os.path.join(model_dir, \"metrics_val_best_weights.json\")\n",
    "            helper_functions.save_dict_to_json(val_metrics, best_json_path)\n",
    "    return map_train,map_test\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "AANlGk6SO9IK"
   },
   "outputs": [],
   "source": [
    "def create_logger(filepath):\n",
    "\n",
    "    logger = logging.getLogger()\n",
    "    logger.setLevel(logging.INFO)\n",
    "\n",
    "    if not logger.handlers:\n",
    "        # Logging to a file\n",
    "        file_handler = logging.FileHandler(filepath+\"train.log\")\n",
    "        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))\n",
    "        logger.addHandler(file_handler)  \n",
    "        # Logging to console\n",
    "        stream_handler = logging.StreamHandler()\n",
    "        stream_handler.setFormatter(logging.Formatter('%(message)s'))\n",
    "        logger.addHandler(stream_handler)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "LnEmWHENLC7d"
   },
   "outputs": [],
   "source": [
    "def createNetwork(params_path,output_dir,num_classes,restore_file=None):\n",
    "  #Get Path for Hyper Parameters\n",
    "  json_path = os.path.join(params_path, 'params.json')\n",
    "  assert os.path.isfile(json_path), \"No json configuration file found at {}\".format(json_path)\n",
    "  params = helper_functions.Json_Parser(json_path) #Load the json file\n",
    "  print(params)\n",
    "\n",
    "  dataloaders = data_loader.fetch_dataloader(['train', 'val'], \"/content/mnt/My Drive/Dataset_bk/Spectograms/\", params, mixing=True)\n",
    "  train_dl = dataloaders['train']\n",
    "  val_dl   = dataloaders['val']\n",
    "  #train_dl=torch.utils.data.DataLoader(train_dataset,batch_size=params.batch_size,shuffle=True)\n",
    "  #val_dl=torch.utils.data.DataLoader(valid_dataset,batch_size=params.batch_size,shuffle=True)\n",
    "\n",
    "  # use GPU if available\n",
    "  params.cuda = torch.cuda.is_available()\n",
    "  # Set the random seed for reproducible experiments\n",
    "  torch.manual_seed(1240)\n",
    "  if params.cuda:\n",
    "    torch.cuda.manual_seed(1240)\n",
    "  create_logger(output_dir)\n",
    "  \n",
    "  # Create the input data pipeline\n",
    "  logging.info(\"Loading the datasets...\")\n",
    "\n",
    "  # Define the model and optimizer\n",
    "  if params.model == 1:\n",
    "    logging.info('  -- Training using DenseNet')\n",
    "\n",
    "    model = net.DenseNet(growthRate=params.growthRate, depth=params.depth, reduction=params.reduction,\n",
    "                            bottleneck=True, nClasses=num_classes).cuda() if params.cuda else net.DenseNet(growthRate=params.growthRate, \n",
    "                            depth=params.depth, reduction=params.reduction,\n",
    "                            bottleneck=True, nClasses=num_classes)\n",
    "  elif params.model == 2:\n",
    "    logging.info('  -- Training using ResNet')\n",
    "    model = net.ResNet(params,num_classes).cuda() if params.cuda else net.ResNet(params,num_classes)\n",
    "\n",
    "  elif params.model == 3:\n",
    "    logging.info('  -- Training using SqueezeNet')\n",
    "    model = net.SqueezeNet(input_shape=(3, 128, 128), nb_classes=num_classes)\n",
    "\n",
    "  elif params.model == 4:\n",
    "    logging.info('  -- Training using Inception')\n",
    "    model = net.InceptionNet(params,num_classes).cuda() if params.cuda else net.InceptionNet(params,num_classes)\n",
    "\n",
    "  # optimizer from pytorch\n",
    "  if params.optimizer == 1:\n",
    "    logging.info('  ---optimizer is Adam')\n",
    "    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)\n",
    "    if hasattr(params,'lambd'):\n",
    "      optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.lambd)\n",
    "  elif params.optimizer == 2:\n",
    "    logging.info('  ---optimizer is SGD')\n",
    "    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params.learning_rate)\n",
    "    if hasattr(params,'lambd'):\n",
    "      optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.lambd)\n",
    "  loss_fn=None\n",
    "\n",
    "  print('  ---loss function is MSE'); print('')\n",
    "  loss_fn=net.loss_fn\n",
    "\n",
    "\n",
    "  logging.info(\"Starting training for {} epoch(s)\".format(params.num_epochs))\n",
    "  return train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, net.metrics, params, params_path,\n",
    "                       restore_file)\n",
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
    "id": "yWSryi3PcT0L"
   },
   "outputs": [],
   "source": [
    "inc_train_001,inc_test_001=createNetwork('/content/mnt/My Drive/Code/HyperParameters/inception-v4_01/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "8csPifFcyT1X"
   },
   "outputs": [],
   "source": [
    "res_train_001,res_test_001=createNetwork('/content/mnt/My Drive/Code/HyperParameters/resnet_01/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "bGLeku5u7V95"
   },
   "outputs": [],
   "source": [
    "res_train_002,res_test_002=createNetwork('/content/mnt/My Drive/Code/HyperParameters/resnet_02/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "BioSqskS7arW"
   },
   "outputs": [],
   "source": [
    "res_train_003,res_test_003=createNetwork('/content/mnt/My Drive/Code/HyperParameters/resnet_03/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "NVsMCjbf7eJx"
   },
   "outputs": [],
   "source": [
    "res_train_004,res_test_004=createNetwork('/content/mnt/My Drive/Code/HyperParameters/resnet_04/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "kWTlxo0YLOL7"
   },
   "outputs": [],
   "source": [
    "import  matplotlib.pyplot as plt\n",
    "print(res_train_001)\n",
    "print(res_train_002)\n",
    "print(res_train_003)\n",
    "print(res_train_004)\n",
    "print(res_test_001)\n",
    "print(res_test_002)\n",
    "print(res_test_003)\n",
    "print(res_test_004)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "pGxH7xU8KP2P"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(res_train_001)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,res_train_001, 'k')\n",
    "train_line.set_label('Resnet- Training')\n",
    "test_line,=plt.plot(x ,res_test_001, 'b')\n",
    "test_line.set_label('Resnet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "sVSXP-VNKdwQ"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(res_train_002)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,res_train_002, 'k')\n",
    "train_line.set_label('Resnet- Training')\n",
    "test_line,=plt.plot(x ,res_test_002, 'b')\n",
    "test_line.set_label('Resnet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "U7_BEuKqKjMq"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(res_train_003)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,res_train_003, 'k')\n",
    "train_line.set_label('Resnet- Training')\n",
    "test_line,=plt.plot(x ,res_test_003, 'b')\n",
    "test_line.set_label('Resnet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "gfkkapDZKjqy"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(res_train_004)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,res_train_004, 'k')\n",
    "train_line.set_label('Resnet- Training')\n",
    "test_line,=plt.plot(x ,res_test_004, 'b')\n",
    "test_line.set_label('Resnet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "lNbBnjKo5Zz9"
   },
   "outputs": [],
   "source": [
    "\n",
    "#train_dataset=BirdDataset('/content/mnt/My Drive/Dataset_bk/traindataset.csv')\n",
    "#valid_dataset=BirdDataset('/content/mnt/My Drive/Dataset_bk/valdataset.csv')\n",
    "train_001,test_001=createNetwork('/content/mnt/My Drive/Code/HyperParameters/densenet01/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Sk3thlpRySQK"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "dnLx_CzI5-IQ"
   },
   "outputs": [],
   "source": [
    "train_002,test_002=createNetwork('/content/mnt/My Drive/Code/HyperParameters/densenet02/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "SHN33xqZKLxY"
   },
   "outputs": [],
   "source": [
    "train_003,test_003=createNetwork('/content/mnt/My Drive/Code/HyperParameters/densenet03/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "zX5is3S2-49w"
   },
   "outputs": [],
   "source": [
    "train_004,test_004=createNetwork('/content/mnt/My Drive/Code/HyperParameters/densenet04/','/content/mnt/My Drive/Dataset_bk/Output/',11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Ri5ykFM2WpNK"
   },
   "outputs": [],
   "source": [
    "import  matplotlib.pyplot as plt\n",
    "print(train_001)\n",
    "print(train_002)\n",
    "print(train_003)\n",
    "print(train_004)\n",
    "print(test_001)\n",
    "print(test_002)\n",
    "print(test_003)\n",
    "print(test_004)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "mYFN3ioaW1mF"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(train_001)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,train_001, 'k')\n",
    "train_line.set_label('DenseNet- Training')\n",
    "test_line,=plt.plot(x ,test_001, 'b')\n",
    "test_line.set_label('DenseNet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "kfvtG3ksaJgu"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(train_002)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,train_002, 'k')\n",
    "train_line.set_label('DenseNet- Training')\n",
    "test_line,=plt.plot(x ,test_002, 'b')\n",
    "test_line.set_label('DenseNet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "3cxuSYmdbTer"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "x=[*range(10, len(train_003)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "train_line,=plt.plot(x ,train_003, 'k')\n",
    "train_line.set_label('DenseNet- Training')\n",
    "test_line,=plt.plot(x ,test_003, 'b')\n",
    "test_line.set_label('DenseNet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Q8_JcBYjcfea"
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 5))\n",
    "\n",
    "x=[*range(10, len(train_004)+10-1, 1)]\n",
    "x.insert(0,0)\n",
    "\n",
    "train_line,=plt.plot(x ,train_004, 'k')\n",
    "train_line.set_label('DenseNet- Training')\n",
    "test_line,=plt.plot(x ,test_004, 'b')\n",
    "test_line.set_label('DenseNet- Test')\n",
    "axes = plt.gca()\n",
    "axes.set_ylim([0,100])\n",
    "plt.legend()\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Epochs\")\n",
    "plt.yscale(\"linear\")\n",
    "plt.yticks(np.arange(0, 100, 5.0))\n",
    "plt.grid(True)\n",
    "plt.axis('tight')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "train.ipynb",
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
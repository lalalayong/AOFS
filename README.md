# AOFS: Arbitrary Oriented Few-Shot Object Detection in Remote Sensing Images
# Model
![Architecture](./docs/Architecture.png)



# 1. Installation 

## 1.1 Requirements
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

## 1.2 Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it.
```
conda create -n Py38_Torch1.10_cuda11.3 python=3.8 
conda activate Py38_Torch1.10_cuda11.3
```
b. Install PyTorch and torchvision following the [official instructions.](https://pytorch.org/)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
c. Clone the AOFS repository.
```
git clone https://github.com/lalalayong/AOFS.git
```
d. Install AOFS.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

## 1.3 Install DOTA_devkit(Custom) 
```
cd AOFS/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```



# 2. Prepare dataset

Get the NWPU-R data from [this](https://drive.google.com/drive/folders/144MFcNlRLFn3Oos0H4eyUgmqpfzox0MP?usp=drive_link), or prepare custom dataset files: make sure that the data label format complies with the [DOTA](https://captain-whu.github.io/DOTA/dataset.html) dataset annotation format.

### 2.1 Split the dataset

```
python DOTA_devkit/ImgSplit_multi_process.py
```

### 2.2 Generate labels

```
python tools/label_nwpu.py $DATA_PATH
```

### 2.3 Generate per-class labels(used for AFM Module)

```
python tools/label_1c_nwpu.py $DATA_PATH
```

### 2.4 Generate few-shot datasets

Change the ''DROOT'' varibale in scripts/gen_fewlist_nwpu.py to $DATA_PATH

```
python tools/gen_fewlist_nwpu.py
```

### 2.5 Generate training dictionary

```
python tools/gen_dict_file.py $DATA_PATH nwpu
```



# 3. Training

### 3.1 Base Training  

Modify Config for NWPU-R Dataset
Change the cfg/fewyolov5_nwpu.data file：

```
data = nwpu
rand = 0
novel = data/nwpu_novels.txt
novelid = 0
meta = data/nwpu_traindict_full.txt
train = $DATA_ROOT/training.txt
valid = $DATA_ROOT/evaluation.txt
gpus = 0
num_workers=8
```

Train the Model：

```
python train.py --cfg1 models/AOFS_s.yaml 
				--cfg2 models/reweight.yaml 
				--data data/nwpu_poly.yaml 
				--cfgdata cfg/fewyolov5_nwpu.data 
				--hyp data/hyps/hyp.finetune_nwpu.yaml 
				--epochs 100 --batch-size 1 --img 1024 --device 0 --noval
```

Evaluate the Model：

```
python val.py --weights weights/base_best.pt 
			  --data data/nwpu_poly.yaml 
			  --cfgdata cfg/fewyolov5_nwpu.data 
			  --batch-size 1 --img 1024 --task val --device 0 --save-json 
			  --name base_nwpu_AOFS_s_run1_split
```

Get the metrics：

```python
python DOTA_devkit/dota_evaluation_task1.py
    	--base_path	runs\val\base_nwpu_AOFS_s_run1_split\base_best
		--annopath $DATA_ROOT\evaluation\labelTxt\{:s}.txt
		--imagesetfile $DATA_ROOT\imgnamefile.txt  # File names of all images in the validation set
```

### 3.2 Few-shot Tuning

Modify Config for NWPU-R Dataset
Change the cfg/fewtunev5_nwpu_10shot.data file (change the shot number to try different shots)：

```
data = nwpu
tuning = 1
novel = data/nwpu_novels.txt
novelid = 0
rand = 0
max_epoch = 10000
repeat = 100
dynamic = 0
train = $DATA_ROOT/training.txt
meta = data/nwpu_traindict_bbox_10shot.txt
valid = $DATA_ROOT/evaluation.txt
gpus = 0
num_workers=8
```

  Train the Model with 10 shot：

```
python train.py --weights weights/base_best.pt 
				--data data/nwpu_poly.yaml 
				--cfgdata cfg/fewtunev5_nwpu_10shot.data 
				--hyp data/hyps/hyp.finetune_nwpu.yaml 
				--batch-size 1 --img 1024 --device 0 --noval
```

Evaluate the Model：

```
python val.py --weights weights/tune_best.pt 
			  --data data/nwpu_poly.yaml 
			  --cfgdata cfg/fewtunev5_nwpu_10shot.data 
			  --batch-size 1 --img 1024 --task val --device 0 --save-json 
			  --name tune_nwpu_10shot_AOFS_s_run1_split
```

Get the metrics：

```python
python DOTA_devkit/dota_evaluation_task1.py \
    	--base_path runs\val\tune_nwpu_10shot_AOFS_s_run1_split\tune_best
		--annopath $DATA_ROOT\evaluation\labelTxt\{:s}.txt
		--imagesetfile $DATA_ROOT\imgnamefile.txt
```



# Acknowledgements

This project uses excellent codes from other open source projects. Special thanks to the following authors：

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [Few-shot YOLOv3](https://github.com/lixiang-ucas/FSODM)
- [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb)
- [CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)


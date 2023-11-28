## Conditional generation of images

Steps for training a new conditional network:

Step 1: Unzip the image folders in to a folder named 'data' in the home directory

```.bash
unzip data.zip
rm -rf __MACOSX
rm -rf data/.DS_Store
rm -rf `find -type d -name .ipynb_checkpoints`
```

Step 2: Create 'dataset.json' file for datasets with class labels

```.bash
python make_json_labels.py
```

Step 3: Preprocess the images to the required specification

```.bash
python dataset_tool.py --source=data/ --dest=datasets/faces256.zip --transform=center-crop --width=256 --height=256
```

Step 4: Transfer learn from the pretrained unconditional model till convergence

```.bash
python train.py --outdir=training-runs --data=datasets/faces256.zip \
--gpus=8 --batch=512 --resume=pretrained_models/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl --snap=50 --cfg=paper256 --aug=ada --target=0.5
```

Step 5: Train one step of the conditional model (till the initial model .pkl file is generated)

```.bash
python train.py --outdir=training-runs --data=datasets/faces256.zip \
--cond=1 --gpus=8 --batch=512 --snap=50 --cfg=paper256 --aug=ada --target=0.5 --kimg=1
```

Step 6: Transfer weights from unconditional to conditional. Make sure to set the right paths to the weights

```.bash
python weight_transfer.py
```

Step 7: Resume training of the unconditional model till convergence

```.bash
python train.py --outdir=training-runs --data=datasets/faces256.zip \
--cond=1 --gpus=8 --batch=512 --snap=50 --cfg=paper256 --aug=ada --target=0.5 \
--resume=pretrained_models/cond-weight-transfer-resume.pkl
```

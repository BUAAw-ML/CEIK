## Overview

To improve the reliability and prediction performance of pre-trained model, related works retrieve the external explicit knowledge to help produce answers. This project works on better retrieval and exploitation of external explicit knowledge in VQA. 


## Requirements: 

torch,
torchvision,
torchaudio,
transformers==4.12.5,
faiss-gpu,
tensorboard,
setuptools==59.5.0,
wandb,
pytorch-lightning,
jsonnet,
easydict,
pandas, 
scipy,
opencv-python,
fuzzywuzzy,
scikit-image, 
matplotlib,
timm,
scikit-learn,
sentencepiece 


## Download Datasets

You can prepare the OK-VQA dataset as follows:

### COCO images
`data/ok-vqa/train2014`: [Train images](http://images.cocodataset.org/zips/train2014.zip)

`data/ok-vqa/val2014`: [Test images](http://images.cocodataset.org/zips/val2014.zip)

### OKVQA Dataset
`data/ok-vqa/mscoco_train2014_annotations.json`: [Training annotations](https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip)

`data/ok-vqa/mscoco_val2014_annotations.json`: [Testing annotations](https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip)

`data/ok-vqa/OpenEnded_mscoco_train2014_questions.json`: [Training questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip)

`data/ok-vqa/OpenEnded_mscoco_val2014_questions.json`: [Testing questions](https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip)

### Google Search Corpus
[Official download link](https://drive.google.com/drive/folders/15uWx33RY5UmR_ZmLO6Ve1wyzbXsLxV6o?usp=sharing)

Data can be saved to `data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus.csv`.


You can also download other datasets and place it at './data'

## Feature Extraction
### VinVL Features (object detection/attributes/relations)
#### Step 1: Install environments
VinVL needs a separate env.

Refer to [Offical installation guide](https://github.com/microsoft/scene_graph_benchmark/blob/main/INSTALL.md)

Since HPC uses A-100, which requires a higher version of CUDA, the recommended environment with CUDA 10.1 does not work.

```
conda create --name sg_benchmark python=3.7 -y
conda activate sg_benchmark
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install ipython h5py nltk joblib jupyter pandas scipy -y
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5 
python -m pip install cityscapesscripts
pip install pycocotools scikit-image timm einops
cd materials/scene_graph_benchmark
python setup.py build develop
```


#### Step 2: Generating OKVQA datasets
```
cd materials/scene_graph_benchmark
python tools/prepare_data_for_okvqa.py
```
This command generates trainset/testset of OKVQA datasets to `datasets/okvqa/`, which will be used in object detection.

#### Step 3: Download pre-trained models
```
mkdir models
mkdir models/vinvl
/path/to/azcopy copy https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth ./models/vinvl/
```

#### Step 4: Running models
`vinvl_vg_x152c4` is a pre-trained model with object and attribute detection:
For OKVQA dataset:
```
python tools/test_sg_net.py \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_x152c4_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```


`vinvl_large` is a pre-trained model with **only** object detection. But it was pre-trained on more object detection datasets!
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_testset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```
```
python tools/test_sg_net.py  \
    --config-file sgg_configs/vgattr/vinvl_large_okvqa_trainset.yaml  \
    TEST.IMS_PER_BATCH 8  \
    MODEL.WEIGHT models/vinvl/vinvl_large.pth  \
    MODEL.ROI_HEADS.NMS_FILTER 1  \
    MODEL.ROI_HEADS.SCORE_THRESH 0.2  \
    DATA_DIR "./datasets/"  \
    TEST.IGNORE_BOX_REGRESSION True  \
    MODEL.ATTRIBUTE_ON True  \
    TEST.OUTPUT_FEATURE True
```

#### Step 5: Recommended Save Path
The object/attribute data can be saved to `data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv`.

### Oscar+ Features (image captioning)
#### Step 1: Download data
We can download COCO-caption data with azcopy:
```
cd materials/Oscar
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/datasets/coco_caption' ./oscar_dataset --recursive
```
Reference: [offical download page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md)

#### Step 2: Download the pre-trained model
We can download [COCO captioning large](https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/coco_captioning_large_scst.zip) here, or refer to the [official download page](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#Image-Captioning-on-COCO) for the model checkpoints.

Save the pre-trained model to `pretrained_models/coco_captioning_large_scst`.

#### Step 3: Running the inference
```
python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml oscar_dataset/coco_caption/[train/val/test].yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --output_prediction_path './output/[train/val/test]_predictions.json' \
    --eval_model_dir pretrained_models/coco_captioning_large_scst/checkpoint-4-50000
```


#### Step 4: Recommended Save Path
The data can be saved to `./data/ok-vqa/pre-extracted_features/captions/train_predictions.json`.


### Google OCR Features
First, enable Google OCR APIs; download the key file to `google_ocr_key.json`. This is **not** free! Ask me for the already generated features.
```
cd src
python ocr.py
```
The detected features will be saved to `./data/ok-vqa/pre-extracted_features/OCR`.

## Training Dense Passage Retrieval

```
python main.py ../configs/okvqa/DPR.jsonnet \
    --mode train \
    --experiment_name OKVQA_DPR_FullCorpus  \
    --accelerator auto --devices auto  \
    --strategy ddp \
    --modules exhaustive_search_in_testing \
    --opts train.epochs=10 \
            train.batch_size=8 \
            valid.step_size=1 \
            valid.batch_size=32 \
            train.additional.gradient_accumulation_steps=4 \
            train.lr=0.00001
```

### Prepare FAISS index files for dynamic DPR retrieval

```
python tools/prepare_faiss_index.py  \
    --csv_path ../data/ok-vqa/pre-extracted_features/passages/okvqa_full_corpus_title.csv \
    --output_dir  ../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-new-framework \
    --dpr_ctx_encoder_model_name ../Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder \
    --dpr_ctx_encoder_tokenizer_name ../Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/item_encoder_tokenizer \
```

## Train and Test

Training:
```
python main.py ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices 1  \
    --modules force_existence  \
    --opts train.epochs=20  \
            train.batch_size=1  \
            valid.step_size=1  \
            valid.batch_size=1  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            model_config.loss_ratio.additional_loss=1  \
            data_loader.additional.num_knowledge_passages=6
```

Testing:
```
python main.py ../configs/RAVQA.jsonnet  \
    --mode test  \
    --experiment_name OKVQA_RA-VQA_FullCorpus  \
    --accelerator auto --devices auto  \
    --modules force_existence  \
    --opts data_loader.additional.num_knowledge_passages=6  \
            test.load_model_path=../Experiments/OKVQA_RA-VQA_FullCorpus/train/saved_model/epoch_06.ckpt
```



## Notes

+ The VQA model adopted in this project is T5, and there are many more powerful models that can be applied to the VQA task recently. The method proposed in this project can be applied to the latest model to achieve better performance. 
+ Joint training and Pseudo Relevance Labels are often adopted for the retriever training. Pseudo Relevance Labels is based on whether the correct answer appears in the retrieved snippet, rather than whether the pre-trained model can output the correct answer after adding the retrieved snippet. This project proposes a better retriever training method in the joint training framework. 
+ This publication version was made in a rush due to the current heavy workload of the author. We will add follow-up patches to make codes more readible and ensure reproducibility.


## Citation
If this code helped your research, please kindly cite the paper 'Coordinating explicit and implicit knowledge for knowledge-based VQA' and 'Retrieval Augmented Visual Question Answering with Outside Knowledge'. In addition, we would like to thank the related works that helped this project.




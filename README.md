# CS433-ML-PROJECT2-TB DETECTION

### Detecting TB from chest x-rays in a population of patients living with HIV and diabetes in West Africa

**Team members**
Haozhe Qi (EDIC) <haozhe.qi@epfl.ch>
Mu Zhou (EDIC) <mu.zhou@epfl.ch>
Anna Paulish (CSE) <anna.paulish@epfl.ch>

In this repository, you can find the code for TB detection from chest x-rays with HIV and diabetes in West Africa. 

In this project, we aim to:
- make a survey on the existing studies on TB detection from chest x-rays,
- test the existing methods (with code) on our dataset,
- do data preprocessing (especiall label remove for each images),
- prepare a pipeline for TB detection and show the performance on each subgroup (HIV and Diabetes).


### Contribution
- A new TB detection pipeline that can perform well on the challenging TB dataset. And various attempts for computer vision techniques like data preprocessing and augmentation, image classification models and training tricks.
- A label remove method that can automatically remove the labels in the original photos, and a label removed dataset that can be used to train learning based label remove methods.
- A wide thorough research of the existing TB detection methods and datasets, and a fair comparison with them.

### Baseline
To test as many as advanced computer techniques as we can, instead of using existing TB detection method as our baseline, we resort to build our baseline from scratch with deep learning classification library [TIMM](https://github.com/rwightman/pytorch-image-models). TIMM is a frequently updated library that already has 2.5k stars on the github. With the help of this library, we successfully build our baseline and test various advanced techniques in network architecture, loss function, optimiser and learning rate schedule.


### Data
Due to the privacy issue, we only release the data in the report. You can find the link in **section 3.1** in our report.

### How to reproduce the reuslts

You can run a series of experiments by using:
```
bash run.sh
```
or run a single process with
```sh
python train.py --A_des 'train resnet with StepLR' \
                --model resnet50 --sched step --decay-epochs 50\
                --data_dir /home/project/data2/our_data_processed \
                --dataset_type all &> logs/res_step_lr.out
```
You can put your description of each experiment with argument **A_des**. You can update the model, path of dataset, and type of dataset by arguments **model** *(resnet50, efficientnet_b2)* , **data_dir** and **dataset_type** *('all' for whole dataset, 'D' for diabetes, 'H' for HIV)*.


### Code structure
```
- experiments
    ├── dataset.csv                ### whole dataset
    ├── train.csv                  ### training data
    ├── test.csv                   ### test data
─ preprocess                  
    ├── data_analysis.py           ### evaluate the number of each dataset
    ├── prepare_dataset.py         ### split train / val / test dataset
    ├── detect_rectangle.py        ### remove the labels (as described in report section 3.2.2 - improved method 1)
    ├── find_rectangle_new.py      ### remove the labels (as described in report section 3.2.3 - improved method 2)
    └── border_remove.py           ### remove the border and crop data (as described in section 3.3.1)
─ dataset  
    ├── dataset.py                 ### define the class of TB dataset
    └── data_loader.py             ### load the data
─ arguments.py                     ### all the arguments we need 
─ f1_loss.py                       ### calculate F1 loss
─ focal_loss.py                    ### focal loss
─ train.py                         ### train and test script
─ run.sh                           ### script to run the experiments
```

### Algorithm details - Label remove

#### detect_rectangle.py (section 3.2.2)
The procedure:


#### find_rectangle_new.py (section 3.2.3)
The procedure:
![remove2](./figs/rec_detection.png)


### Comparison works in the report

- [TBCNN](https://github.com/frapa/tbcnn) 
- [XTBTorch](https://github.com/frapa/tbcnn)
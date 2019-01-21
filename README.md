# Brain-Image-Analysis
Paper list and resources on machine learning for brain image (e. g. fMRI and sMRI) analysis. 

Contributed by Jinlong Hu, Yuezhen Kuang and Lijie Cao.

##### Table of Contents  

1. [Survey](#survey)  
2. [Resting-state fMRI (voxel)](#Resting-state-fMRI-on-voxel-level)
3. [Resting-state fMRI (region)](#Resting-state-fMRI-on-region-level )  
4. [Task fMRI](#task-fmri)
5. [sMRI and others](#sMRI-and-other-data)
6. Special diseases: [Parkinson](#Parkinson), [Depression](#depression)
7. [Dataset](#dataset)
8. Other algorithms: [Multiview learning](#Multiview-learning)

## Survey
#### On voxel
1. **Deep Learning in Medical Image Analysis**
   - *Dinggang Shen, et al.*   2017.
   
1. **Applications of Deep Learning to MRI Images: A Survey**
   - *Jin Liu, et al.*  2018.
   - [paper](https://www.researchgate.net/profile/Jin_Liu20/publication/323491805_Applications_of_deep_learning_to_MRI_images_A_survey/links/5aa0be5caca272d448b2175f/Applications-of-deep-learning-to-MRI-images-A-survey.pdf)

#### On region (network /graph)
1. **A Comprehensive Survey on Graph Neural Networks**
   - *Zonghan Wu, et al.* 2019.
   - [paper](https://arxiv.org/pdf/1901.00596.pdf)
   - [中文解读](https://mp.weixin.qq.com/s/0rs8Wur7Iv6jSpFz5C-KNg)
   - More graph neural networks (GNN) papers, see [GNN-paper-list](https://github.com/largeapp/GNNPapers)
   
## Resting-state fMRI on voxel level 
#### Deep learning for voxel
1. **Deep Learning in Medical Imaging: fMRI Big Data Analysis via Convolutional Neural Networks**
   - *Amirhessam Tahmassebi，et al.*  2018.
   
1. **deep learning of resting state networks from independent component analysis**
   - *Yiyu Chou，et al.*   2018.
   
1. **Learning Neural Markers of Schizophrenia Disorder Using Recurrent Neural Networks**
   - *Jumana Dakka, et al.*  2017.
   
1. **Using deep learning to investigate the neuroimaging correlates of psychiatric and neurological disorders: Methods and applications**
   - *Sandra Vieira, et al.*  2017.
   
1. **using resting state functional mri to build a personalized autism diagnosis system**
   - *Omar Dekhil, et al.*  ISBI 2018.
   
1. **Whole Brain fMRI Pattern Analysis Based on Tensor Neural Network**
   - *XIAOWEN XU, et al.*  2018.
   
1. **2-channel convolutional 3d deep neural network (2cc3d) for fmri analysis: asd classification and feature learning**
   - *Xiaoxiao Li, et al.*  ISBI 2018.
   
1. **Brain Biomarker Interpretation in ASD Using Deep Learning and fMRI**
   - *Xiaoxiao Li, et al.*  2018.  

   
1. **The Unsupervised Hierarchical Convolutional Sparse Auto-Encoder for Neuroimaging Data Classification**
   - *Xiaobing Han, et al.*  2015.
   
1. **Learning Neural Markers of Schizophrenia Disorder Using Recurrent Neural Networks**
   - *Jumana Dakka, et al.*  2017.
   
1. **Classification of Alzheimer’s Disease Using fMRI Data and Deep Learning Convolutional Neural Networks**
   - *Saman Sarraf, Ghassem Tofighi*  2016.
  
1. **Deep learning for neuroimaging: a validation study**
   - *Sergey M. Plis, *  2014.
   
1. **The Unsupervised Hierarchical Convolutional Sparse Auto-Encoder for Neuroimaging Data Classification**
   - *Xiaobing Han, et al.* 2015.
   - 自动编码器，ADHD-200，ADNI数据 

1. **Group-wise Sparse Representation Of Resting-state fMRI Data For Better Understanding Of Schizophrenia**
   - *Lin Yuan, et al.*  2017.
   
   
1. **Neuroscience meets Deep Learning**
   - *Dhruv Nathawani, et al.* 
   - CNN, CMU 2008数据
   
1. **Brain Age Prediction Based On Resting-state Functional Connectivity Patterns Using Convolutional Neural Networks**
   - *Hongming Li* 
   - 3D, t-SNE结果分析
   
1. **Voxelwise 3D Convolutional and Recurrent Neural Networks for Epilepsy and Depression Diagnostics from Structural and Functional MRI Data**
   - *Marina Pominova, et al.*  2018.

1. **Automatic Recognition of fMRI-derived Functional Networks using 3D Convolutional Neural Networks**
   - *Yu Zhao, et al.*  2017.
   
   
1. **3D CNN Based Automatic Diagnosis of Attention Deficit Hyperactivity Disorder Using Functional and Structural MRI**
   - *LIANG ZOU, et al.*  2017.
   
1. **3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study**
   - *Jose Dolz, et al.* 2016.
   - 分割，多个数据集
   
1. **Multi-Scale 3D Convolutional Neural Networks for Lesion Segmentation in Brain MRI**
   - *Konstantinos Kamnitsas, et al.*
   
1. **3-D Functional Brain Network Classification using Convolutional Neural Networks**
   - *Dehua Ren, et al.* 2017.

   
1. **Modeling 4D fMRI Data via Spatio-Temporal Convolutional Neural Networks (ST-CNN)**
   - *Yu Zhao, et al.*  2018.
   
1. **3D Deep Learning for Multi-modal Imaging-Guided Survival Time Prediction of Brain Tumor Patients**
   - *Dong Nie, et al.*  2016.
   - 多模态
   
1. **DeepAD: Alzheimer’s Disease Classification via Deep Convolutional Neural Networks using MRI and fMRI**
   - *Saman Sarraf, et al.*  2016.
   - 多模态：MRI, fMRI   
   
 
1. **Alzheimer's Disease Classification Based on Combination of Multi-model Convolutional Networks**
   - *Fan Li, et al.*  2017.
   - 使用多个多尺度的3D 卷积自动编码器
   

1. **Multi-tasks Deep Learning Model for classifying MRI images of AD/MCI Patients**
   - *S.Sambath Kumar, et al.*  2017.
   
1. **Retrospective head motion estimation in structural brain MRI with 3D CNNs**
   - *Juan Eugenio Iglesias, et al.* 
   - 识别头部是否移动，提高ABIDE预测准确率。
   
1. **Transfer learning improves resting-state functional connectivity pattern analysis using convolutional neural networks**
   - *Pál Vakli, et al.* 2018.
   - fMRI, 迁移学习
   
   
1. **Towards Alzheimer’s Disease Classification through Transfer Learning**
   - *Marcia Hon, et al.*  BIBM 2017.
   - 使用迁移学习进行阿兹海默症疾病分类
   
   
1. **Deep neural network with weight sparsity control and pre-training extracts hierarchical features and enhances classification performance: Evidence from whole-brain resting-state functional connectivity patterns of schizophrenia**
   - *Junghoe Kim, et al.* 2016.

   
1. **基于卷积神经网络的ADHD的判别分析**
   - *俞一云，何良华* 2017.
   - [PPT](https://wenku.baidu.com/view/606cb472974bcf84b9d528ea81c758f5f61f29da.html)

  
#### Non-deep-learning for voxel
1. **Multi-way Multi-level Kernel Modeling for Neuroimaging Classification**
   - *Lifang He, et al.*   CVPR 2017.
   
1. **Spatio-Temporal Tensor Analysis for Whole-Brain fMRI Classication**
   - *Guixiang Ma, et al.* 
   
1. **Feature Selection with a Genetic Algorithm for Classification of Brain Imaging Data**
   - *Annamária Szenkovits, et al.* 2017.
   - 特征选择

1. **Building a Science of Individual Differences from fMRI**
   - *Julien Dubois* 2016.
   - 从组到个体的研究
   
1. **Feature fusion via hierarchical supervised local CCA for diagnosis of autism spectrum disorder**
   - *Feng Zhao, et al.*  2016.
   - 不同特征融合，多模态数据   
   
   

## Resting-state fMRI on region level 
#### Deep learning for region
1. **Resting State fMRI Functional Connectivity-Based Classification Using a Convolutional Neural Network Architecture**
   - *Regina Júlia Meszlényi, et al.* 2017.
   - 499个脑区的网络用CNN
   
1. **Identifying Connectivity Patterns for Brain Diseases via Multi-side-view Guided Deep Architectures**
   - *Jingyuan Zhang, et al.*  2016.
 
1. **Do Deep Neural Networks Outperform Kernel Regression for Functional Connectivity Prediction of Behavior?**
   - *Tong He, et al.* 2018.
   - [paper](https://www.biorxiv.org/content/biorxiv/early/2018/11/19/473603.full.pdf)
   - 精简版：**Is deep learning better than kernel regression for functional connectivity prediction of fluid intelligence?** [paper](http://holmeslab.yale.edu/wp-content/uploads/2018-He.pdf)


#### Non-deep-learning for region
1. **Resting-State Functional Connectivity in Autism Spectrum Disorders: A Review**
   - *Jocelyn V. Hull, et al.*  2017.

1. **A Novel Approach to Identifying a Neuroimaging Biomarker for Patients With Serious Mental Illness**
   - *Alok Madan, et al.*
   
1. **Classification of Resting State fMRI Datasets Using Dynamic Network Clusters**
   - *Hyo Yul Byun, et al.* 2014
   - 动态功能网络聚类
   
1. **全图表征学习的研究进展**
   - 唐建，中国计算机学会通讯，2018.03
   - 全图嵌入方法

## task fMRI
#### Deep learning for voxel
1. **Deep learning of fMRI big data: a novel approach to subject-transfer decoding**
   - *Sotetsu Koyamada, et al.*  2015.
   
1. **Brains on Beats**
   - *Umut Guclu, et al.* 
   - 用DNN测人脑对音乐的反应。
   
1. **deep learning for brain decoding**
   - *Orhan Firat, et al.*  2014.
   - 自编码器
   
1. **Learning Representation for fMRI Data Analysis using Autoencoder**
   - *Suwatchai Kamonsantiroj, et al.* 2016.
   - 自动编码器, CMU 2008数据
   
1. **modeling task fMRI data via deep convolutional autoencoder**
   - *Heng Huang, et al.*  2017.
   - 卷积自动编码器
   
1. **Learning Deep Temporal Representations for fMRI Brain Decoding**
   - *Orhan Firat, et al.* 2015.
   
1. **Task-specific feature extraction and classification of fMRI volumes using a deep neural network initialized with a deep belief network: Evaluation using sensorimotor tasks**
   - *Hojin Jang, et al.*  2017.
   

#### Non-deep-learning for region
1. **Improving accuracy and power with transfer learning using a meta-analytic database**
   - *Yannick Schwartz, et al.* 2012.
   - 迁移学习
 
 
## sMRI and other data
   
1. **Alzheimer’s Disease Diagnostics By Adaptation Of 3d Convolutional Network**
   - *Ehsan Hosseini-Asl, et al.*  2016.
   - 数据：sMRI
   
1. **Alzheimer’s disease diagnostics by a 3D deeply supervised adaptable convolutional network**
   - *Ehsan Hosseini Asl, et al.*  2018.
   - 数据：sMRI ADNI, 迁移学习与域适应

1. **3D CNN-based classification using sMRI and MD-DTI images for Alzheimer disease studies**
   - *Alexander Khvostikov, et al.* 2018.
   - 数据：sMRI,DTI
   
1. **Deep MRI brain extraction: A 3D convolutional neural network for skull stripping**
   - *Jens Kleesiek, et al.* 2016. 
   - 数据：sMRI

1. **Predicting Alzheimer’s disease: a neuroimaging study with 3D convolutional neural networks**
   - *Adrien Payan and Giovanni Montana* 2015.
   - 数据：sMRI
   
1. **Automatic Detection Of Cerebral Microbleeds Via Deep Learning Based 3d Feature Representation**
   - *Hao Chen, et al.*  2015.
   - 数据：SWI
   
1. **learning representations from eeg with deep recurrent-convolutional neural networks**
   - *Pouya Bashivan, et al.*  ICLR 2016.
   - 数据：EEG

1. **Classification of Clinical Significance of MRI Prostate Findings Using 3D Convolutional Neural Networks**
   - *Alireza Mehrtash, et al.*
   - 数据：Multi-parametric magnetic resonance imaging (mpMRI), DWI and DCE-MRI modalities
   
1. **Marginal Space Deep Learning: Efficient Architecture for Detection in Volumetric Image Data**
   - *Florin C. Ghesu, et al.* 
   - 数据：超声，非脑成像，2D到nD
   

## Parkinson 
1. **Discriminating cognitive status in Parkinson’s disease through functional connectomics and machine learning**
   - *Alexandra Abós, et al.*  2017.
   
1. **Graph Theoretical Metrics and Machine Learning for Diagnosis of Parkinson's Disease Using rs-fMRI**
   - *Amirali Kazeminejad, et al.* 2017.
   
1. **Joint feature-sample selection and robust diagnosis of Parkinson’s disease from MRI data**
   - *Ehsan Adeli, et al.* 2016
   
1. **Aberrant regional homogeneity in Parkinson’s disease: A voxel-wise meta-analysis of resting-state functional magnetic resonance imaging studies**
   - *PingLei Pan, et al.*  2016.
   
1. **Abnormal Spontaneous Brain Activity in Early Parkinson’s Disease With Mild Cognitive Impairment: A Resting-State fMRI Study**
   - *Zhijiang Wang, et al.* 2018.
   
1. **Can neuroimaging predict dementia in Parkinson’s disease?**
   - *Juliette H. Lanskey, et al.*  2018.
   
1. **Classification of Resting-State fMRI for Olfactory Dysfunction in Parkinson’s Disease using Evolutionary Algorithms**
   - *Amir Dehsarvi, et al.* 2018.
   
      
1. **Decreased interhemispheric homotopic connectivity in Parkinson's disease patients with freezing of gait: A resting state fMRI study**
   - *Junyi Li, et al.* 2018.
   
1. **Model-based and Model-free Machine Learning Techniques for Diagnostic Prediction and Classifcation of Clinical Outcomes in Parkinson’s Disease**
   - *Chao Gao, et al.*  2018.
   
1. **On the Integrity of Functional Brain Networks in Schizophrenia, Parkinson’s Disease, and Advanced Age: Evidence from Connectivity-Based Single-Subject Classification**
   - *Rachel N. Pl€aschke, et al.*  2017.
   
   
1. **Resting State fMRI: A Valuable Tool for Studying Cognitive Dysfunction in PD**
   - *Kai Li, et al.* 2018.
   
1. **The Parkinson’s progression markers initiative (PPMI) – establishing a PD biomarker cohort**
   - *Kenneth Marek, et al.* 2018.
   - PPMI 数据描述
   

## Depression
1. **Studying depression using imaging and machine learning methods**
   - *Meenal J. Patel, et al.* 2015.

1. **Dynamic Resting-State Functional Connectivity in Major Depression**
   - *Roselinde H Kaiser, et al.* 2016.
   
1. **Detecting Neuroimaging Biomarkers for Depression: A Meta-analysis of Multivariate Pattern Recognition Studies**
   - *Joseph Kambeitz, et al.* 2016.
   - 其他论文结果收集分析
   
1. **Depression Disorder Classification of fMRI Data Using Sparse Low-Rank Functional Brain Network and Graph-Based Features**
   - *Xin Wang, et al.* 2016.
   
1. **Biomarker approaches in major depressive disorder evaluated in the context of current hypotheses**
   - *Mike C Jentsch, et al.* 2015.
   
1. **Accuracy of automated classification of major depressive disorder as a function of symptom severity**
   - *Rajamannar Ramasubbu, et al.* 2016
   
1. **Resting-state connectivity biomarkers define neurophysiological subtypes of depression**
   - *Andrew T Drysdale, et al.* 2017.
   - 亚型
   
1. **Diagnostic classification of unipolar depression based on restingstate functional connectivity MRI: effects of generalization to a diverse sample**
   - *Benedikt Sundermann, et al.*  2017.
   
1. **Multivariate Classification of Blood Oxygen Level–Dependent fMRI Data with Diagnostic Intention: A Clinical Perspective**
   - *B. Sundermann, et al.* 2014. 
   
## dataset
1. Human Connectome Project (HCP)
   - [HCP](https://www.humanconnectome.org/)
   
1. Openfmri & openneuro
   - [openneuro](https://openneuro.org/)
   
1. Parkinson's Progression Markers Initiative (PPMI)
   - [PPMI](https://www.ppmi-info.org/)
   
1. Autism Brain Imaging Data Exchange (ABIDE)
   - [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/)
   
## Multiview learning
#### Survey
1. **A Survey on Multi-view Learning**
   - *Chang Xu, Dacheng Tao, Chao Xu* 2013
   - [Paper](https://arxiv.org/pdf/1304.5634.pdf)
   
1. **Multi-view learning overview: Recent progress and new challenges**
   - *Jing Zhao,Xijiong Xie, Xin Xu, Shiliang Sun* 2017
   - [Paper](https://www.sciencedirect.com/science/article/pii/S1566253516302032)
   
#### Tutorial
1. **Multiview Feature Learning Tutorial**  *@ CVPR 2012*
   - [Tutorial link](http://www.cs.toronto.edu/~rfm/multiview-feature-learning-cvpr/)
   
1. **Multiview Feature Learning** *@ IPAM 2012*
   - [Tutorial link](http://helper.ipam.ucla.edu/publications/gss2012/gss2012_10790.pdf)
   
   
#### MVL with Deep Learning
1. **On deep multi-view representation learning**
   - *Wang, Weiran, et al.* 2015.
1. **Multi-view deep network for cross-view classification** 
   - *Kan, Meina, Shiguang Shan, and Xilin Chen* 2016.

1. **Multi-view perceptron: a deep model for learning face identity and view representations** 
   - *Zhu, Zhenyao, et al.* 2014.

1. **A multi-view deep learning approach for cross domain user modeling in recommendation systems**
   - *Elkahky, Ali Mamdouh, Yang Song, and Xiaodong He* 2015.

1. **A novel channel-aware attention framework for multi-channel EEG seizure detection via multi-view deep learning**
   - *Yuan, Ye, et al.* 2018.

1. **Volumetric and multi-view cnns for object classification on 3d data** 
   - *Qi, Charles R., et al.* 2016.
   
#### Multimodal Deep Learning
1. **Multimodal deep learning**
   - *Ngiam, Jiquan, et al.* ICML 2011.
   - [paper](http://ai.stanford.edu/~ang/papers/icml11-MultimodalDeepLearning.pdf)
   
1. **Multimodal learning with deep boltzmann machines**
   - *Srivastava, Nitish, and Ruslan R. Salakhutdinov* NIPS 2012
   - [paper](http://120.52.51.18/www.cs.toronto.edu/~rsalakhu/papers/Multimodal_DBM.pdf)

1. **Deep multimodal learning: A survey on recent advances and trends**
   - *Ramachandram, Dhanesh, and Graham W. Taylor* 2017.   

 
#### Brain Image
1. **Deep Learning Approaches to Unimodal and Multimodal Analysis of Brain Imaging Data With Applications to Mental Illness**
   - *Calhoun, Vince, and Sergey Plis* 2018.
   
1. **Multimodal neuroimaging feature learning with multimodal stacked deep polynomial networks for diagnosis of Alzheimer's disease**
   - *Shi, Jun, et al.* 2018.
   
1.  **Exploring diagnosis and imaging biomarkers of Parkinson's disease via iterative canonical correlation analysis based feature selection**
    - *Liu L , Wang Q , Adeli E , et al.* 2018.
    - Discussed in lab meeting (LJ Cao).
   

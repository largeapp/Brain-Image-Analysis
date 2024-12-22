# Brain-Image-Analysis
Paper list and resources on machine learning for brain image (e. g. fMRI and sMRI) analysis. 

Contributed by Jinlong Hu, Yuezhen Kuang, and Lijie Cao, from School of Computer Science and Engineering, South China University of Technology, Guangzhou, China.

(Our research collection on artificial intelligence for brain image analysis is available on [this link](https://github.com/largeapp/AI-for-Brain-Image-Analysis))

##### Table of Contents  

1. [Survey](#survey)  
2. [Resting-state fMRI (voxel)](#Resting-state-fMRI-on-voxel-level)
3. [Resting-state fMRI (region)](#Resting-state-fMRI-on-region-level) : [Special issue](#Special-issue)
4. [Task fMRI](#task-fmri)
5. [sMRI and others](#sMRI-and-other-data)
6. Special diseases: [Parkinson](#Parkinson), [Autism](#Autism), [Depression](#depression)
7. [Dataset](#dataset)
8. Other algorithms: [Multiview learning](#Multiview-learning)

## Survey
  
#### On machine learning
1. **Machine learning studies on major brain diseases: 5-year trends of 2014–2018**
   - [paper](https://link.springer.com/article/10.1007/s11604-018-0794-4), 2018
  
1. **Machine Learning for Predicting Cognitive Diseases: Methods, Data Sources and Risk Factors**
   - [paper](https://link.springer.com/article/10.1007/s10916-018-1071-x), 2018
  
1. **Adaptive Sparse Learning for Neurodegenerative Disease Classification**
   - [paper](https://ieeexplore.ieee.org/abstract/document/8241617), 2017

1. **Classification on Brain Functional Magnetic Resonance Imaging: Dimensionality, Sample Size, Subject Variability and Noise**
    - [paper](https://www.cs.purdue.edu/homes/jhonorio/fmrisynth_bookchapter14.pdf)

1. **Classification and Prediction of Brain Disorders Using Functional Connectivity: Promising but Challenging**
    - [paper](https://www.frontiersin.org/articles/10.3389/fnins.2018.00525/full), 2018

#### On brain connectivity dynamics
1. **Brain Connectivity Dynamics** issue, NeuroImage, October 2018
     - [link](https://www.sciencedirect.com/journal/neuroimage/vol/180/part/PB)

1. **Dynamic Graph Metrics: Tutorial, Toolbox, and Tale**
    - Ann E. Sizemore and Danielle S. Bassett, 2017
    - [code](https://github.com/asizemore/Dynamic-Graph-Metrics)

1. **The dynamic functional connectome: State-of-the-art and perspectives**
    - Maria Giulia Pretia, etc., NeuroImage, 2017

1. **BRAPH: A graph theory software for the analysis of brain connectivity**
    - Mite Mijalkov,  etc. , PLOS ONE, 2017.
    - [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178798)
    - [code](http://www.braph.org/)    
  
#### On deep learning
1. **Deep Learning in Medical Image Analysis**
   - *Dinggang Shen, et al.*   2017.
   
1. **Applications of Deep Learning to MRI Images: A Survey**
   - *Jin Liu, et al.*  2018.
   - [paper](https://www.researchgate.net/profile/Jin_Liu20/publication/323491805_Applications_of_deep_learning_to_MRI_images_A_survey/links/5aa0be5caca272d448b2175f/Applications-of-deep-learning-to-MRI-images-A-survey.pdf)
   
1. **A Comprehensive Survey on Graph Neural Networks**
   - *Zonghan Wu, et al.* 2019.
   - [paper](https://arxiv.org/pdf/1901.00596.pdf)
   - [with Chinese](https://mp.weixin.qq.com/s/0rs8Wur7Iv6jSpFz5C-KNg)
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
   - *Sergey M. Plis*, 2014.
   
1. **The Unsupervised Hierarchical Convolutional Sparse Auto-Encoder for Neuroimaging Data Classification**
   - *Xiaobing Han, et al.* 2015.
   - auto encoding，ADHD-200，ADNI data 

1. **Group-wise Sparse Representation Of Resting-state fMRI Data For Better Understanding Of Schizophrenia**
   - *Lin Yuan, et al.*  2017.
   
   
1. **Neuroscience meets Deep Learning**
   - *Dhruv Nathawani, et al.* 
   - CNN, CMU 2008 data
   
1. **Brain Age Prediction Based On Resting-state Functional Connectivity Patterns Using Convolutional Neural Networks**
   - *Hongming Li* 
   - 3D, t-SNE analysis
   
1. **Voxelwise 3D Convolutional and Recurrent Neural Networks for Epilepsy and Depression Diagnostics from Structural and Functional MRI Data**
   - *Marina Pominova, et al.*  2018.

1. **Automatic Recognition of fMRI-derived Functional Networks using 3D Convolutional Neural Networks**
   - *Yu Zhao, et al.*  2017.
   
   
1. **3D CNN Based Automatic Diagnosis of Attention Deficit Hyperactivity Disorder Using Functional and Structural MRI**
   - *LIANG ZOU, et al.*  2017.
   
1. **3D fully convolutional networks for subcortical segmentation in MRI: A large-scale study**
   - *Jose Dolz, et al.* 2016.
   - segmentation，multiple data sets.
   
1. **Multi-Scale 3D Convolutional Neural Networks for Lesion Segmentation in Brain MRI**
   - *Konstantinos Kamnitsas, et al.*
   
1. **3-D Functional Brain Network Classification using Convolutional Neural Networks**
   - *Dehua Ren, et al.* 2017.

   
1. **Modeling 4D fMRI Data via Spatio-Temporal Convolutional Neural Networks (ST-CNN)**
   - *Yu Zhao, et al.*  2018.
   
1. **3D Deep Learning for Multi-modal Imaging-Guided Survival Time Prediction of Brain Tumor Patients**
   - *Dong Nie, et al.*  2016.
   - multi-modal
   
1. **DeepAD: Alzheimer’s Disease Classification via Deep Convolutional Neural Networks using MRI and fMRI**
   - *Saman Sarraf, et al.*  2016.
   - multi-modal：MRI, fMRI   
   
 

1. **Multi-tasks Deep Learning Model for classifying MRI images of AD/MCI Patients**
   - *S.Sambath Kumar, et al.*  2017.
   
1. **Retrospective head motion estimation in structural brain MRI with 3D CNNs**
   - *Juan Eugenio Iglesias, et al.* 
   - head moving, ABIDE data set.
   
1. **Transfer learning improves resting-state functional connectivity pattern analysis using convolutional neural networks**
   - *Pál Vakli, et al.* 2018.
   - fMRI, transfer learing
   
   
1. **Towards Alzheimer’s Disease Classification through Transfer Learning**
   - *Marcia Hon, et al.*  BIBM 2017.
   - transfer learning
   
   
1. **Deep neural network with weight sparsity control and pre-training extracts hierarchical features and enhances classification performance: Evidence from whole-brain resting-state functional connectivity patterns of schizophrenia**
   - *Junghoe Kim, et al.* 2016.

1. **Reproducibility of importance extraction methods in neural network based fMRI classification**
   - *Athanasios Gotsopoulos, et al.*  NeuroImage 2018.
   - Important voxels  
   
1. **Spatiotemporal feature extraction and classification of Alzheimer’s disease using deep learning 3D-CNN for fMRI data**
    - *Harshit Parmar*,2020
    - [paper](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-7/issue-05/056001/Spatiotemporal-feature-extraction-and-classification-of-Alzheimers-disease-using-deep/10.1117/1.JMI.7.5.056001.full?SSO=1)
  
#### Non-deep-learning for voxel

1. **Multi-way Multi-level Kernel Modeling for Neuroimaging Classification**
   - *Lifang He, et al.*   CVPR 2017.
   
1. **Spatio-Temporal Tensor Analysis for Whole-Brain fMRI Classication**
   - *Guixiang Ma, et al.* 
   
1. **Feature Selection with a Genetic Algorithm for Classification of Brain Imaging Data**
   - *Annamária Szenkovits, et al.* 2017.
   - feature selection

1. **Building a Science of Individual Differences from fMRI**
   - *Julien Dubois* 2016.
   - from group to individual
   
1. **Feature fusion via hierarchical supervised local CCA for diagnosis of autism spectrum disorder**
   - *Feng Zhao, et al.*  2016.
   - feature fusion, multi-modal data   
   
   

## Resting-state fMRI on region level 
#### Deep learning for region
1.  **GAT-LI: a graph attention network based learning and interpreting method for functional brain network classification**
     - *Jinlong Hu, et al*, 2021

1. **Interpretable Learning Approaches in Resting-State Functional Connectivity Analysis: The Case of Autism Spectrum Disorder**
   - *Jinlong Hu, et al*, 2020
   - [code](https://github.com/largeapp/ifc)

1. **Resting State fMRI Functional Connectivity-Based Classification Using a Convolutional Neural Network Architecture**
   - *Regina Júlia Meszlényi, et al.* 2017.
   - 499 brain regions, CNN
   
1. **Identifying Connectivity Patterns for Brain Diseases via Multi-side-view Guided Deep Architectures**
   - *Jingyuan Zhang, et al.*  2016.
 
1. **Do Deep Neural Networks Outperform Kernel Regression for Functional Connectivity Prediction of Behavior?**
   - *Tong He, et al.* 2018.
   - [paper](https://www.biorxiv.org/content/biorxiv/early/2018/11/19/473603.full.pdf)
   - simple version：**Is deep learning better than kernel regression for functional connectivity prediction of fluid intelligence?** [paper](http://holmeslab.yale.edu/wp-content/uploads/2018-He.pdf)

1. **Metric learning with spectral graph convolutions on brain connectivity networks**
   -*Sofia IraKtena, et al.* 2018
   - [paper](https://www.sciencedirect.com/science/article/pii/S1053811917310765)
   
1. **Multi Layered-Parallel Graph Convolutional Network (ML-PGCN) for Disease Prediction**
   - *Anees Kazi, Shadi Albarqouni, Karsten Kortuem, Nassir Navab* 2018.
   - [paper](https://arxiv.org/abs/1804.10776)
   
1. **Classifying resting and task state brain connectivity matrices using graph convolutional networks**
   - *Michael Craig, et al.* 
   - [paper](https://www.researchgate.net/profile/Michael_Craig15/publication/320347164_Classifying_resting_and_task_state_brain_connectivity_matrices_using_graph_convolutional_networks/links/5a38db04458515919e7278ab/Classifying-resting-and-task-state-brain-connectivity-matrices-using-graph-convolutional-networks.pdf)
   
1. **Multi-View Graph Convolutional Network and Its Applications on Neuroimage Analysis for Parkinson's Disease**
   - *xi zhang, et al.* 2018.
   - [paper](https://arxiv.org/abs/1805.08801)
   -  data：PPMI, DTI

#### Non-deep-learning for region
1. **Resting-State Functional Connectivity in Autism Spectrum Disorders: A Review**
   - *Jocelyn V. Hull, et al.*  2017.

1. **A Novel Approach to Identifying a Neuroimaging Biomarker for Patients With Serious Mental Illness**
   - *Alok Madan, et al.*
   
1. **Classification of Resting State fMRI Datasets Using Dynamic Network Clusters**
   - *Hyo Yul Byun, et al.* 2014
   - dynamic brain network, clustering
   

   
#### Special issue 
 Contributed by Lijie. 
1. **Spectral Graph Convolutions for Population-Based Disease Prediction** 2017.
   - subjects as nodes

1. **Distance Metric Learning using Graph Convolutional Networks: Application to Functional Brain Networks** 2017.
   - brain networks as input，mutric learning with GCN Siamese network
   
1. **Disease Prediction using Graph Convolutional Networks: Application to Autism Spectrum Disorder and Alzheimer’s Disease**
1. **Multi Layered-Parallel Graph Convolutional Network (ML-PGCN) for Disease Prediction**
1. **SELF-ATTENTION EQUIPPED GRAPH CONVOLUTIONS FOR DISEASE PREDICTION**
   - Paper 3, 4, and 5 refer to Paper 1.

1. **Metric Learning with Spectral Graph Convolutions on Brain Connectivity Networks**
   - refer to Paper 2.

1. **Similarity Learning with Higher-Order Proximity for Brain Network Analysis**
   - refer to Paper 6, and introduce higher-order information
   
1. **Multi-View Graph Convolutional Network and Its Applicationson Neuroimage Analysis for Parkinson’sDisease**
   - refer to Paper 6, with multi view
   
1. **Graph Saliency Maps through Spectral Convolutional Networks: Application to Sex Classiﬁcation with Brain Connectivity**
   - refer to Paper 6, with explanation
   
1. **Integrative Analysis of Patient Health Records and Neuroimages via Memory-based Graph Convolutional Network**
   - refer to Paper 6, multi-modal



## task fMRI
#### Deep learning for voxel
1.  **A Multichannel 2D Convolutional Neural Network Model for Task-Evoked fMRI Data Classification**
    - *Jinlong Hu, et al.*, 2019.
    - [code](https://github.com/largeapp/M2DCNN)

1. **Deep learning of fMRI big data: a novel approach to subject-transfer decoding**
   - *Sotetsu Koyamada, et al.*  2015.
   
1. **Brains on Beats**
   - *Umut Guclu, et al.* 
   - DNN, reaction to the music。
   
1. **deep learning for brain decoding**
   - *Orhan Firat, et al.*  2014.
   - auto encoder
   
1. **Learning Representation for fMRI Data Analysis using Autoencoder**
   - *Suwatchai Kamonsantiroj, et al.* 2016.
   - auto encoder, CMU 2008 data
   
1. **modeling task fMRI data via deep convolutional autoencoder**
   - *Heng Huang, et al.*  2017.
   - convolution autoencoder
   
1. **Learning Deep Temporal Representations for fMRI Brain Decoding**
   - *Orhan Firat, et al.* 2015.
   
1. **Task-specific feature extraction and classification of fMRI volumes using a deep neural network initialized with a deep belief network: Evaluation using sensorimotor tasks**
   - *Hojin Jang, et al.*  2017.
   

#### Non-deep-learning for region
1. **Improving accuracy and power with transfer learning using a meta-analytic database**
   - *Yannick Schwartz, et al.* 2012.
   - transfer learning
 
 
## sMRI and other data
   
1. **Alzheimer’s Disease Diagnostics By Adaptation Of 3d Convolutional Network**
   - *Ehsan Hosseini-Asl, et al.*  2016.
   - sMRI
   
1. **Alzheimer’s disease diagnostics by a 3D deeply supervised adaptable convolutional network**
   - *Ehsan Hosseini Asl, et al.*  2018.
   - sMRI ADNI, transfer learing and adapting
   
1. **Alzheimer's Disease Classification Based on Combination of Multi-model Convolutional Networks**
   - *Fan Li, et al.*  2017.
   - multi 3D auto-encoding convolutinal networks
   - sMRI (ADNI)

1. **3D CNN-based classification using sMRI and MD-DTI images for Alzheimer disease studies**
   - *Alexander Khvostikov, et al.* 2018.
   - sMRI,DTI
   
1. **Deep MRI brain extraction: A 3D convolutional neural network for skull stripping**
   - *Jens Kleesiek, et al.* 2016. 
   - sMRI

1. **Predicting Alzheimer’s disease: a neuroimaging study with 3D convolutional neural networks**
   - *Adrien Payan and Giovanni Montana* 2015.
   - sMRI
   
1. **Using structural MRI to identify bipolar disorders – 13 site machine learning study in 3020 individuals from the ENIGMA Bipolar Disorders Working Group**
   - [link](https://www.nature.com/articles/s41380-018-0228-9)
   - sMRI
   
1. **Sex Differences in the Adult Human Brain: Evidence from 5216 UK Biobank Participants**
   - [link](https://academic.oup.com/cercor/article/28/8/2959/4996558)
   - [Chinese analysis](http://k.sina.com.cn/article_5994750011_16550a03b00100fdqn.html?cre=tianyi&mod=pcpager_fintoutiao&loc=5&r=9&doct=0&rfunc=86&tj=none&tr=9)
   - sMRI
 
  
### others
1. **Automatic Detection Of Cerebral Microbleeds Via Deep Learning Based 3d Feature Representation**
   - *Hao Chen, et al.*  2015.
   - SWI
   
1. **learning representations from eeg with deep recurrent-convolutional neural networks**
   - *Pouya Bashivan, et al.*  ICLR 2016.
   - EEG

1. **Classification of Clinical Significance of MRI Prostate Findings Using 3D Convolutional Neural Networks**
   - *Alireza Mehrtash, et al.*
   - Multi-parametric magnetic resonance imaging (mpMRI), DWI and DCE-MRI modalities
   
1. **Marginal Space Deep Learning: Efficient Architecture for Detection in Volumetric Image Data**
   - *Florin C. Ghesu, et al.* 
   - data: supersound，non brain imaging，2D to nD

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
   - PPMI data.
   
1. **Multi-View Graph Convolutional Network and Its Applications on Neuroimage Analysis for Parkinson’s Disease**
   - [paper](https://arxiv.org/pdf/1805.08801.pdf), 2018
   - DTI
   - multi-view, DTI with multi edge building methods, multiple graphs.
   
1. **A Fully-Automatic Framework for Parkinson’s Disease Diagnosis by Multi-Modality Images**
   - [paper](https://arxiv.org/ftp/arxiv/papers/1902/1902.09934.pdf)
   - sMRI, PET, multi-modal
   
1. **Multi-task Sparse Low-Rank Learning for Multi-classification of Parkinson’s Disease**
   - [paper](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_41)
   - PPMI
   
1. **Parkinson's Disease Diagnosis via Joint Learning from Multiple Modalities and Relations**
   - [paper](https://ieeexplore.ieee.org/abstract/document/8453792)
   - PPMI, multi-modal 

1.  [more about Parkinson](https://github.com/largeapp/Brain-Image-Analysis/blob/master/parkinsons.md)
 
## Autism
1. **The Autism Brain Imaging Data Exchange: Towards Large-Scale Evaluation of the Intrinsic Brain Architecture in Autism**
   - [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4162310/)
   - ABIDE 1 
   
1. **Enhancing studies of the connectome in autism using the autism brain imaging data exchange II**
   - [paper](https://www.nature.com/articles/sdata201710)
   - ABIDE 2 
   
1. **Predicting autism spectrum disorder using domain-adaptive cross-site evaluation**
   - *Bhaumik R, Pradhan A, Das S, et al.*, Neuroinformatics, 2018.
   - [paper](https://link.springer.com/article/10.1007/s12021-018-9366-0)
   - dataset: ABIDE
   
1. **Promises, Pitfalls, and Basic Guidelines for Applying Machine Learning Classifiers to Psychiatric Imaging Data, with Autism as an Example**
   - *Pegah Kassraian-Fard, et al.*, 2016.
   - [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133050/)
   - dataset: ABIDE
   
1. **Identification of autism spectrum disorder using deep learning and the ABIDE dataset**
   - *Heinsfeld A S, Franco A R, Craddock R C, et al.* , 2018
   - dataset: ABIDE
   - [paper](https://www.sciencedirect.com/science/article/pii/S2213158217302073)
   - algorithm：deep learning, DNN
   
1. **Age and Gender Effects on Intrinsic Connectivity in Autism Using Functional Integration and Segregation**
   - *Teague Rhine Henry, Gabriel S. Dichter, and Kathleen Gates*, 2017
   - dataset: ABIDE
   
1. **Enhancing the representation of functional connectivity networks by fusing multi‐view information for autism spectrum disorder diagnosis**
   - *Huifang Huang  Xingdan Liu  Yan Jin  Seong‐Whan Lee  Chong‐Yaw Wee  Dinggang Shen*, 2018
   - *Human brain mapping*, February 15, 2019
   - dataset: ABIDE
   
1. **Towards Accurate Personalized Autism Diagnosis Using Different Imaging Modalities: sMRI, fMRI, and DTI**
   - *ElNakieb Y, Ali M T, Dekhil O, et al.*  2018 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT).
   - multi-modal：sMRI, fMRI, DTI



## Depression
1. **Studying depression using imaging and machine learning methods**
   - *Meenal J. Patel, et al.* 2015.

1. **Dynamic Resting-State Functional Connectivity in Major Depression**
   - *Roselinde H Kaiser, et al.* 2016.
   
1. **Detecting Neuroimaging Biomarkers for Depression: A Meta-analysis of Multivariate Pattern Recognition Studies**
   - *Joseph Kambeitz, et al.* 2016.
   - meta analysis
   
1. **Depression Disorder Classification of fMRI Data Using Sparse Low-Rank Functional Brain Network and Graph-Based Features**
   - *Xin Wang, et al.* 2016.
   
1. **Biomarker approaches in major depressive disorder evaluated in the context of current hypotheses**
   - *Mike C Jentsch, et al.* 2015.
   
1. **Accuracy of automated classification of major depressive disorder as a function of symptom severity**
   - *Rajamannar Ramasubbu, et al.* 2016
   
1. **Resting-state connectivity biomarkers define neurophysiological subtypes of depression**
   - *Andrew T Drysdale, et al.* 2017.
   - subtypes.
   
1. **Diagnostic classification of unipolar depression based on restingstate functional connectivity MRI: effects of generalization to a diverse sample**
   - *Benedikt Sundermann, et al.*  2017.
   
1. **Multivariate Classification of Blood Oxygen Level–Dependent fMRI Data with Diagnostic Intention: A Clinical Perspective**
   - *B. Sundermann, et al.* 2014. 
   
1. **Identification of depression subtypes and relevant brain regions using a data-driven approach**
   - *Tomoki Tokuda, et al.* 2018. scientific reports
   - [link](https://www.nature.com/articles/s41598-018-32521-z)
   - subtypes
   - [media reports](https://www.medicalnewstoday.com/articles/323559.php)
   
   
   
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
   

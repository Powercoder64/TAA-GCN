# Repository for TAA-GCN: A temporally aware Adaptive Graph Convolutional Network for Age Estimation [(Pattern Recognition Paper)](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005465)

![aaaa](https://raw.githubusercontent.com/Powercoder64/TAA-GCN/main/web/pipeline.jpg)

**Abstract**: *This paper proposes a novel age estimation algorithm, the Temporally-Aware Adaptive Graph Convolutional Network (TAA-GCN). Using a new representation based on graphs, the TAA-GCN utilizes skeletal, posture, clothing, and facial information to enrich the feature set associated with various ages. Such a novel graph representation has several advantages: First, reduced sensitivity to facial expression and other appearance variances; Second, robustness to partial occlusion and non-frontal-planar viewpoint, which is commonplace in real-world applications such as video surveillance. The TAA-GCN employs two novel components, (1) the Temporal Memory Module (TMM) to compute temporal dependencies in age; (2) Adaptive Graph Convolutional Layer (AGCL) to refine the graphs and accommodate the variance in appearance. The TAA-GCN outperforms the state-of-the-art methods on four public benchmarks, UTKFace, MORPHII, CACD, and FG-NET. Moreover, the TAA-GCN showed reliability in different camera viewpoints and reduced quality images.*

Please look at our Pattern Recognition paper:
[Link to our paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005465)

**Installing dependencies:**

Prerequisites:    
- PyTorch > 1.4   
- torchvision   
- pyyaml   
- argparse   
- numpy   
- h5py   
- opencv-python   
- imageio   
- scikit-learn   
- scikit-video   

Create a new Conda environment:

```conda create -n TAAGCN python=3.7```   
```conda activate TAAGCN```

Install the libraries:
```conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch```   
```pip install PyYAML```   
```pip install argparse```   
```pip install h5py```   
```pip install opencv-python```   
```pip install imageio```   
```pip install scikit-learn```   
```pip install scikit-video```

Copy/download the repository and go to the root of the source code.   
Go to the torchlight folder and run the following:  

```cd TAA-GCN```   
```python setup.py install```


**Preparing data:**   

We provide you the download links for the data and the pre-trained models. 
The [UTKFACE](https://susanqq.github.io/UTKFace/) dataset includes both facial and body information. Download the data from [here](https://drive.google.com/file/d/15BIJlUsJ-F6HGUSYUXwTZLvdyKOskN_U/view?usp=sharing) and download the model from [here](https://drive.google.com/file/d/10WXfP3e5mPvH1qXzQGTyje69nnPzJ_yF/view?usp=sharing).  

Please extract the .zip files and copy the downloaded ```data``` and ```model``` folders to the root of the source codes.

Fix the data and model paths in ```./config/ddgcn/utk_skel_face.yaml``` accordingly. 

**Data structure:**

The input data is a *NxJXF* dimensional vector, where *N* is the number of samples, *J* is the number of skeletal-facial keypoints, and *F* is the size of each keypoint. In our experiments, *J=39* that includes *19* facial keypoints and *20* skeletal keypoints. *F=1536* that includes tiled coordinates of each keypoint followed by the flatten image patch centered around each keypoint.

**Running the models:**

Fix the path for the config file in ```run``` and use the following to run the model:

```sh run```


**References:**

We appreciate the following [reference](https://arxiv.org/abs/1801.07455):

Yan, Sijie et al. "Spatial temporal graph convolutional networks for skeleton-based action recognition." AAAI 2018.


**Citation:**

If you find this repository useful please cite us:

```
@article{korban2023TAAGCN,
  title={TAA-GCN: A temporally aware Adaptive Graph Convolutional Network for age estimation},
  author={Korban, Matthew and Youngs, Peter and Acton, Scott T},
  journal={Pattern Recognition},
  volume={134},
  pages={109066},
  year={2023},
  publisher={Elsevier}
}
  ```
  
  ```
  Korban, Matthew, Peter Youngs, and Scott T. Acton. "TAA-GCN: A temporally aware Adaptive Graph Convolutional Network for age estimation." Pattern Recognition 134 (2023): 109066.
  ```

Currently, this repository only includes the data and model for the UTKFace dataset since it is the only public age dataset that includes both face and body. We might add the data and models for face-only datasets as well.  


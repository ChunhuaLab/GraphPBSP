# GraphPBSP: protein binding site prediction based on Graph Attention Network and pre-trained model ProstT5.

We develop a model called GraphPBSP, which is an effective predictor based on Graph Attention Network with Convolutional Neural Network and Multilayer Perceptron for binding site prediction.

Authors: Xiaohan Sun, Zhixiang Wu, Jingjie Su and Chunhua Li. 

The process includes two steps: installation and prediction.

Here, we take a protein-protein complex with PDB ID 5NRM for example to show the prediction process, which includes 18 experimental binding sites. 


The sequence and its conresponding experimental labels of 5NRM (chain A) are the following:

Sequence = 'TGFNLSIDTVEGNPGSSVVVPVKLSGISKNGISTADFTVTYDATKLEYISGDAGSIVTNPGVNFGINKESDGKLKVLFLDYTMSTGYISTDGVFANLNFNIKSSAAIGSKAEVSISGTPTFGDSTLTPVVAKVTNGAVNLE'.

Labels   = '000000000000000000000000000000000101000000000000000000000000010111101000001010101110000000000000000000000000000000000001010001010000000000000'.


## Step 1 Installation

* Python version: 3.8

  pip install biopython == 1.81
  pip install scikit-learn == 1.3.0 
  pip install pandas == 2.0.3 
  pip install numpy == 1.24.4
  pip install scipy == 1.5.4
  
* Pre-trained model ProstT5
     
  Download the pre-trained model ProstT5 from https://github.com/mheinzinger/ProstT5, which should is stored in the folder ¡°softwares/Rostlab¡±.
  
  Install according to the official tutorials£º
  
  pip install torch
  pip install transformers
  pip install sentencepiece

* DSSP

  Download the software from https://swift.cmbi.umcn.nl/gv/dssp, it has been given in the folder ¡°softwares/dssp¡±.

## Step 2 Prediction

* Place the PDB file of 5NRM in 'example/PDB'. 

* Run the following commands:
  

  neighbors=25
  
  pdbname='5NRM'
  
  chain='A'
  
  PDB_path='example/PDB/'
  
  Type='protein' 
  
  testdata='5NRM_A.txt'
  
  echo  Begin to extract features!
  
  cd softwares
  
  python ../codes/dssp.py $PDB_path $Type
  
  cd ..
  
  python ./codes/get_dssp.py $testdata $Type
  
  python ./codes/prostT5.py $Type $testdata
      
  python ./codes/consider_neighbor_nodes_edges.py $PDB_path 4$Type $testdata $neighbors
  
  echo The calculation has been completed !
  
  echo Begin to predicte binding sites !
  
  python predicted.py $pdbname $chain $Type
         

Or directly run the following command:

  ./run.sh
 
The finally output is shown in "./results/5NRM/predected_result.txt".

The predicted binding sites label: 000000000000000000000000000000000111000000000000000000000010011111111000001011101111000000000000000000000000000000000001000001010000000000000


## Training GraphPBSP

Follow the steps in the prediction section to extract features.

After extracting the features, to train GraphPBSP run the following command. 

  cd GraphPBSP/utils
  
  python train.py 



## Help

For any questions, please contact us by chunhuali@bjut.edu.cn.

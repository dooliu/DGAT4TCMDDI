# DGAT4TCMDDI

<h2>A Dual-GAT methods for DDI prediction of TCM</h2>

The Dual Graph Attention Network (DGAT) is a novel framework designed to predict Traditional Chinese Medicine drug-drug interactions (TCMDDI) by extracting key structural features of active molecules within herbal ingredients. DGAT leverages graph-based representations of chemical molecules and employs an attention mechanism to extract deep structural features, enabling effective prediction of TCMDDI through the capture of spatial structural relationships among different compounds. Furthermore, a comprehensive dataset encompassing three distinct categories of herbal ingredients was constructed, informed by traditional TCM principles, to support the development and validation of DGAT.

<h2>Installation</h2>

DGAT should work on linux platform.

MacOS X and windows are not tested.

It is recommended to create a conda environment for DGL-LifeSci with for example:

```txt
conda create -n DGAT python=3.7
```

and you can install other package by next order:

```txt
pip3 install -r requirements.txt
```

<h2>Example of use</h2>

Train DGAT model on data1:

```txt
python main -mo DGAT -a attentivefp -b attentivefp -c ./data/data1.csv -sc1 a -sc2 b -t labels -s cross -scl 3 -me mae -n 105
```

user different dataset as train set and test set:

```txt
python train_cross_dataset -mo DGAT -a attentivefp -b attentivefp -c_1 ./data/data2.csv -c_2 ./data/data3.csv  -sc1 a -sc2 b -t labels -s cross -scl 3 -me mae -n 1005
```
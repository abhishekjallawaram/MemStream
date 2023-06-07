# MemStream / EDEM (Encoder-Decoder-Encoder Memory-based streaming anomaly detection approach)


This repository implements MemStream, a memory-based streaming anomaly detection approach, and introduces EDEM, an extended version of MemStream.

## Introduction

MemStream is a memory-based streaming anomaly detection method described in the paper "MemStream: Memory-Based Streaming Anomaly Detection" by Siddharth Bhatia et al. It utilizes an encoder-decoder-encoder memory architecture to detect anomalies in streaming data.

EDEM (Encoder-Decoder-Encoder Memory) is an extension of MemStream, which enhances its anomaly detection capabilities.

## Usage

* To use the EDEM approach, replace 'memstream.py' with 'memstream-ede.py' in the code.
* Run the '*.sh' files for pre-processing vs sampling comparison.

## Command line options
  * `--dataset`: The dataset to be used for training. Choices 'NSL', 'KDD', 'UNSW', 'DOS'. (default 'NSL')
  * `--beta`: The threshold beta to be used. (default: 0.1)
  * `--memlen`: The size of the Memory Module (default: 2048)
  * `--dev`: Pytorch device to be used for training like "cpu", "cuda:0" etc. (default: 'cuda:0')
  * `--lr`: Learning rate (default: 0.01)
  * `--epochs`: Number of epochs (default: 5000)
  * '--RQ1' : Hypertune best metrics : Default False
  * '--RQ2' : Effect of Activation Functions : Default False
  * '--RQ3' : Memory Poisoning Prevention Analysis : Default False
  * '--RQ4' : Concept Dript Analysis : Default False
  * '--RQ5' : Impact of Memory : Default False
  * '--sp' : Sampling method : Default : rand (Random seed) / str (Stratified sampling) / ovr (Random Oversampling) / sm (SMOTE) / ens_m (Ensemble : only for EDEM)

## Demo

1. KDDCUP99: Run `python3 memstream.py --dataset KDD --beta 1 --memlen 256`
2. NSL-KDD: Run `python3 memstream.py --dataset NSL --beta 0.1 --memlen 2048`
3. UNSW-NB 15: Run `python3 memstream.py --dataset UNSW --beta 0.1 --memlen 2048`
4. CICIDS-DoS: Run `python3 memstream.py --dataset DOS --beta 0.1 --memlen 2048`
5. SYN: Run `python3 memstream.py --dataset SYN --beta 1 --memlen 16`
6. Ionosphere: Run `python3 memstream.py --dataset ionosphere --beta 0.001 --memlen 4`
7. Cardiotocography: Run `python3 memstream.py --dataset cardio --beta 1 --memlen 64`
8. Statlog Landsat Satellite: Run `python3 memstream.py --dataset statlog --beta 0.01 --memlen 32`
9. Satimage-2: Run `python3 memstream.py --dataset satimage-2 --beta 10 --memlen 256`
10. Mammography: Run `python3 memstream.py --dataset mammography --beta 0.1 --memlen 128`
11. Pima Indians Diabetes: Run `python3 memstream.py --dataset pima --beta 0.001 --memlen 64`
12. Covertype: Run `python3 memstream.py --dataset cover --beta 0.0001 --memlen 2048`


## Research Questions
To execute different research questions, add the corresponding --RQ flag with a value of True.

RQ1: Hypertune best metrics
RQ2: Effect of Activation Functions
RQ3: Memory Poisoning Prevention Analysis
RQ4: Concept Dript Analysis : Default False
RQ5: Impact of Memory : Default False
sp: Sampling method : Default : rand (Random seed) / str (Stratified sampling) / ovr (Random Oversampling) / sm (SMOTE) / ens_m (Ensemble : only for EDEM)



## Input file format
MemStream expects the input multi-aspect record stream to be stored in a contains `,` separated file.

## Datasets
Processed Datasets can be downloaded from [here](https://drive.google.com/file/d/1najJ13lSwPpB9lkGk-6ZzgAV65m8ux7Y/view?usp=sharing). Please unzip and place the files in the data folder of the repository.

1. [KDDCUP99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
2. [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
3. [UNSW-NB 15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
4. [CICIDS-DoS](https://www.unb.ca/cic/datasets/ids-2018.html)
5. Synthetic Dataset (Introduced in paper)
6. [Ionosphere](https://archive.ics.uci.edu/ml/index.php)
7. [Cardiotocography](https://archive.ics.uci.edu/ml/index.php)
8. [Statlog Landsat Satellite](https://archive.ics.uci.edu/ml/index.php)
9. [Satimage-2](http://odds.cs.stonybrook.edu)
10. [Mammography](http://odds.cs.stonybrook.edu)
11. [Pima Indians Diabetes](https://archive.ics.uci.edu/ml/index.php)
12. [Covertype](https://archive.ics.uci.edu/ml/index.php)



## general areas

1) GPU scheduling (DRA or some better support for nvidia plugin)
2) AI related scalers (metrics)
3) Glue - for operating the auto-scaleable models
4) Adding and releasing GPU enabled nodes dynamically to k8s cluster
5) Multi-cluster (overflow to a cluster that have GPUs 'attached' say spotinstances that are cheaper)
6) Maybe introduce a new CRD on the KEDA level that will support heterogeneous settings across replicas (currently all replicas of a single deployment or statefulset are identical)


## Types of 'AIs'
- standard methods - decision trees, PCA, naive bayes, K-NN, etc.
- neural nets - different topologies, RNNs for text for instance
- supervised vs unsupervised
- classification vs regression vs clustering vs GANs vs prediction vs anomaly detection vs ..

## Types of popular AI workloads:
- generative AI (neural nets)
    - text, image, sound
- recommenders (collaborative filtering or neural nets)
- data exploration/experiments + notebooks


## Some popular AI 'frameworks'
- TensorFlow
- PyTorch
- MLFlow
- Spark
- Weights & Biases
- Keras
- ..

### Before training
Data pre-processing & feature engineering - defining the and encoding the training sample into input of the algorithm, what fields may make sense for the model, introducing a new ones.

There might be a data processing jobs performing data normalization and what not. Long running batch jobs that might have dependencies among them forming a DAG.

Hyperparameter tunning - what topology is the best one for the neural net and given task? In general what model params should we use and froze them for the training part.

## Training
### data
Training often requires providing a larg amount of data samples to the training algorithm (S3 compatible storage). Then model itself can be quite a beast. Format might be framework specific: HDF5 for tensorflow, '.pth' for PyTorch (checkpointing is also a related topic, these algorithms have a state and may need to save this state -> harder to scale).

### algorithm
For all kind of neural nets, backpropagation algorithm is used for training them. Training of neural nets can be parallelized using:

### - Data parellelism
Training data spread across multiple training units and training the nets with the same topology independly. At the end the weights are averaged across all multiple training jobs.

### - Model parellelism (also refered to as Tensor parallelism)
Splitting a single neural network across many devices, each responsible for computing a portion of the model's operations. This is more suitable for training LLMs from scratch. Folks who do this often have their own solutions (Google, Meta, Apple..).

## Inference

Operating the trained models requires less resources, but GPUs might still be needed. Because the feed forward mode of nnet computation is still buch of matrix/tensor multiplications in the end.

It highly depends on a specific use-case, but for inference mode the time of the overall computation might be much smaller in case user is calculating an output for just one or small amount of inputs. Compared to the training where it can run for hours/days.

If the granularity of requests is small enough, we may consider even some serverless use-cases here. A task can borrow a fraction of GPU for some short amount of time to calculate the result and release it back to the pool. With high number of incomming requests (captured as Prometheus HTTP metric), we can make use of KEDA to increase the number of replicas for given workload.


## Transfer learning

Idea: take some already trained model (mostly neural net) and assume the inner layers of the network can make sense of edges, colors, body features (example for image recognition/classification use-case), then remove last or couple of last layers that participate in giving the final result. Plug in some new layer(s) with randomly initialized weights and train this resulting network again to a custom use-case.

The layers from the original network also can have frozen weights so that we reduce the search space for model parameters to significantly smaller one that is much easier to train.

### Applications
- style transfer - train the network with just one image to be able to mimic that style (represented as Gram matrix) of art of that image

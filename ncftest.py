"""
Toy project to play with NCF
Using NCF implementation from https://github.com/microsoft/recommenders
Parts of the code are copied from  https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/ncf_deep_dive.ipynb
"""


import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools as it
import json
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.utils.constants import SEED as DEFAULT_SEED

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


#  top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Constant model parameters

# Note, recommenders uses a checkpoint system for doing training in parts, but model.fit()
# doesn't return the training loss so without forking the library we can't use adaptive 
# epochs where the code automatically detects when it has converged enough.
# Therefore, we use constant EPOCHS
EPOCHS = 100

BATCH_SIZE = 256

SEED = DEFAULT_SEED  # Set None for non-deterministic results

train_file = "./train.csv"
test_file = "./test.csv"
leave_one_out_test_file = "./leave_one_out_test.csv"

def load():
  # Run once to save the data locally
  df = movielens.load_pandas_df(
      size=MOVIELENS_DATA_SIZE,
      header=["userID", "itemID", "rating", "timestamp"]
  )
  
  train, test = python_chrono_split(df, 0.75)
  
  test = test[test["userID"].isin(train["userID"].unique())]
  test = test[test["itemID"].isin(train["itemID"].unique())]
  
  leave_one_out_test = test.groupby("userID").last().reset_index()
  train.to_csv(train_file, index=False)
  test.to_csv(test_file, index=False)
  leave_one_out_test.to_csv(leave_one_out_test_file, index=False)

def train(runID, newparams, data, overwrite=False):
  if os.path.exists(".pretrain/%s/checkpoint"%runID) and not overwrite:
    print(f"runid {runID} already trained, no need to re-train")
    return # no need to re-train

  print()
  print(f" == Start training model {runID} == ")
  print()

  # set the default runparams dict by assuming default value unless defined in newparams dict
  runparams = {
    "n_users": data.n_users, 
    "n_items": data.n_items,
    "model_type": "NCF",
    "n_factors": 4,
    "layer_sizes": [16,8,4],
    "n_epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": 1e-3,
    "verbose": 0,
    "seed": SEED
  }

  # override defaults by user-specified ones
  for key, val in newparams.items():
    runparams[key] = val

  model = NCF (**runparams)

  # define which NCF attributes to save to metadata file
  keys_to_save = ["n_users", "n_items", "model_type", "n_factors", "layer_sizes", "n_epochs", "batch_size", "learning_rate", "seed"]
  rundata = {key: model.__dict__[key] for key in keys_to_save}

  # save new entry in the metadata file
  if not os.path.exists("metadata.json"):
    metadata = {}
  else:
    with open("metadata.json") as fp:
      metadata = json.load(fp)
    
  # add current run
  metadata[runID] = rundata

  # write back
  with open("metadata.json", "w") as fp:
    json.dump(metadata, fp, indent=4, sort_keys=True)

  with Timer() as train_time:
    model.fit(data)

  print(f"{runID} took {train_time.interval} seconds for training")
  # save training data
  model.save(dir_name=".pretrain/%s"%runID)

def predict(runID, data, runparams = None):
    # load model params

    pfx = "metadata"

    # if runparams not passed, load from metadata file
    if runparams is None:
      print("load metadata")
      with open("metadata.json") as fp:
        metadata = json.load(fp)
        if runID not in metadata:
          sys.exit(f"No run parameters for {runID} exists in metadata.json")

        runparams = metadata[runID]

    print("init model")
    model = NCF (**runparams)

    model.load(neumf_dir=".pretrain/%s"%runID)

    # recommenders library has a bug where loading pre-trained weights makes model.predict() crash
    # unless we manually set the four attributes below
    model.user2id = data.user2id
    model.item2id = data.item2id
    model.id2user = data.id2user
    model.id2item = data.id2item

    print()
    print("start predictions for %s"%runID)
    eval_ndcg, eval_hr = evaluate(model, data)
    print("done")

    
    return eval_hr, eval_ndcg

def evaluate(model, data):
    k = TOP_K
    
    ndcgs = []
    hit_ratio = []
    
    for b in data.test_loader():
        user_input, item_input, labels = b
        output = model.predict(user_input, item_input, is_list=True)
    
        output = np.squeeze(output)
        rank = sum(output >= output[0])
        if rank <= k:
            ndcgs.append(1 / np.log(rank + 1))
            hit_ratio.append(1)
        else:
            ndcgs.append(0)
            hit_ratio.append(0)
    
    eval_ndcg = np.mean(ndcgs)
    eval_hr = np.mean(hit_ratio)
    return eval_ndcg, eval_hr

def modelParameterSearch():
  # n_factors to try
  nfactorarr = [4, 8,16, 32, 64]
  # number of layers for MLP to try
  numlayers = [3, 4, 5]

  layersizearr = []
  paramspace = []
  # iterate over different depths and n_factors to create layer_sizes that follow the NCF formula
  for depth in numlayers:
    for nf in nfactorarr:
      ini = nf
      layers = [ini * 2**d for d in range(depth)] # build the layer size arr
      layersizearr.append(layers[::-1]) 

  for layers in layersizearr:
    paramspace.append((layers[-1], layers))

  data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=False)

  for n_factor, layer_sizes in paramspace:
    model_type = "NeuMF"
    layerstr = f"{tuple(layer_sizes)}".replace(" ", "") # remove spaces from the layersize tuple

    # runID is just a string with no practical purpose other than to be an ID, so it could also just be a running number or a random string
    runID = f"{model_type}-{n_factor}factors-{EPOCHS}epochs-{layerstr}layers"

    newparams = {"model_type": model_type, "n_factors": n_factor, "layer_sizes": layer_sizes}
    train(runID, newparams, data)


def modelParameterEvaluation():
  data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=False)

  # get all runIDs we have trained
  metadata = {}
  with open("metadata.json") as fp:
    metadata = json.load(fp)

  scores = {}
  for runID in metadata.keys():
    runparams = metadata[runID]
    paramtuple = (runparams["n_factors"],tuple(runparams["layer_sizes"]))
    hr, ndcg = predict(runID, data, runparams)
    scores[paramtuple] = ndcg

  scores = sorted(scores.items(), key=lambda x: x[1])[::-1]

  print("Model parameters ordered by scores")
  for keys, score in scores:
    print(f"params {keys}, score {score:.4f}")


def main():
  # only needs to be called once to initialize local data files
  #load() 

  # iterate over model parameters, saving the model hyperparameters together with trained weights for later use
  modelParameterSearch()  

  # iterate over all pre-trained models to evaluate their performance
  modelParameterEvaluation()

if __name__ == "__main__":
  main()

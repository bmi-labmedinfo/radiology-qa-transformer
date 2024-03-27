import random
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict, ClassLabel, concatenate_datasets
from sklearn.model_selection import StratifiedKFold

from configuration import Config
config = Config()

class Dataset_creation:
   def __init__(self,undersampling=False):

     # FACTUAL QA DATASET
     f_raw_dataset = load_dataset("json", data_files=config.INPUT_FILE_PATH + "radiology_factual.json", field="data")
     self.f_raw_dataset = f_raw_dataset["train"].add_column('type', ["factual"] * len(f_raw_dataset["train"]))
     print("Factual Dataset:", self.f_raw_dataset)

     # FREE-TEXT QA DATASET
     t_raw_dataset = load_dataset("json", data_files=config.INPUT_FILE_PATH + "radiology_free_text.json", field="data")
     self.t_raw_dataset = t_raw_dataset["train"].add_column('type', ["free-text"] * len(t_raw_dataset["train"]))
     print("Free-text Dataset:", self.t_raw_dataset)

     # MULTICHOICE QA DATASET
     mc_raw_dataset = load_dataset("json", data_files=config.INPUT_FILE_PATH + "radiology_multichoice.json",field="data")
     mc_raw_dataset = mc_raw_dataset.remove_columns("subject")
     mc_raw_dataset = mc_raw_dataset["train"].add_column('type', ["multichoice"] * len(mc_raw_dataset["train"]))
     # if the model is uncased, lowercase all the options
     self.mc_raw_dataset = mc_raw_dataset.map(lambda example: {"options": [o.lower() for o in example["options"]]})
     if undersampling:
        self.mc_raw_dataset= self.under_sampling(self.mc_raw_dataset,id_ans="2")
     print("Multichoice Dataset:", self.mc_raw_dataset)

   def under_sampling(self,data,id_ans):
      #id) 1:true/false 2: 1/2+ 3:limited/advanced
      idx_0=[]
      idx_station=[]
      other_mc=[]
      undersampled_idx=[]
      for i in np.arange(len(data)):
         if data["id"][i][-1] == id_ans and data["correct"][i] == 1:
           idx_station.append(i)
         elif data["id"][i][-1] == id_ans and data["correct"][i] == 0:
           idx_0.append(i)
         else: other_mc.append(i)
      for idx in np.arange(len(idx_0)):
        undersampled_idx.append(random.choice(idx_station))

      new_data_1 = Dataset.from_dict(data[undersampled_idx])
      data_0 = Dataset.from_dict(data[idx_0])
      mc_data = Dataset.from_dict(data[other_mc])
      print(len(new_data_1),len(data_0))
      dataset= concatenate_datasets([data_0,new_data_1,mc_data])

      return dataset

   def cross_validation_split(self,k_fold=5):
     # to obtain k indipendent test set
     folds = StratifiedKFold(n_splits=config.K_FOLD)
     dataset = concatenate_datasets([self.f_raw_dataset, self.t_raw_dataset, self.mc_raw_dataset])
     dataset = dataset.shuffle(seed=1)
     splits = folds.split(dataset, dataset["type"])
     test_set = []
     train_set = []

     for i, (train_index, test_index) in enumerate(splits):
       ft = 0; mc = 0; f = 0
       for list_type in dataset[train_index]["type"]:
         if list_type == "free-text": ft += 1
         if list_type == "factual": f += 1
         else: mc += 1
       test_set.append(test_index)
       train_set.append(train_index)

     test_index_CV = pd.DataFrame(data=zip(test_set[0], test_set[1], test_set[2], test_set[3], test_set[4]))
     train_index_CV = pd.DataFrame(data=zip(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4]))
     dataset_CV = pd.DataFrame(data=dataset, columns=dataset.features)
     test_index_CV.to_csv(config.INPUT_FILE_PATH + "test_index_CV.csv")
     train_index_CV.to_csv(config.INPUT_FILE_PATH + "train_index_CV.csv")
     dataset_CV.to_csv(config.INPUT_FILE_PATH + "dataset_CV.csv")
     print("> 5 CROSS-VALIDATION: COMPLETED!")

def context_token_length(type,context,str_question,*str_options):
  tokenizer = config.TOKENIZER
  if type == "factual" or type == "free-text": question_len = len(tokenizer.tokenize( " Domanda: " + str(str_question)))
  else: question_len = len(tokenizer.tokenize("Opzioni: " + str(str_options)  + " . Domanda: " + str_question))
  output_len = config.MAX_INPUT_TOKENS - question_len
  context_token = tokenizer.tokenize(context)
  tronc_context = context_token[0:output_len]
  new_context = tokenizer.convert_tokens_to_ids(tronc_context)
  new_context = tokenizer.decode(new_context)
  return new_context

def input_output_builder(input_type, question, answer, context, options):
  prefix_separator = " "
  if question[-1] != "?": str_question = question + "?"
  else: str_question = question
  new_context  = context

  if input_type == "factual" or input_type == "free-text":
    input_text = new_context + prefix_separator + " Domanda: " + str_question
    output_text = answer
  else: # (input_type == multichoice)
    option_separator = ")"
    characters = "'[]"
    for x in range(len(characters)):
      options = options.replace(characters[x], "")
    opt = options.split(",")
    correct_str = opt[int(answer[0])].strip()
    str_options = " ".join([str(i+1) + option_separator + " " + opt[i] for i in range(len(opt))])
    if question[-1] != "?": str_question = question + "?"
    else: str_question = question
    new_context = context
    input_text = new_context + prefix_separator + "Opzioni: " + str_options + prefix_separator + " . Domanda: " + str_question
    output_text = correct_str

  return input_text, output_text

def io_wrapper(example):
    if example["type"] == "free-text" or example["type"] == "factual":
      example["input_text"],example["output_text"] = input_output_builder(input_type=example["type"], question=example["question"], answer=example["answer"], context=example["context"], options=list())
    else:
      example["input_text"],example["output_text"] = input_output_builder(input_type=example["type"], question=example["question"], answer=example["correct"], context=example["context"], options=example["options"])
    return example

# Splitting in batches for ex-post combination
def splitting_batches(dataset, tokenizer):
  batches_single_entry = []

  for entry in dataset:
    if entry["type"] == "factual" or entry["type"] == "free-text":
      question_len = len(tokenizer.tokenize(" Domanda: " + str(entry["question"])))
    else:
      option_separator = ")"
      characters = "'[]"
      for x in range(len(characters)):
        entry["options"] = entry["options"].replace(characters[x], "")
      opt = entry["options"].split(",")
      correct_str = opt[int(entry["correct"][0])]
      str_options = " ".join([str(i+1) + option_separator + " " + opt[i] for i in range(len(opt))])
      question_len = len(tokenizer.tokenize("Opzioni: " + str(str_options)  + " . Domanda: " + str(entry["question"])))

    output_len = config.MAX_INPUT_TOKENS - question_len
    context_tokens = tokenizer.tokenize(entry["context"])
    context_tokens_n = len(context_tokens)
    n = (int(context_tokens_n/output_len) + (context_tokens_n % output_len>0))
    start_pos_batch = 0
    for ni in np.arange(n):
      if start_pos_batch+output_len <= context_tokens_n:
        tronc_context = context_tokens[start_pos_batch:start_pos_batch+output_len]
      else: tronc_context = context_tokens[start_pos_batch:context_tokens_n]
      new_context = tokenizer.convert_tokens_to_ids(tronc_context)
      new_context = tokenizer.decode(new_context)
      start_pos_batch = start_pos_batch+output_len

      if entry["type"] == "factual" or entry["type"] =="free-text":
        element = {"batches":n, "id":entry["id"], "type":entry["type"],"context":new_context, "question":entry["question"], "answer":entry["answer"], "options":"None", "correct":"None"}
      else:
        element = {"batches":n, "id":entry["id"], "type":entry["type"],"context":new_context, "question":entry["question"], "answer": "None", "options":opt, "correct":entry["correct"]}
      batches_single_entry.append(element)

  dataset_batches = pd.DataFrame([], columns= ["batches","id","type","context","question","answer","options","correct"])
  for i in np.arange(len(batches_single_entry)):
    dataset_batches.loc[i] = list(batches_single_entry[i].values())
  datasetDict = Dataset.from_pandas(dataset_batches.astype(str))

  return datasetDict

# k-fold CV splitting for ex-post combination
def epc_k_fold_splitting(k_fold, seed, tokenizer):
  test_index_CV = pd.read_csv(config.INPUT_FILE_PATH + "test_index_CV.csv")
  train_index_CV = pd.read_csv(config.INPUT_FILE_PATH + "train_index_CV.csv")
  dataset_CV = pd.read_csv(config.INPUT_FILE_PATH + "dataset_CV.csv")
  test_index_CV = test_index_CV.drop("Unnamed: 0", axis='columns')
  train_index_CV = train_index_CV.drop("Unnamed: 0", axis='columns')
  dataset_CV = dataset_CV.drop("Unnamed: 0", axis='columns')

  dataset_CV = dataset_CV.astype(str)

  test_set = Dataset.from_pandas(pd.DataFrame(dataset_CV.iloc[test_index_CV.iloc[:,k_fold],:]))
  test_set = test_set.remove_columns("__index_level_0__")
  train_dev_set = Dataset.from_pandas(pd.DataFrame(dataset_CV.iloc[train_index_CV.iloc[:,k_fold],:]))
  train_dev_set = train_dev_set.remove_columns("__index_level_0__")
  train_dev_set = train_dev_set.add_column("type_encoding", train_dev_set["type"])
  type_code = ClassLabel(num_classes = 3, names=['factual', 'free-text', 'multichoice'])
  train_dev_set = train_dev_set.cast_column("type_encoding", type_code)
  temp_split = train_dev_set.train_test_split(stratify_by_column="type_encoding", seed= seed, test_size=0.2)
  train_set = temp_split["train"]
  dev_set = temp_split["test"]

  train_batch_set = splitting_batches(train_set,tokenizer)
  test_batch_set = splitting_batches(test_set,tokenizer)
  dev_batch_set = splitting_batches(dev_set,tokenizer)

  train_batch_set = train_batch_set.map(io_wrapper, remove_columns = ["context","question","answer","options","correct","__index_level_0__"])
  dev_batch_set = dev_batch_set.map(io_wrapper, remove_columns = ["context","question","answer","options","correct","__index_level_0__"])
  test_batch_set = test_batch_set.map(io_wrapper, remove_columns = ["context","question","answer","options","correct","__index_level_0__"])

  multitask_dataset = DatasetDict({"train": train_batch_set, "test": test_batch_set, "dev": dev_batch_set})
  multitask_dataset["test"] = multitask_dataset["test"].shuffle(seed)
  multitask_dataset["train"] = multitask_dataset["train"].shuffle(seed)
  multitask_dataset["dev"] = multitask_dataset["dev"].shuffle(seed)
  print("> EPC MULTITASK_DATASET: CREATED")
  return multitask_dataset

# Considering the 1st batch for batch-truncation
def bt_splitting(dataset,tokenizer):
  new_dataset = []

  for entry in dataset:
    if entry["type"] == "factual" or entry["type"] == "free-text":
      question_len = len(tokenizer.tokenize(" Domanda: " + str(entry["question"])))
    else:
      option_separator = ")"
      characters = "'[]"
      for x in range(len(characters)):
        entry["options"] = entry["options"].replace(characters[x], "")
      opt = entry["options"].split(",")
      correct_str = opt[int(entry["correct"][0])]
      str_options = " ".join([str(i + 1) + option_separator + " " + opt[i] for i in range(len(opt))])
      question_len = len(tokenizer.tokenize("Opzioni: " + str(str_options) + " . Domanda: " + str(entry["question"])))

    output_len = config.MAX_INPUT_TOKENS - question_len
    context_tokens = tokenizer.tokenize(entry["context"])
    tronc_context = context_tokens[:output_len]

    new_context = tokenizer.convert_tokens_to_ids(tronc_context)
    dec_context = tokenizer.decode(new_context)


    if entry["type"] == "factual" or entry["type"] == "free-text":
        element = {"id": entry["id"], "type": entry["type"], "context": dec_context,
                   "question": entry["question"], "answer": entry["answer"], "options": "None", "correct": "None"}
    else:
        element = {"id": entry["id"], "type": entry["type"], "context": dec_context,
                   "question": entry["question"], "answer": "None", "options": opt, "correct": entry["correct"]}
    new_dataset.append(element)

  dataset_batches = pd.DataFrame([], columns=["id", "type", "context", "question", "answer", "options",
                                              "correct"])
  for i in np.arange(len(new_dataset)):
    dataset_batches.loc[i] = list(new_dataset[i].values())
  datasetDict = Dataset.from_pandas(dataset_batches.astype(str))

  return datasetDict

# k-fold CV splitting for batch-truncation
def bt_k_fold_splitting(k_fold, seed):
  data = Dataset_creation(undersampling=True)
  data.cross_validation_split(k_fold= k_fold)
  test_index_CV = pd.read_csv(config.INPUT_FILE_PATH + "test_index_CV.csv")
  train_index_CV = pd.read_csv(config.INPUT_FILE_PATH + "train_index_CV.csv")
  dataset_CV = pd.read_csv(config.INPUT_FILE_PATH + "dataset_CV.csv")
  test_index_CV = test_index_CV.drop("Unnamed: 0", axis='columns')
  train_index_CV = train_index_CV.drop("Unnamed: 0", axis='columns')
  dataset_CV = dataset_CV.drop("Unnamed: 0", axis='columns')

  dataset_CV = dataset_CV.astype(str)

  test_set = Dataset.from_pandas(pd.DataFrame(dataset_CV.iloc[test_index_CV.iloc[:, k_fold], :]))
  test_set = test_set.remove_columns("__index_level_0__")
  train_dev_set = Dataset.from_pandas(pd.DataFrame(dataset_CV.iloc[train_index_CV.iloc[:, k_fold], :]))
  train_dev_set = train_dev_set.remove_columns("__index_level_0__")
  train_dev_set = train_dev_set.add_column("type_encoding", train_dev_set["type"])
  type_code = ClassLabel(num_classes=3, names=['factual', 'free-text', 'multichoice'])
  train_dev_set = train_dev_set.cast_column("type_encoding", type_code)
  temp_split = train_dev_set.train_test_split(stratify_by_column="type_encoding", seed=seed, test_size=0.2)
  train_set = temp_split["train"]
  dev_set = temp_split["test"]

  train_split_set = bt_splitting(train_set, config.TOKENIZER)
  test_split_set = bt_splitting(test_set, config.TOKENIZER)
  dev_split_set = bt_splitting(dev_set,config.TOKENIZER)

  train_set = train_split_set.map(io_wrapper,remove_columns=["context", "question", "answer", "options", "correct"])
  dev_set = dev_split_set.map(io_wrapper,remove_columns=["context", "question", "answer", "options", "correct"])
  test_set = test_split_set.map(io_wrapper, remove_columns=["context", "question", "answer", "options", "correct"])

  multitask_dataset = DatasetDict({"train": train_set, "test": test_set, "dev": dev_set})
  multitask_dataset["test"] = multitask_dataset["test"].shuffle(seed)
  multitask_dataset["train"] = multitask_dataset["train"].shuffle(seed)
  multitask_dataset["dev"] = multitask_dataset["dev"].shuffle(seed)
  print("> BT MULTITASK_DATASET: CREATED")
  return multitask_dataset